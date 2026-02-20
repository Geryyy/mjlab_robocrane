import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin
import casadi as ca
from casadi import *
from scipy.spatial.transform import Rotation
from importlib.resources import files # for accessing casadi data files
import cranebrain.utils.pinutil as pu
from cranebrain.common.load_model import load_mujoco_model, load_pinocchio_model, get_gripper_point_frame_id
from cranebrain.utils.util import forward_kinematics, pos_rpy_to_homtrans, cubic_joint_interpolation, plot_trajectory
from colorama import Fore, Style
import mujoco
import mujoco.viewer
from cranebrain.common.sample_workspace import sample_cylindrical_workspace_segment, visualize_workspace

def Rx(q):
    rotmat = np.array([[1, 0, 0],
                        [0, cos(q), -sin(q)],
                        [0, sin(q), cos(q)]])
    return rotmat

def Ry(q):
    rotmat = np.array([[cos(q), 0, sin(q)],
                        [0, 1, 0],
                        [-sin(q), 0, cos(q)]])
    return rotmat

def Rz(q):
    rotmat = np.array([[cos(q), -sin(q), 0],
                        [sin(q), cos(q), 0],
                        [0, 0, 1]])
    return rotmat

class SteadyState: 

    def __init__(self, pin_model, pin_data, tool_frame_id, tol=1e-6, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter

        self.pin_model = pin_model
        self.pin_data = pin_data
        #print("pin_model.type: ", type(pin_model))
        self.cpin_model = cpin.Model(pin_model)
        self.cpin_data = self.cpin_model.createData()

        self.tool_frame_id = tool_frame_id

        # get transformation from tool frame to last joint frame 
        q_neutral = pin.neutral(pin_model)
        pin.forwardKinematics(pin_model, pin_data, q_neutral)
        pin.updateFramePlacements(pin_model, pin_data)
        # pin.framesForwardKinematics(pin_model, pin_data, q_neutral)

        # intialize casadi functions
        self.q_in = ca.SX.sym('q_in', 9,1)
        self.q_u = self.q_in[7:]
        self.q_a = self.q_in[:7]
        
        self.J = ca.jacobian(self._gravitational_forces_ca(self.q_in), self.q_u)
        self.F_gg_u = ca.Function('F_gg_u', [self.q_in], [self._gravitational_forces_ca(self.q_in)], ['q0'], ['qpp_u'])
        self.F_J = ca.Function('F_J', [self.q_in], [self.J], ['q0'], ['J'])

        self.ik_solver = self.optProblemGeneratorInverseKinematics()

        self.cost_weight_parameters = np.array([1, 1, 0.001])

        self.q_ub = np.array([170, 120, 170, 120, 170, 120, 175, 35, 35]) * np.pi / 180
        self.q_lb = -self.q_ub

        self.lbg = np.zeros(2)
        self.ubg = np.zeros(2)

    def compute_frot(self, q, Hd):
        pd = Hd[:3, 3]
        Rd = Hd[:3, :3]
        pin.forwardKinematics(self.pin_model, self.pin_data, q.T)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        oMdes = pin.SE3(Rd, pd.T)
        oMact = self.pin_data.oMf[self.tool_frame_id]
        # iMd = oMdes.actInv(oMact)
        oMerr = oMdes * oMact.inverse()
        # oRerr = oMdes.rotation @ oMact.rotation.T

        print("Rd: ", Rd)
        print("Ract: ", oMact.rotation)

        R_ed = oMerr.rotation
        frot = 2*(3-np.trace(R_ed))
        return frot


    def check_limits(self, q):
        q_lower_bound = np.array(self.q_lb).flatten()
        q_upper_bound = np.array(self.q_ub).flatten()

        if len(q) != len(q_lower_bound) or len(q) != len(q_upper_bound):
            print(f"Error: Input joint configuration 'q' has length {len(q)}, "
                f"but joint limits have length {len(q_lower_bound)} and {len(q_upper_bound)}.")
            return False

        within_lower_bounds = np.all(q >= q_lower_bound)
        within_upper_bounds = np.all(q <= q_upper_bound)

        return within_lower_bounds and within_upper_bounds
    

    def optProblemGeneratorInverseKinematics(self):
        dof = self.pin_model.nq
        q = SX.sym('q', 1, dof)
        pd = SX.sym('pd', 1, 3) # goal / desired position
        Rd = SX.sym('Rd', 3, 3)
        q_init = SX.sym('q_init', 1, dof)
        cost_weight_parameters = SX.sym('cost_weight_parameters', 1, 3)

        cpin.forwardKinematics(self.cpin_model, self.cpin_data, q.T)
        cpin.updateFramePlacements(self.cpin_model, self.cpin_data)

        oMdes = cpin.SE3(Rd, pd.T)
        R_des = oMdes.rotation
        oMact = self.cpin_data.oMf[self.tool_frame_id]
        R_act = oMact.rotation
        # iMd = oMdes.actInv(oMact)
        oMerr = oMact.inverse() * oMdes
        epos = oMerr.translation

        
        R_ed = oMerr.rotation

        # yaw_act = atan2(R_act[1,0], R_act[0,0])
        # yaw_des = atan2(R_des[1,0], R_des[0,0])
        # yaw_err = yaw_act - yaw_des
        # R_err = oMerr.rotation
        # yaw_err = atan2(R_err[1,0], R_err[0,0])
        # general orientation error
        frot = cost_weight_parameters[0] * 2*(3-trace(R_ed))
        # frot = cost_weight_parameters[0] * yaw_err*yaw_err
        fpos = cost_weight_parameters[1] * dot(epos, epos)

        q_err = q - q_init
        fq = cost_weight_parameters[2] * dot(q_err, q_err)
        f = fpos + frot + fq

        # constraints
        g = cpin.computeGeneralizedGravity(self.cpin_model, self.cpin_data, q.T)[7:]        

        # optimization variables
        opt_var = vertcat(q[:])
        # params = horzcat(cost_weight_parameters, pd, Rd[:].T, q0)
        params = horzcat(cost_weight_parameters, pd, Rd[:].T, q_init)

        print('opt_var shape: ' + str(opt_var.shape))
        print('params shape: ' + str(params.shape))

        nlp = {'x' : opt_var,
                'f' : f,
                'g' : g,
                'p' : params}

        ipopt_options = {'tol' : 1e-3,
                        'max_iter' : 50,
                        'linear_solver' : 'ma57',
                        # 'linear_system_scaling' : 'none',
                        # 'ma57_automatic_scaling' : 'no',
                        # 'ma57_pre_alloc' : 10,
                        # 'mu_strategy' : 'monotone',
                        # 'fixed_mu_oracle' : 'probing',
                        # 'expect_infeasible_problem' : 'no',
                        # 'print_info_string' : 'no',
                        # 'fast_step_computation' : 'yes',
                        'print_level' : 0} # 5
        nlp_options = {'ipopt' : ipopt_options,
                        'print_time' : 0}
        F = nlpsol('F', 'ipopt', nlp, nlp_options)

        # export code for compiling
        codegen_options = {'cpp' : True,
                        'indent' : 2}
        #F.generate_dependencies('gen_ik_nlp_deps.cpp', codegen_options)

        # ik_solver_path = files('pympc.code_generation').joinpath('gen_ik_nlp_deps.so')

        # check if file exists maybe?
        #F = nlpsol('F', 'ipopt', str(ik_solver_path), nlp_options)

        return F
    

    def inverse_kinematics(self, H_0_tool, q0 = np.zeros(9), q_des=np.zeros(9), inv_z_axis=True, pos_err_limit=1e-2):
        oMd = H_0_tool
        pd = oMd[:3, 3]
        rotmat = oMd[:3, :3]

        # convert oMd to rpy angles 
        r = Rotation.from_matrix(rotmat)
        rpy = r.as_euler('xyz', degrees=False)
        
        if inv_z_axis:
            # let gripper face down (z-axis down) and rotate around z-axis
            Rd = np.diag([1,-1,-1]) @ rotmat#Rz(rpy[2]) #@ Ry(rpy[1]) @ Rx(rpy[0])
        else:
            Rd = rotmat#Rz(rpy[2])

        params = np.concatenate([self.cost_weight_parameters, pd, Rd[:].flatten('F'), q_des.flatten()])
                
        r = self.ik_solver(x0 = q0, p = params.T, lbx = self.q_lb.T, ubx = self.q_ub.T, lbg = self.lbg.T, ubg = self.ubg.T)
        self.stats = self.ik_solver.stats()
        self.iterations = self.stats['iter_count']
        if(self.stats['return_status'] != 'Solve_Succeeded'):
            print('not converged with status: ' + str(self.stats['return_status']))
            result = r['x'].full()
            q_res = result[0]
            H_out = pu.get_frameSE3(self.pin_model, self.pin_data, q_res, self.tool_frame_id).homogeneous
            return q_res, False
        else:        
            result = r['x'].full()
            q_res = result[0]
            H_out = pu.get_frameSE3(self.pin_model, self.pin_data, q_res, self.tool_frame_id).homogeneous
            # compare position with desired position
            p_out = H_out[:3, 3]
            p_des = pd
            # print("H_ik: ", H_out)
            # print("H_des: ", oMd)

            pos_err = np.linalg.norm(p_out - p_des)
            if pos_err > pos_err_limit:
                print("position error: ", pos_err)
                return q_res, False
            
            # everything is fine :)   
            return q_res, True
    


    def _gravitational_forces_ca(self, q,):
        qpp_u = cpin.computeGeneralizedGravity(self.cpin_model, self.cpin_data, q)[7:]
        return qpp_u
    
    def find_steady_state(self, q_init):
        q0_a = q_init[0:7]
        # print("q_init: ", q_init)
        
        q_ss = self._newton_raphson_casadi(q_init)
        try:
            q_ss = q_ss.flatten()
        except:
            q_ss = q_ss.full().flatten()
            
        return q_ss
    
    def _newton_raphson_casadi(self, q_init):
        # Newton-Raphson iteration
        q_current = q_init.copy()
        for i in range(self.max_iter):
            # qpp_val, J_val = self.F(q_current)
            qpp_val = self.F_gg_u(q_current)
            J_val = self.F_J(q_current)

            # print("qpp_val: ", qpp_val)
            # print("J_val: ", J_val)
            qpp_norm = ca.norm_2(qpp_val)
            
            if qpp_norm < self.tol:
                # print(f"Converged after {i+1} iterations.")
                return q_current
            
            # Solve for the update (delta_q)
            # print("J_val.shape: ", J_val.shape)
            # print("qpp_val.shape: ", qpp_val.shape)

            delta_q = -ca.solve(J_val, qpp_val)
            
            # Update current estimate of q
            q_current[7:] += delta_q.full().flatten()
        
        print("Warning: Maximum iterations reached without convergence.")
        return q_current
    
    def get_acceleration_and_gradient(self, q):
        qpp_val = self.F_gg_u(q)
        J_val = self.F_J(q)
        return qpp_val, J_val



def check_ik(num_samples=10):

    samples = sample_cylindrical_workspace_segment(
        r_min=0.3,
        r_max=0.5,
        z_min=0.1,
        z_max=0.5,
        grid_spacing=0.1,
        theta_min=0.0,
        theta_max=np.pi 
    )
    samples = samples[:num_samples]
    
    
    model_path = "./robocrane/robocrane.xml"  
    mj_model, mj_data = load_mujoco_model(model_path)
    pin_model, pin_data = load_pinocchio_model(model_path)
    tool_frame_id = get_gripper_point_frame_id(pin_model)

    nq = pin_model.nq
    ss_obj = SteadyState(pin_model, pin_data, tool_frame_id)

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        # Configure viewer
        viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_WIREFRAME] = False
        viewer.opt.label = mujoco.mjtLabel.mjLABEL_SITE
        viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

        for i in range(num_samples):

            x,y,z,yaw = samples[i]
            pos = [x,y,z]
            rpy = [0,0,yaw]
            H = pos_rpy_to_homtrans(pos, rpy)
            q_init = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, +1.5708, 0.0, 0.0, 0.0])
            q_ik, succ = ss_obj.inverse_kinematics(H, q_init, q_init, inv_z_axis=True)
            '''
            if not succ:
                print(Fore.RED + "ik not successful!" + Style.RESET_ALL)
                continue
            else:
                print(Fore.GREEN + "ik success!!!" + Style.RESET_ALL)
            
            print(f"pos {pos}, rpy {rpy}")
            print(f"Sample {i+1}: Joint configuration = {q_ik}")
            print(f"Sample {i+1}: Homogeneous Transformation = {H}")
            '''
            mj_data.qpos[:nq] = q_ik
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            input("next...")

        print("Finished displaying random joint configurations.")
        viewer.close()


def ik_of_sample(sample) -> tuple[bool, np.ndarray]:
    model_path = "./robocrane/robocrane.xml"  
    mj_model, mj_data = load_mujoco_model(model_path)
    pin_model, pin_data = load_pinocchio_model(model_path)
    tool_frame_id = get_gripper_point_frame_id(pin_model)

    nq = pin_model.nq
    ss_obj = SteadyState(pin_model, pin_data, tool_frame_id)

    x,y,z,yaw = sample
    pos = [x,y,z]
    rpy = [0,0,yaw]
    H = pos_rpy_to_homtrans(pos, rpy)
    q_init = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, +1.5708, 0.0, 0.0, 0.0])
    q_ik, succ = ss_obj.inverse_kinematics(H, q_init, q_init, inv_z_axis=True)
    '''
    if not succ:
        print(Fore.RED + "ik not successful!" + Style.RESET_ALL)
        return False, np.zeros(9)
    else:
        print(Fore.GREEN + "ik success!!!" + Style.RESET_ALL)
    '''
    return True, q_ik


def p2p_trajectory_random(q_start=None, num_pts=10, plot=False):
    samples = sample_cylindrical_workspace_segment(
    r_min=0.3,
    r_max=0.5,
    z_min=0.1,
    z_max=0.5,
    grid_spacing=0.1,
    theta_min=0.0,
    theta_max=np.pi 
)
    
    
    success = False

    while not success:
        start = samples[np.random.choice(len(samples))]
        goal = samples[np.random.choice(len(samples))]

        if q_start is None:
            success1, q_start = ik_of_sample(start)
            #print(f"start: {success1}" )
        else:
            success1 = True
        success2, q_goal = ik_of_sample(goal)
        #print(f"goal: {success2}" )
        if success1 and success2:
            success = True
        else:
            a=1
            #print(f"start {start}, goal {goal}")


    q, q_vel, q_acc = cubic_joint_interpolation(q_start, q_goal, num_pts)
    if plot:
        plot_trajectory(q, q_vel, q_acc)

    return q, q_vel, q_acc





if __name__ == "__main__":
    # check_ik()
    p2p_trajectory_random()


