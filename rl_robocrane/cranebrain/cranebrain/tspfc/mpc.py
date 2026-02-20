import sys
import numpy as np
import time
import mujoco
from casadi import *
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, AcadosSim, AcadosSimSolver
import pinocchio as pin
import pinocchio.casadi as cpin
from scipy.interpolate import make_interp_spline
from colorama import Fore, Style, init
pinutil_path = os.path.abspath(os.path.join("/home/ubuntu/", "python"))
sys.path.append(pinutil_path)
from ..common.load_model import load_mujoco_model, load_pinocchio_model, get_gripper_point_frame_id, get_tool_body_id, model_path
from ..utils import pinutil as pu
from ..common import BSplines as bs
import os
from .ocp import OCP
from ..utils.plotter import plot_results, plot_pfc_results, close_all
from ..common.SteadyState import SteadyState



class MPC:
    def __init__(self, pin_model, pin_data, tool_frame_id, tool_body_id, dt, N_horizon, q0, qp0, theta0, theta_dot0, algorithm, as_rti_iter,
                 ctr_pts_ts=None, ctr_pts_js=None, regenerate=True):
        self.dof_u = 2
        self.dof = pin_model.nq
        self.dof_a = self.dof - self.dof_u

        self.pin_model = pin_model
        self.pin_data = pin_data
        self.cpin_model = cpin.Model(pin_model)
        self.cpin_data = self.cpin_model.createData()
        self.tool_frame_id = tool_frame_id
        self.tool_body_id = tool_body_id

        self.dt = dt
        self.N_horizon = N_horizon
        self.Tf = N_horizon * dt
        self.pval_ctrl_ts_ts = ctr_pts_ts
        self.pval_ctrl_ts_js = ctr_pts_js
        self.ocp = OCP(self.pin_model, self.pin_data, self.tool_frame_id, self.tool_body_id, N_horizon, self.Tf, q0, qp0, theta0, theta_dot0,
                         algorithm, as_rti_iter, regenerate, ctr_pts_ts, ctr_pts_js)

        self.plot_horizon = False
        self.init = True

        self.t_preparation = 0
        self.t_feedback = 0

        self.ss_obj = SteadyState(pin_model, pin_data, tool_frame_id)

        self.pval_weight_pos = 10*np.ones(3)
        self.pval_weight_rot = 0.1*np.ones(1)

        self.pval_weight_q = np.concatenate([5*np.ones(7), 0.1*np.ones(2)])
        self.pval_weight_qp = np.concatenate([0.1*np.ones(7), 5.0*np.ones(2)])
        self.pval_weight_theta = 10*np.ones(1)
        self.pval_weight_theta_dot = 10*np.ones(1)

        self.pval_weight_u = 2*np.ones(7)
        self.pval_weight_v = 10*np.ones(1)
        
        # inertia = self.pin_data.Ycrb[tool_body_id]
        inertia = self.pin_model.inertias[tool_body_id]
        self.pval_mass = np.array([inertia.mass])
        self.pval_com = np.array([inertia.lever]).flatten()
        # inertia_tcp = inertia.inertia.diagonal()

        self.pval_theta_ref = np.ones(1)  # reference path parameter


    def set_ts_ctr_pts(self, ctr_pts):
        if ctr_pts.shape != (self.ocp.n_control_points, self.ocp.nts):
            raise ValueError("ctr_pts wrong dimensions! Dimension {}".format(ctr_pts.shape))
        self.pval_ctrl_ts_ts = ctr_pts

    def set_js_ctr_pts(self, ctr_pts):
        if ctr_pts.shape != (self.ocp.n_control_points, self.ocp.nq):
            raise ValueError("ctr_pts wrong dimensions! Dimension {}".format(ctr_pts.shape))
        self.pval_ctrl_ts_js = ctr_pts
    
    def set_weight_param(self, weight_pos, weight_rot, weight_q, weight_qp, weight_theta, weight_theta_dot, weight_u, weight_v):
        self.param_weighs = np.array([weight_pos, weight_rot, weight_q, weight_qp, weight_theta, weight_theta_dot, weight_u, weight_v])

    def set_weight_pos(self, weight_pos):
        if len(weight_pos) != 3:
            raise ValueError("weight_pos should be of length 3")
        self.pval_weight_pos = weight_pos

    def set_weight_rot(self, weight_rot):
        if len(weight_rot) != 1:
            raise ValueError("weight_rot should be of length 1")
        self.pval_weight_rot = weight_rot

    def set_weight_q(self, weight_q):
        if len(weight_q) != 9:
            raise ValueError("weight_q should be of length 9")
        self.pval_weight_q = weight_q

    def set_weight_qp(self, weight_qp):
        if len(weight_qp) != 9:
            raise ValueError("weight_qp should be of length 9")
        self.pval_weight_qp = weight_qp

    def set_weight_theta(self, weight_theta):
        if len(weight_theta) != 1:
            raise ValueError("weight_theta should be of length 1")
        self.pval_weight_theta = weight_theta

    def set_weight_theta_dot(self, weight_theta_dot):
        if len(weight_theta_dot) != 1:
            raise ValueError("weight_theta_dot should be of length 1")
        self.pval_weight_theta_dot = weight_theta_dot

    def set_weight_u(self, weight_u):
        if len(weight_u) != 7:
            raise ValueError("weight_u should be of length 7")
        self.pval_weight_u = weight_u

    def set_weight_v(self, weight_v):
        if len(weight_v) != 1:
            raise ValueError("weight_v should be of length 1")
        self.pval_weight_v = weight_v

    def set_mass(self, mass):
        if type(mass) is not float:
            raise ValueError("mass should be a float")
        self.pval_mass = np.array([mass])

    def set_com(self, com):
        if len(com) != 3:
            raise ValueError("com should be of length 3")
        self.pval_com = com

    def set_theta_ref(self, theta_ref):
        if not isinstance(theta_ref, (float, np.floating)):
            print("type(theta_ref): ", type(theta_ref))
            raise ValueError("theta_ref should be a float")
        self.pval_theta_ref = np.array([theta_ref])


    def set_ocp_param(self, theta0):
        p_val_init = self.ocp.solver.get(0,"p")
        pval_ts = np.reshape(self.pval_ctrl_ts_ts, (self.ocp.n_control_points * self.ocp.nts))
        pval_js = np.reshape(self.pval_ctrl_ts_js, (self.ocp.n_control_points * self.ocp.nq))

        weight_theta_scaler = 1.0/np.max([0.1,(np.abs(1.0-theta0))])
        # weight_theta_scaler = 1.0
        pval_weight_theta_scaled = weight_theta_scaler*self.pval_weight_theta

        pval = np.concatenate([self.pval_weight_pos, self.pval_weight_rot,
                               self.pval_weight_q, self.pval_weight_qp, pval_weight_theta_scaled, self.pval_weight_theta_dot,
                               self.pval_weight_u, self.pval_weight_v,
                               self.pval_mass, self.pval_com, self.pval_theta_ref,
                               pval_ts, pval_js])
        for i in range(self.N_horizon+1):
            self.ocp.solver.set(i, "p", pval)

    def reset(self, x0, x_ref):
        x_traj = np.linspace(x0, x_ref, self.N_horizon)
        for i in range(self.N_horizon):
            self.ocp.solver.set(i, "x", x_traj[i])
        # print(Fore.GREEN + "mpc reset" + Style.RESET_ALL)
        # print(Fore.GREEN + "x0: \n{}".format(x0) + Style.RESET_ALL)
        # print(Fore.GREEN + "x_ref: \n{}".format(x_ref) + Style.RESET_ALL)


    
    def hard_reset(self, x0: np.ndarray):
        """
        Fully reset acados memories so a bad QP (e.g., NaNs) cannot poison later solves.
        Keeps generated code; recreates the solver only if needed.
        """
        s = self.ocp.solver

        # 1) Try the Python-level reset (wipes NLP & inner QP memory).
        #    This exists in recent acados (see forum note below).
        try:
            s.reset()  # clears internal memories & warm starts
            print("[acados] solver.reset() done.")
        except Exception:
            # 2) Fallback: ask to reset QP memory explicitly if exposed
            try:
                s.options_set("reset_qp_memory", 1)
                print("[acados] options_set('reset_qp_memory', 1) done.")
            except Exception:
                # 3) Last resort: rebuild the solver (no codegen)
                print("[acados] Recreating solver (no codegen)…")
                self.ocp._create_acados_solver_and_integrator()
                s = self.ocp.solver

        # 4) Disable warm start for the next solve or two
        try:
            s.options_set("qp_solver_warm_start", 0)
        except Exception:
            pass

        # 5) Reinitialize state & input guesses and initial-state box
        zero_u = np.zeros(self.ocp.nu)
        for i in range(self.N_horizon + 1):
            s.set(i, "x", x0.copy())
            if i < self.N_horizon:
                s.set(i, "u", zero_u)
        s.set(0, "lbx", x0)
        s.set(0, "ubx", x0)

        # 6) Reapply parameters and let acados precompute dependencies if available
        try:
            self.set_ocp_param(theta0=x0[-2])                 # your param packer
            s.set_p_global_and_precompute_dependencies()      # precompute (if present)
        except Exception:
            pass

        # 7) Make the MPC do a fresh horizon init on next iterate()
        self.init = True


    def get_y_ref(self, x, u):
        n_ctrl_pts = self.ocp.n_control_points
        k = self.ocp.spline_order
        t = bs.knot_vector(n_ctrl_pts, k)
        pval = self.ocp.solver.get(0, 'p')
        pval_ts = pval[self.ocp.id_ctrl_pts_ts:self.ocp.id_ctrl_pts_js]
        pval_js = pval[self.ocp.id_ctrl_pts_js:]
        
        theta = x[-2]
        c_ts = np.reshape(pval_ts, (n_ctrl_pts, self.ocp.nts))
        p_ref = bs.bspline(theta, t, c_ts, k)

        c_js = np.reshape(pval_js, (n_ctrl_pts, self.ocp.nq))
        q_ref = bs.bspline(theta, t, c_js, k)
        
        q_dot_ref = np.zeros(self.pin_model.nq)
        theta_ref = pval[self.ocp.id_theta_ref]
        theta_dot_ref = np.zeros(1)
        qpp_a_ref = np.zeros(7)
        v_ref = np.zeros(1)
        y_ref = np.hstack([p_ref, q_ref, q_dot_ref, theta_ref, theta_dot_ref, qpp_a_ref, v_ref])
        return y_ref

    def get_y_act(self, x, u):
        theta = x[-2]
        theta_dot = x[-1]
        q = x[:self.dof]
        q_dot = x[self.dof:-2]
        qpp_a = u[:self.dof_a]
        theta_dotdot = u[-1]
        pos_yaw = self.ocp.forward_kin4D(q)
        y_act = np.hstack([pos_yaw, q, q_dot, theta, theta_dot, qpp_a, theta_dotdot])
        return y_act

    def get_error(self, x, u):
        y_ref = self.get_y_ref(x, u)
        y = self.get_y_act(x, u)
        error = y - y_ref
        print("theta: ", x[-2])
        print("y_ref: ", y_ref[:self.dof])
        print("y: ", y[:self.dof])
        return error

    def get_cost(self):
        return self.ocp.get_cost()

    def predict(self, t0, x0, u0):
        t1 = t0 + self.dt
        x1 = self.ocp.integrator.simulate(x=x0, u=u0)
        return t1, x1

    def ocp_solve(self, t1, x1):
        if self.ocp.algorithm != "SQP":
            self.ocp.solver.set(0, "lbx", x1)
            self.ocp.solver.set(0, "ubx", x1)
            self.ocp.solver.options_set('rti_phase', 1)
            status = self.ocp.solver.solve()
            t_preparation = self.ocp.solver.get_stats('time_tot')
            self.ocp.solver.options_set('rti_phase', 2)
            status = self.ocp.solver.solve()
            t_feedback = self.ocp.solver.get_stats('time_tot')
            u1 = self.ocp.solver.get(0, "u")
        else:
            u1 = self.ocp.solver.solve_for_x0(x0_bar=x1)
            t_preparation = 0
            t_feedback = self.ocp.solver.get_stats('time_tot')
        return u1, t_preparation, t_feedback

    def get_trajectory(self, t_horizon, x_horizon, u_horizon):
        # t_traj = [ac.Time(t) for t in t_horizon]
        t_traj = [t for t in t_horizon]
        q_traj = [x_horizon[i][:self.dof] for i in range(self.N_horizon)]
        qp_traj = [x_horizon[i][self.dof:-2] for i in range(self.N_horizon)]
        theta_traj = [x_horizon[i][-2] for i in range(self.N_horizon)]
        u_traj = [u_horizon[i][:self.dof_a] for i in range(self.N_horizon)]
        v_traj = [u_horizon[i][-1] for i in range(self.N_horizon)]
        return t_traj, q_traj, qp_traj, theta_traj, u_traj, v_traj

    def get_result(self, x_horizon, u_horizon):
        x_first = x_horizon[1]
        u_first = u_horizon[1]
        return u_first, x_first

    def get_solver_status(self):
        return self.ocp.get_status()

    def iterate(self, t0, q0, qp0, theta0, theta_dot0, u0, v0):
        x0 = np.concatenate([q0, qp0, [theta0, theta_dot0]])
        # scale weight_theta

        self.set_ocp_param(theta0)
        if self.init:
            # TODO: set horizon based on ts and js splines
            self.reset(x0, x0)
            self.init = False

        # if self.plot_horizon:
        #     close_all()
        #     t_init, x_init, u_init = self.get_ocp_result()
        #     u_init = u_init[:-1]
        #     plot_pfc_results(t_init, self.ocp.u_limit, np.array(u_init), np.array(x_init),
        #                      title="before", plt_show=False, linestyle='--')
        #     input("Wait for user input...")

        u1, t_preparation, t_feedback = self.ocp_solve(t0, x0)
        self.ocp.check_limits()
        t_horizon, x_horizon, u_horizon = self.get_ocp_result()
        t_traj, q_traj, qp_traj, theta_traj, u_traj, v_traj = self.get_trajectory(t_horizon, x_horizon, u_horizon)
        u1, x1 = self.get_result(x_horizon, u_horizon)

        if self.plot_horizon:
            u_horizon = u_horizon[:-1]
            plot_pfc_results(t_horizon, self.ocp.u_limit, np.array(u_horizon), np.array(x_horizon),
                             title=self.ocp.algorithm, plt_show=False)
            input("Wait for user input...")
        return t_traj, q_traj, qp_traj, theta_traj, u_traj, v_traj, u1, x1

    def simulate(self, x0, u0):
        x1 = self.ocp.integrator.simulate(x=x0, u=u0)
        return x1

    def get_ocp_result(self):
        t = np.linspace(0, self.Tf, self.N_horizon)
        x = []
        u = []
        for i in range(self.N_horizon):
            x.append(self.ocp.solver.get(i, "x"))
            u.append(self.ocp.solver.get(i, "u"))
        return t, x, u
    
    # def plot_horizoon(self):
    #     t_horizon, x_horizon, u_horizon = self.get_ocp_result()

    #     for i in range(self.N_horizon):
    #         x = np.array(x_horizon[i]).flatten()
    #         u = np.array(u_horizon[i]).flatten()
            
    #         y_act = self.get_y_act(x, u)
    #         y_ref = self.get_y_ref(x, u)

    #         q_act = y_act[:self.dof]
    #         q_ref = y_ref[:self.dof]
    #         qp_act = y_act[self.dof:-2]
    #         qp_ref = y_ref[self.dof:-2]
    #         theta_act = y_act[-2]
    #         theta_ref = y_ref[-2]
    #         theta_dot_act = y_act[-1]
    #         theta_dot_ref = y_ref[-1]

            

    
    # TODO: remove inv_z_axis - not necessary as g constraint handles u.a. angles
    def inverse_kinematics(self, H_des, q_init, q_null_des=np.zeros(9), inv_z_axis=True):
        q_ik, success = self.ss_obj.inverse_kinematics(H_des, q_init, q_null_des, inv_z_axis)
        # if not success:
        #     raise Exception("IK failed")
        return q_ik, success


    def forward_kinematics(self, q):
        pin.forwardKinematics(self.pin_model, self.pin_data, q.T)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        oMf_tool = self.pin_data.oMf[self.tool_frame_id]
        return oMf_tool.homogeneous


    def get_debug_data(self):
        y_ref = []
        y = []
        x_values = []
        u_values = []
        p_values = []
        for i in range(self.N_horizon):
            y_ref.append(self.ocp.get_y_ref(i))
            y.append(self.ocp.get_y(i))
            x_values.append(self.ocp.get_x(i))
            u_values.append(self.ocp.get_u(i))
            p_values.append(self.ocp.get_p(i))
        return y_ref, y, x_values, u_values, p_values
    

    def print_debug_data(self):
        y_ref_list, y_list, x_list, u_list, p_list = self.get_debug_data()
        num_stages = len(y_ref_list)
        if num_stages == 0:
            print("No debug data available.")
            return
        np.set_printoptions(linewidth=140)
        for stage in range(num_stages):
            print(f"Stage:\t{stage}")
            print(f"y_ref:\t{y_ref_list[stage]}")
            print(f"y:\t{y_list[stage]}")
            print(f"p:\t{p_list[stage]}")
            print("-" * 140)


from ..utils.util import homtrans_to_pos_yaw
from ..utils.pinutil import get_frameSE3

def main(algorithm, as_rti_iter):
    mj_model, mj_data = load_mujoco_model(model_path)
    pin_model, pin_data = load_pinocchio_model(model_path)
    tool_frame_id = get_gripper_point_frame_id(pin_model)
    tool_body_id = get_tool_body_id(pin_model)
    
    k = 2
    q0 = np.array([0, 0, 0, np.pi/2, 0, -np.pi/2, 0, 0, 0])
    joint_offset = np.array([np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0])
    qd = q0 + joint_offset
    qw = q0 + 0.5*joint_offset
    via_pts = [q0, qw, qd]    
    ctr_pts, knot_vec = bs.compute_control_points(via_pts, k)

    # pos_yaw = np.array([0.5, -0.25, 0.3, -1.67])
    ctr_pts_ts = []
    for i in range(ctr_pts.shape[0]):
        T = get_frameSE3(pin_model, pin_data, ctr_pts[i], tool_frame_id).homogeneous
        pos_yaw = homtrans_to_pos_yaw(T)
        ctr_pts_ts.append(pos_yaw)
    ctr_pts_js = np.tile(q0, 3)
    ctr_pts_ts = np.array(ctr_pts_ts)

    print("q0: " + str(q0))
    print("qw: " + str(qw))
    print("qd: " + str(qd))
    qp0 = np.zeros(pin_model.nq)
    u0 = np.zeros(pin_model.nq - 2)
    v0 = 0
    theta0 = 0
    theta_dot0 = 0
    x0 = np.concatenate([q0, qp0, [theta0, theta_dot0]])
    dt = 0.05
    N_horizon = 50
    Tf = N_horizon * dt
    controller = MPC(pin_model, pin_data, tool_frame_id, tool_body_id, dt, N_horizon, q0, qp0, theta0, theta_dot0, algorithm, as_rti_iter,
                     ctr_pts_js=ctr_pts_js,ctr_pts_ts=ctr_pts_ts, regenerate=False)
    t = np.linspace(0, 5 * Tf, 5 * N_horizon)
    t_exec = []
    x_res = []
    u_res = []
    m = mj_model
    d = mj_data
    iteration = 0
    with mujoco.viewer.launch_passive(m, d) as viewer:
        for t0 in t:
            if not viewer.is_running():
                break
            d.qpos[:pin_model.nq] = q0
            mujoco.mj_forward(m, d)
            viewer.sync()
            start = time.time()
            t_traj, q_traj, qp_traj, theta_traj, u_traj, v_traj, u1, x1 = controller.iterate(t0, q0, qp0, theta0, theta_dot0, u0, v0)
            end = time.time()
            t_exec.append(end - start)
            x_res.append(x1)
            u_res.append(u1)
            q0 = x1[:9]
            qp0 = x1[9:-2]
            theta0 = x1[-2]
            theta_dot0 = x1[-1]
            u0 = u1[:-1]
            v0 = u1[-1]
            yref = controller.get_y_ref(x1, u1)
            yact = controller.get_y_act(x1, u1)
            theta_ref = yref[4+2*9]
            print("---")
            print("t0: " + str(t0))
            print("qd: ", np.round(qd, 2).T)
            print("q0: ", np.round(q0, 2).T)
            print("qp0: ", np.round(qp0, 2).T)
            print("u0: ", np.round(u0, 2).T)
            print("v0: ", np.round(v0, 2).T)
            print("theta0: ", np.round(theta0, 2))
            print("theta_dot0: ", np.round(theta_dot0, 2))
            print("theta_ref: ", np.round(theta_ref, 2))
            if controller.get_solver_status() != 0:
                print("Solver failed at t = " + str(t0))
                print("iteration: " + str(iteration))
                print("Solver status: " + str(controller.get_solver_status()))
                exit()
            iteration += 1
    print("Average execution time: " + str(np.mean(t_exec)))
    print("std execution time: " + str(np.std(t_exec)))
    print("max execution time: " + str(np.max(t_exec)))
    print("min execution time: " + str(np.min(t_exec)))
    print("plot results...")
    u_res = np.array(u_res)
    x_res.append(x_res[-1])
    x_res = np.array(x_res)
    t_res = np.append(t, t[-1])
    print("u_res.shape: ", u_res.shape)
    print("x_res.shape: ", x_res.shape)
    print("t_res.shape: ", t_res.shape)
    plot_pfc_results(t_res, controller.ocp.u_limit, u_res, x_res, title=algorithm)

if __name__ == '__main__':
    main(algorithm="RTI", as_rti_iter=1)