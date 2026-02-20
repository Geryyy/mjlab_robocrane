
import sys
import os
# sys.path.insert(0, '../common')
# sys.path.insert(0, '../utils')
from .crane_model import export_robocrane_pfc_model
from ..utils.plotter import plot_results, plot_pfc_results

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver
import numpy as np
import scipy.linalg
import casadi as ca
from pinocchio import casadi as cpin
import pinocchio as pin

pinutil_path = os.path.abspath(os.path.join("/home/ubuntu/", "python"))
sys.path.append(pinutil_path)
from colorama import Fore, Style, init
from ..common.load_model import load_mujoco_model, load_pinocchio_model, get_gripper_point_frame_id, get_tool_body_id, model_path
from ..common import BSplines as bs

# TODO: implement taskspace cost function for path-following control 

def R_z(yaw):
    R = ca.SX.zeros(3, 3)
    R[0, 0] = ca.cos(yaw)
    R[0, 1] = -ca.sin(yaw)
    R[0, 2] = 0
    R[1, 0] = ca.sin(yaw)
    R[1, 1] = ca.cos(yaw)
    R[1, 2] = 0
    R[2, 0] = 0
    R[2, 1] = 0
    R[2, 2] = 1
    return R

def R_x(roll):
    R = ca.SX.zeros(3, 3)
    R[0, 0] = 1
    R[0, 1] = 0
    R[0, 2] = 0
    R[1, 0] = 0
    R[1, 1] = ca.cos(roll)
    R[1, 2] = -ca.sin(roll)
    R[2, 0] = 0
    R[2, 1] = ca.sin(roll)
    R[2, 2] = ca.cos(roll)
    return R


class OCP:
    REAL_TIME_ALGORITHMS = ["RTI", "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]
    ALGORITHMS = ["SQP"] + REAL_TIME_ALGORITHMS

    def __init__(self, pin_model, pin_data, tool_frame_id, tool_body_id, N_horizon, Tf, q0, qp0, theta0, theta_dot0, algorithm="SQP", as_rti_iter=1, regenerate=False, ctr_pts_ts=None, ctr_pts_js=None):
        """
        Initializes the OCP (Optimal Control Problem) object.
        """

        # 1. Store input parameters
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.cpin_model = cpin.Model(self.pin_model)
        self.cpin_data = self.cpin_model.createData()
        self.tool_frame_id = tool_frame_id
        self.tool_body_id = tool_body_id
        self.N_horizon = N_horizon
        self.Tf = Tf
        self.algorithm = algorithm
        self.as_rti_iter = as_rti_iter
        self.regenerate = regenerate
        self.ctr_pts_ts = ctr_pts_ts
        self.ctr_pts_js = ctr_pts_js

        self._set_limits()
        self.x0 = np.concatenate([q0, qp0, [theta0, theta_dot0]])
        self._set_parameters()
        self.ocp = AcadosOcp()
        self._set_acados_model()
        self._set_acados_parameters()
        self._set_acados_cost()
        self._set_acados_constraints()
        self._set_acados_solver_options()
        self._create_acados_solver_and_integrator()

    def _set_limits(self):
        """Sets joint and control limits."""
        self.q_ub = np.array([170, 120, 170, 120, 170, 120, 175, 45, 45]) * np.pi / 180 * 0.95
        self.q_lb = -self.q_ub
        self.qp_ub = np.array([85, 85, 100, 75, 130, 135, 135, 50, 50]) * np.pi / 180
        self.qp_lb = -self.qp_ub
        self.u_limit = 1

    def _set_parameters(self):
        """Sets spline parameters."""
        self.spline_order = 2
        self.n_control_points = 3
        self.nq = 9
        self.nts = 4  # dimension pos + yaw

        # cost function weight parameters indices
        self.id_weight_pos = 0
        self.id_weight_rot = self.id_weight_pos + 3
        self.id_weight_q = self.id_weight_rot + 1
        self.id_weight_qp = self.id_weight_q + 9
        self.id_weight_theta = self.id_weight_qp + 9
        self.id_weight_theta_dot = self.id_weight_theta + 1
        self.id_weight_u = self.id_weight_theta_dot + 1
        self.id_weight_v = self.id_weight_u + 7
        # inertia parameters indices
        self.id_mass_value = self.id_weight_v + 1
        self.id_com_value = self.id_mass_value + 1
        self.id_theta_ref = self.id_com_value + 3  # reference for theta
        # control points indices
        self.id_ctrl_pts_ts = self.id_theta_ref + 1
        self.id_ctrl_pts_js = self.id_ctrl_pts_ts + self.n_control_points*self.nts

    def _set_acados_model(self):
        """Sets the Acados model."""
        model, p_val = export_robocrane_pfc_model(self.cpin_model, self.cpin_data, self.tool_frame_id, self.tool_body_id, self.n_control_points, taskspace_flag=True)
        model.name = "tspfc_model"
        self.ocp.model = model

        if not isinstance(self.ctr_pts_ts, np.ndarray):
            self.ocp.parameter_values = p_val
        else:
            print("ctr_pts_ts: ", self.ctr_pts_ts)
            print("dim ctr_pts_ts: ", self.ctr_pts_ts.shape)
            pval_ts = np.reshape(self.ctr_pts_ts, (self.n_control_points * self.nts))
            pval_js = np.reshape(self.ctr_pts_js, (self.n_control_points * self.nq))
            pval_ctrl = np.concatenate([pval_ts, pval_js])
            p_val[self.id_ctrl_pts_ts:] = pval_ctrl # take default weight and inertia parameter
            print("pval: ", p_val)
            print("dim pval: ", p_val.shape)
            self.ocp.parameter_values = p_val

        self.pval_weight_pos_init = p_val[self.id_weight_pos : self.id_weight_pos + 3]
        self.pval_weight_rot_init = p_val[self.id_weight_rot]
        self.pval_weight_q_init = p_val[self.id_weight_q : self.id_weight_q + 9]
        self.pval_weight_qp_init = p_val[self.id_weight_qp : self.id_weight_qp + 9]
        self.pval_weight_theta_init = p_val[self.id_weight_theta]
        self.pval_weight_theta_init_dot = p_val[self.id_weight_theta_dot]
        self.pval_weight_u_init = p_val[self.id_weight_u : self.id_weight_u + 7]
        self.pval_weight_v_init = p_val[self.id_weight_v]
        self.pval_mass_value_init = p_val[self.id_mass_value]
        self.pval_com_value_init = p_val[self.id_com_value : self.id_com_value + 3]

        self.nx = model.x.rows()
        self.nu = model.u.rows()

    def _set_acados_parameters(self):
        """Sets Acados dimensions and solver options."""
        self.ocp.dims.N = self.N_horizon
        self.ocp.solver_options.N_horizon = self.N_horizon

    def _calculate_kinematics(self, q, cpin_model, cpin_data):
        """Calculates forward kinematics and yaw."""
        cpin.forwardKinematics(cpin_model, cpin_data, q)
        cpin.updateFramePlacements(cpin_model, cpin_data)
        oMact = cpin_data.oMf[self.tool_frame_id].homogeneous
        pos_act = oMact[0:3, 3]
        rotmat_act = oMact[0:3, 0:3]
        # yaw_act = ca.atan2(rotmat_act[1, 0], rotmat_act[0, 0])
        return pos_act, rotmat_act

    def _calculate_lagrange_cost(self, q, q_dot, theta, theta_dot, u, 
                                 weight_pos, weight_rot, weight_q, weight_qp, weight_theta, weight_theta_dot, weight_u, weight_v, theta_ref,
                                 ctrl_pts_ts, ctrl_pts_js, cpin_model, cpin_data):
        """Calculates the Lagrange cost term."""
        pos_act, rotmat_act = self._calculate_kinematics(q, cpin_model, cpin_data)
        y = ca.vertcat(pos_act, q, q_dot, theta, theta_dot, u)
        yref, rotmat_des = self.compute_y_ref(self.ocp.model.x, u, theta_ref, ctrl_pts_ts, ctrl_pts_js, cpin_model.nq, self.nts)
        W, Q_rot = self._set_lagrange_cost_weights(weight_pos, weight_rot, weight_q, weight_qp, weight_theta, weight_theta_dot, weight_u, weight_v)
        residual = y - yref
        # rot_err = 2*(3-ca.trace(rotmat_act * rotmat_des.T))
        rot_err = 1 - ca.dot(rotmat_act[:,0], rotmat_des[:,0])
        return ca.dot(residual, ca.mtimes(W, residual)) + Q_rot*ca.dot(rot_err, rot_err)

    def _calculate_mayer_cost(self, q, q_dot, theta, theta_dot, u, 
                              weight_pos_e, weight_rot_e, weight_q_e, weight_qp_e, weight_theta_e, weight_theta_dot_e, theta_ref,
                              ctrl_pts_ts, ctrl_pts_js, cpin_model, cpin_data):
        """Calculates the Mayer cost term."""
        pos_act, rotmat_act = self._calculate_kinematics(q, cpin_model, cpin_data)
        y_end = ca.vertcat(pos_act, q, q_dot, theta, theta_dot)
        y_endref, rotmat_des = self.compute_y_endref(self.ocp.model.x, u, theta_ref, ctrl_pts_ts, ctrl_pts_js, cpin_model.nq, self.nts)
        W_e, Qe_rot = self._set_mayer_cost_weights(weight_pos_e, weight_rot_e, weight_q_e, weight_qp_e, weight_theta_e, weight_theta_dot_e)
        # rot_err = 2*(3-ca.trace(rotmat_act * rotmat_des.T))
        rot_err = 1 - ca.dot(rotmat_act[:,0], rotmat_des[:,0])
        return ca.dot(y_end - y_endref, ca.mtimes(W_e, y_end - y_endref)) + Qe_rot*ca.dot(rot_err, rot_err)


    ### MIX NP AND CASADI
    def _set_lagrange_cost_weights(self, weight_pos, weight_rot, weight_q, weight_qp, weight_theta, weight_theta_dot, weight_u, weight_v):
        """Sets the Lagrange cost weight matrix."""
        Q_diag = ca.vertcat(weight_pos, weight_q, weight_qp, weight_theta, weight_theta_dot)
        Q = ca.diag(Q_diag)
        R_diag = ca.vertcat(weight_u, weight_v)
        R = ca.diag(R_diag)
        Q_rot = weight_rot

        # Use block_diag from CasADi's built-in functions
        block_diag_QR = ca.blockcat([[Q, ca.SX.zeros(Q.shape[0], R.shape[1])],
                             [ca.SX.zeros(R.shape[0], Q.shape[1]), R]])
        return block_diag_QR, Q_rot
    

    def _set_mayer_cost_weights(self, weight_pos_e, weight_rot_e, weight_q_e, weight_qp_e, weight_theta_e, weight_theta_dot_e):
        """Sets the Mayer cost weight matrix."""
        Qe_diag = ca.vertcat(weight_pos_e, weight_q_e, weight_qp_e, weight_theta_e, weight_theta_dot_e)
        Qe = ca.diag(Qe_diag)
        Qe_rot = weight_rot_e
        return Qe, Qe_rot
    

    def _set_acados_cost(self):
        """Sets Acados cost functions."""
        self.ocp.cost.cost_type = "EXTERNAL"
        q = self.ocp.model.x[:self.nq]
        q_dot = self.ocp.model.x[self.nq:-2]
        theta = self.ocp.model.x[-2]
        theta_dot = self.ocp.model.x[-1]
        u = self.ocp.model.u
        weight_pos = self.ocp.model.p[self.id_weight_pos : self.id_weight_pos + 3]
        weight_rot = self.ocp.model.p[self.id_weight_rot]
        weight_q = self.ocp.model.p[self.id_weight_q : self.id_weight_q + 9]
        weight_qp = self.ocp.model.p[self.id_weight_qp : self.id_weight_qp + 9]
        weight_theta = self.ocp.model.p[self.id_weight_theta]
        weight_theta_dot = self.ocp.model.p[self.id_weight_theta_dot]
        weight_u = self.ocp.model.p[self.id_weight_u : self.id_weight_u + 7]
        weight_v = self.ocp.model.p[self.id_weight_v]
        theta_ref = self.ocp.model.p[self.id_theta_ref]
        ctrl_pts_ts = self.ocp.model.p[self.id_ctrl_pts_ts : self.id_ctrl_pts_js]
        ctrl_pts_js = self.ocp.model.p[self.id_ctrl_pts_js :]
        cpin_model = self.cpin_model
        cpin_data = self.cpin_data

        self.ocp.model.cost_expr_ext_cost = self._calculate_lagrange_cost(q, q_dot, theta, theta_dot, u, 
                                                                          weight_pos, weight_rot, weight_q, weight_qp, weight_theta, weight_theta_dot, weight_u, weight_v, theta_ref,
                                                                          ctrl_pts_ts, ctrl_pts_js, cpin_model, cpin_data)

        # self.ocp.cost.cost_type_e = "EXTERNAL"
        # self.ocp.model.cost_expr_ext_cost_e = self._calculate_mayer_cost(q, q_dot, theta, theta_dot, u, 
                                                                        #  weight_pos, weight_rot, weight_q, weight_qp, weight_theta, weight_theta_dot, theta_ref,
                                                                        #  ctrl_pts_ts, ctrl_pts_js, cpin_model, cpin_data)

    def _set_acados_constraints(self):
        """Sets Acados constraints."""
        self.ocp.constraints.lbu = np.concatenate([-self.u_limit * np.ones(self.nu - 1), [-1]])
        self.ocp.constraints.ubu = np.concatenate([self.u_limit * np.ones(self.nu - 1), [1]])
        self.ocp.constraints.x0 = self.x0
        self.ocp.constraints.idxbu = np.arange(self.nu)

        self.ocp.constraints.lbx = np.concatenate([self.q_lb, self.qp_lb, [0], [-1]]) # q, qp, theta, theta_dot
        self.ocp.constraints.ubx = np.concatenate([self.q_ub, self.qp_ub, [1], [1]])
        self.ocp.constraints.idxbx = np.arange(self.nx)

    def _set_acados_solver_options(self):
        """Sets Acados solver options."""
        # 'PARTIAL_CONDENSING_HPIPM', 'FULL_CONDENSING_QPOASES', 'FULL_CONDENSING_HPIPM', 
        # 'PARTIAL_CONDENSING_QPDUNES', 'PARTIAL_CONDENSING_OSQP', 'FULL_CONDENSING_DAQP'
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.qp_solver_iter_max = 20
        self.ocp.solver_options.qp_solver_warm_start = 1
        # self.ocp.solver_options.qp_solver_tol_stat = 1e-4
        # self.ocp.solver_options.qp_solver_tol_eq = 1e-4
        # self.ocp.solver_options.qp_solver_tol_ineq = 1e-4
        # self.ocp.solver_options.regularize_method = 'CONVEXIFY'
        self.ocp.solver_options.globalization = "MERIT_BACKTRACKING"
        # self.ocp.solver_options.eps_sufficient_descent = 1e-2
        # self.ocp.solver_options.nlp_solver_tol_stat = 1e-3
        # self.ocp.solver_options.globalization_fixed_step_length = 0.5
        # self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_QPDUNES'
        self.ocp.solver_options.hessian_approx = 'EXACT'
        self.ocp.solver_options.integrator_type = 'IRK'
        # self.ocp.solver_options.qp_solver_warm_start = 0
        # self.ocp.solver_options.qp_solver_iter_max = 20

        if self.algorithm in self.REAL_TIME_ALGORITHMS:
            self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'
        else:
            self.ocp.solver_options.nlp_solver_type = 'SQP'

        if self.algorithm == "AS-RTI-A":
            self.ocp.solver_options.as_rti_iter = self.as_rti_iter
            self.ocp.solver_options.as_rti_level = 0
        elif self.algorithm == "AS-RTI-B":
            self.ocp.solver_options.as_rti_iter = self.as_rti_iter
            self.ocp.solver_options.as_rti_level = 1
        elif self.algorithm == "AS-RTI-C":
            self.ocp.solver_options.as_rti_iter = self.as_rti_iter
            self.ocp.solver_options.as_rti_level = 2
        elif self.algorithm == "AS-RTI-D":
            self.ocp.solver_options.as_rti_iter = self.as_rti_iter
            self.ocp.solver_options.as_rti_level = 3

        self.ocp.solver_options.qp_solver_cond_N = self.N_horizon
        self.ocp.solver_options.tf = self.Tf

    def _create_acados_solver_and_integrator(self):
        """Creates Acados solver and integrator."""
        solver_json = 'acados_ocp_' + self.ocp.model.name + '.json'
        try:
            self.solver = AcadosOcpSolver(self.ocp, json_file=solver_json, build=self.regenerate, generate=self.regenerate)
        except OSError as e:
            print(Fore.RED + "Could not open {}".format(e) + Style.RESET_ALL)
            self.solver = AcadosOcpSolver(self.ocp, json_file=solver_json, build=True, generate=True)

        self.integrator = AcadosSimSolver(self.ocp, json_file=solver_json)


    def compute_nullspace_projection_full_orientation(self, cpin_model, cpin_data, q):
        """
        Computes the nullspace projection matrix for the task Jacobian
        (full 6D end-effector pose: position and orientation)
        with added damping for safety near singularities.
        """
        nq = self.pin_model.nq
        tool_frame_id = self.tool_frame_id

        # Compute the full 6xnq geometric Jacobian for the end-effector
        cpin.computeJointJacobians(cpin_model, cpin_data, q)
        last_joint_id = cpin_model.joints[-1].id
        J_full = cpin.getJointJacobian(cpin_model, cpin_data, last_joint_id, cpin.ReferenceFrame.WORLD)


        # The task Jacobian is the full 6D Jacobian
        J_task = J_full

        # Calculate the pseudoinverse with damping
        damping = 1e-6  # Small damping factor for safety
        J_JT = J_task @ J_task.T
        J_pinv = J_task.T @ ca.inv(J_JT + damping * ca.SX.eye(J_task.shape[0]))

        # Calculate the nullspace projection matrix
        P_null = ca.SX.eye(nq) - J_pinv @ J_task

        return P_null
    

    def compute_y_ref(self, x, u, theta_ref, ctrl_pts_ts, ctrl_pts_js, nq, nts):
        """Computes the reference output vector y_ref for the current stage."""
        t = bs.knot_vector(self.n_control_points, self.spline_order)
        c_ts = ca.SX.reshape(ctrl_pts_ts, (nts, self.n_control_points)).T
        theta = x[-2]
        q = x[:nq]
        pos_yaw = bs.casadiBsplineVec(theta, t, c_ts, self.spline_order).T
        pos = pos_yaw[:3]
        yaw = pos_yaw[3]
        R_des = R_z(yaw) @ R_x(np.pi)
        c_js = ca.SX.reshape(ctrl_pts_js, (nq, self.n_control_points)).T
        q_ref = bs.casadiBsplineVec(theta, t, c_js, self.spline_order).T
        # q_ref = ca.SX.zeros(nq)
        flag_nullspace_projection = False
        if flag_nullspace_projection:
            P_null = self.compute_nullspace_projection_full_orientation(self.cpin_model, self.cpin_data, q)
            q_dot_N = -0.1*ca.SX.ones(nq) * q
            q_dot_ref = P_null @  q_dot_N #ca.SX.zeros(nq)
        else:
            q_dot_ref = ca.SX.zeros(nq)
        # theta_ref = ca.SX.ones(1)
        theta_dot_ref = ca.SX.zeros(1)
        qpp_a_ref = ca.SX.zeros(7)
        v_ref = ca.SX.zeros(1)
        return ca.vertcat(pos, q_ref, q_dot_ref, theta_ref, theta_dot_ref, qpp_a_ref, v_ref), R_des

    def compute_y_endref(self, x, u, theta_ref, ctrl_pts_ts, ctrl_pts_js, nq, nts):
        """Computes the reference output vector y_endref for the final stage."""
        t = bs.knot_vector(self.n_control_points, self.spline_order)
        c = ca.SX.reshape(ctrl_pts_ts, (nts, self.n_control_points)).T
        theta = x[-2]
        pos_yaw_ref = bs.casadiBsplineVec(theta, t, c, self.spline_order).T
        pos_ref = pos_yaw_ref[:3]
        yaw_ref = pos_yaw_ref[3]
        R_des = R_z(yaw_ref) @ R_x(np.pi)
        c_js = ca.SX.reshape(ctrl_pts_js, (nq, self.n_control_points)).T
        q_ref = bs.casadiBsplineVec(theta, t, c_js, self.spline_order).T
        # q_ref = ca.SX.zeros(nq)
        q_dot_ref = ca.SX.zeros(nq)
        # theta_ref = ca.SX.ones(1)
        theta_dot_ref = ca.SX.zeros(1)
        return ca.vertcat(pos_ref, q_ref, q_dot_ref, theta_ref, theta_dot_ref), R_des

    # TODO: fix this - old version
    # def set_ctr_pts(self, ctr_pts_ts):
    #     if ctr_pts_ts.shape != (self.n_control_points, self.nts):
    #         raise ValueError("ctr_pts_ts should have shape (n_control_points, nts)")

    #     pval = np.reshape(ctr_pts_ts, (self.n_control_points * self.nts), 'F')

    #     for i in range(self.N_horizon):
    #         self.solver.set(i, "p", pval)


    def get_nx(self):
        return self.solver.acados_ocp.dims.nx

    def get_nu(self):
        return self.solver.acados_ocp.dims.nu
    
    def get_status(self):
        return self.solver.get_status()
    

    def forward_kin4D(self, q):
        pos_act, rotmat_act = self.forward_kinematic(q)
        r21 = rotmat_act[1,0]
        r11 = rotmat_act[0,0]
        yaw_act = np.arctan2(r21, r11)
        return np.concatenate([pos_act, [yaw_act]])
    

    def forward_kinematic(self, q):
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        oMact = self.pin_data.oMf[self.tool_frame_id].homogeneous

        pos = oMact[0:3,3]
        rotmat = oMact[0:3,0:3]
        return pos, rotmat
    
      
    def compute_manipulability(self, stage):
        # Get the state at the given stage
        x = self.get_x(stage)
        q = x[:self.nq]  # Extract joint positions

        # Compute the Jacobian at q
        pin.computeJointJacobians(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        # Get the full 6xN Jacobian at the tool frame
        J = pin.getFrameJacobian(self.pin_model, self.pin_data, self.tool_frame_id, pin.LOCAL_WORLD_ALIGNED)

        # Extract translation (3xN) and rotation (3xN) parts
        J_trans = J[:3, :]  # Translational Jacobian
        J_rot = J[3:, :]    # Rotational Jacobian

        # Compute Yoshikawa manipulability measures:
        # Translational: sqrt(det(J_trans * J_trans^T))
        manipulability_trans = np.sqrt(np.linalg.det(J_trans @ J_trans.T))

        # Rotational: sqrt(det(J_rot * J_rot^T))
        manipulability_rot = np.sqrt(np.linalg.det(J_rot @ J_rot.T))

        return {
            "translational": manipulability_trans,
            "rotational": manipulability_rot
        }
    
    # new: untested
    def get_cost(self):
        # return self.solver.cost_get(stage, "yref")
        return self.solver.get_cost()
        
    def get_x(self, stage):
        return self.solver.get(stage, "x")
    
    def get_u(self, stage):
        return self.solver.get(stage, "u")
    
    def get_p(self, stage):
        return self.solver.get(stage, "p")
    
    def get_y(self, stage):
        x = self.get_x(stage)
        u = self.get_u(stage)
        q = x[:self.nq]
        pos_yaw = self.forward_kin4D(q)
        return np.concatenate([pos_yaw, x, u]) # pos, yaw, q, q_dot, theta, theta_dot, u, v
    

    def get_y_ref(self, stage):
        # q_ref = np.zeros(self.nq)
        q_dot_ref = np.zeros(self.nq)
        # theta_ref = [1]
        theta_dot_ref = np.zeros(1)
        qpp_a_ref = np.zeros(7)
        v_ref = [0]

        param = self.get_p(stage)
        param_ts = param[self.id_ctrl_pts_ts:self.id_ctrl_pts_js]
        param_js = param[self.id_ctrl_pts_js:]
        theta_ref = param[self.id_theta_ref]
        x_stage = self.get_x(stage)
        theta_stage = x_stage[-2]

        k = self.spline_order
        t = bs.knot_vector(self.n_control_points, k)
        # c = np.reshape(param, (self.n_control_points, self.nts)).T
        c_ts = np.reshape(param_ts, (self.n_control_points, self.nts))
        pos_yaw_ref = bs.bspline(theta_stage, t, c_ts, k)

        c_js = np.reshape(param_js, (self.n_control_points, self.nq))
        q_ref = bs.bspline(theta_stage, t, c_js, k)

        return np.concatenate([pos_yaw_ref, q_ref, q_dot_ref, theta_ref, theta_dot_ref, qpp_a_ref, v_ref])


    def check_limits(self):
        eps = 1e-4  # small positive tolerance
        limit_violations = False

        for stage in range(self.N_horizon):
            x = self.get_x(stage)
            u = self.get_u(stage)

            q = x[:self.nq]
            q_dot = x[self.nq:-2]
            qpp_a = u[:7]

            # --- q limits ---
            q_hi = q - self.q_ub
            q_lo = self.q_lb - q
            q_violation = np.any(q_hi > eps) or np.any(q_lo > eps)
            if q_violation:
                limit_violations = True
                print(f"Stage {stage}: q limits violated!")
                for i in range(len(q)):
                    if q_hi[i] > eps:
                        print(f"  q[{i}] = {q[i]:.4f} > q_ub[{i}] = {self.q_ub[i]:.4f}")
                    if q_lo[i] > eps:
                        print(f"  q[{i}] = {q[i]:.4f} < q_lb[{i}] = {self.q_lb[i]:.4f}")

            # --- q_dot limits ---
            qp_hi = q_dot - self.qp_ub
            qp_lo = self.qp_lb - q_dot
            qp_violation = np.any(qp_hi > eps) or np.any(qp_lo > eps)
            if qp_violation:
                limit_violations = True
                print(f"Stage {stage}: q_dot limits violated!")
                for i in range(len(q_dot)):
                    if qp_hi[i] > eps:
                        print(f"  q_dot[{i}] = {q_dot[i]:.4f} > qp_ub[{i}] = {self.qp_ub[i]:.4f}")
                    if qp_lo[i] > eps:
                        print(f"  q_dot[{i}] = {q_dot[i]:.4f} < qp_lb[{i}] = {self.qp_lb[i]:.4f}")

            # --- u limits ---
            if np.any(np.abs(qpp_a) > self.u_limit + eps):
                limit_violations = True
                print(f"Stage {stage}: qpp_a limits violated!")
                for i in range(len(qpp_a)):
                    if abs(qpp_a[i]) > self.u_limit + eps:
                        print(f"  qpp_a[{i}] = {qpp_a[i]:.4f} > u_limit = {self.u_limit:.4f}")

        return limit_violations
    

    def initialize_taskspace_consistent(self, q0, qp0, theta0, ctrl_pts_ts, ctrl_pts_js, theta0_dot=0.01):
        """
        Initialize OCP with kinematically consistent taskspace trajectory
        
        Args:
            q0: Initial joint positions [9,]
            qp0: Initial joint velocities [9,] 
            theta0: Initial path parameter
            ctrl_pts_ts: Task space control points [n_control_points, 4] (pos + yaw)
            ctrl_pts_js: Joint space control points [n_control_points, 9]
            theta0_dot: Initial path parameter velocity
        """
        
        dt = self.Tf / self.N_horizon
        
        # Initialize arrays for trajectory
        theta_traj = np.zeros(self.N_horizon + 1)
        theta_dot_traj = np.zeros(self.N_horizon + 1)
        q_traj = np.zeros((self.N_horizon + 1, self.nq))
        q_dot_traj = np.zeros((self.N_horizon + 1, self.nq))
        u_traj = np.zeros((self.N_horizon, self.nu-1))
        v_traj = np.zeros((self.N_horizon, 1))
        
        # Set initial conditions
        theta_traj[0] = theta0
        theta_dot_traj[0] = theta0_dot
        q_traj[0] = np.array(q0)
        q_dot_traj[0] = np.array([qp0])
        
        # Spline setup
        t_knots = bs.knot_vector(self.n_control_points, self.spline_order)
        c_ts = np.reshape(ctrl_pts_ts, (self.n_control_points, self.nts))
        # c_js = np.reshape(ctrl_pts_js, (self.n_control_points, self.nq))
        
        for i in range(self.N_horizon):
            # 1. Integrate theta
            theta_traj[i+1] = np.clip(theta_traj[i] + theta_dot_traj[i] * dt, 0.0, 1.0)
            theta_dot_traj[i+1] = theta0_dot  # Constant for now
            
            # Clamp theta to valid spline domain [0, 1]
            theta_current = theta_traj[i]
            theta_next = theta_traj[i+1]
            
            # 2. Compute task space reference and derivatives
            pos_yaw_ref = bs.bspline(theta_current, t_knots, c_ts, self.spline_order)
            pos_yaw_ref_next = bs.bspline(theta_next, t_knots, c_ts, self.spline_order)
            
            # Task space velocity from spline derivative
            pos_yaw_dot_ref = (pos_yaw_ref_next - pos_yaw_ref) / dt
            
            # Extract components
            pos_ref = pos_yaw_ref[:3]
            yaw_ref = pos_yaw_ref[3]
            pos_dot_ref = pos_yaw_dot_ref[:3]
            yaw_dot_ref = pos_yaw_dot_ref[3]
            
            # 3. Compute Jacobian at current joint configuration
            pin.forwardKinematics(self.pin_model, self.pin_data, q_traj[i])
            pin.computeJointJacobians(self.pin_model, self.pin_data, q_traj[i])
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            
            # Get full 6D Jacobian for tool frame
            J_full = pin.getFrameJacobian(self.pin_model, self.pin_data, 
                                        self.tool_frame_id, pin.LOCAL_WORLD_ALIGNED)
            
            # Extract position and orientation parts
            J_pos = J_full[:3, :self.nq]  # Position Jacobian [3 x nq]
            J_rot = J_full[3:, :self.nq]  # Orientation Jacobian [3 x nq]
            
            # 4. Compute desired task space velocity
            # For orientation: only yaw rotation (around z-axis)
            omega_ref = np.array([0.0, 0.0, yaw_dot_ref])  # Only yaw rotation
            
            # Combined task space velocity [6,]
            v_task_ref = np.concatenate([pos_dot_ref, omega_ref])
            
            # 5. Compute joint velocities using pseudoinverse
            J_task = J_full[:, :self.nq]  # Use full 6D task
            
            # Pseudoinverse with damping for numerical stability
            damping = 1e-6
            J_pinv = J_task.T @ np.linalg.inv(J_task @ J_task.T + damping * np.eye(6))
            
            # Primary task: follow task space trajectory
            q_dot_primary = J_pinv @ v_task_ref
            
            # Nullspace task: regularize joint velocities (avoid singularities, stay near home)
            q_home = np.zeros(self.nq)  # Or use a preferred home configuration
            k_null = 0.1  # Nullspace gain
            q_dot_null = -k_null * (q_traj[i] - q_home)  # Move towards home configuration
            
            # Nullspace projection
            P_null = np.eye(self.nq) - J_pinv @ J_task
            q_dot_secondary = P_null @ q_dot_null
            
            # Combined joint velocity
            q_dot_combined = q_dot_primary #+ q_dot_secondary
            q_dot_traj[i+1] = q_dot_combined
            
            # 6. Integrate joint positions
            q_traj[i+1] = q_traj[i] + q_dot_traj[i+1] * dt
            
            # 7. Compute joint accelerations (control input)
            # Acceleration = (v_new - v_old) / dt
            if i == 0:
                # First step: use initial joint velocities
                q_dot_prev = qp0
            else:
                q_dot_prev = q_dot_traj[i]
            
            q_ddot = (q_dot_traj[i+1] - q_dot_prev) / dt
            
            # Control input: [qpp_a (7 actuated joints), v (path velocity)]
            u_traj[i, :7] = q_ddot[:7]  # First 7 joints are actuated
            # u_traj[i, 7] = theta_dot_traj[i]  # Path velocity
            
            # Clamp controls to limits
            u_traj[i, :7] = np.clip(u_traj[i, :7], -self.u_limit, self.u_limit)
            # u_traj[i, 7] = np.clip(u_traj[i, 7], -1.0, 1.0)
        
        # 8. Set trajectory in Acados solver
        self._set_trajectory_in_solver(q_traj, q_dot_traj, theta_traj, theta_dot_traj, u_traj, v_traj)
        
        # 9. Set parameters
        self._set_parameters_in_solver(ctrl_pts_ts, ctrl_pts_js, theta_traj)
        
        # print(f"Initialized consistent trajectory:")
        # print(f"  theta range: [{theta_traj[0]:.3f}, {theta_traj[-1]:.3f}]")
        # print(f"  q range: [{np.min(q_traj):.3f}, {np.max(q_traj):.3f}]")
        # print(f"  u range: [{np.min(u_traj):.3f}, {np.max(u_traj):.3f}]")


    def _set_trajectory_in_solver(self, q_traj, q_dot_traj, theta_traj, theta_dot_traj, u_traj, v_traj):
        """Set computed trajectory in Acados solver"""
        
        for i in range(self.N_horizon + 1):
            # State: [q, q_dot, theta, theta_dot]
            x_init = np.concatenate([
                q_traj[i],
                q_dot_traj[i], 
                [theta_traj[i], theta_dot_traj[i]]
            ])
            
            if i < self.N_horizon:
                u = np.concatenate([u_traj[i], v_traj[i]])
                self.solver.set(i, "x", x_init)
                self.solver.set(i, "u", u)
            else:
                # Terminal state
                self.solver.set(i, "x", x_init)
        
        # Set initial state constraint
        x0 = np.concatenate([q_traj[0], q_dot_traj[0], [theta_traj[0], theta_dot_traj[0]]])
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)


    def _set_parameters_in_solver(self, ctrl_pts_ts, ctrl_pts_js, theta_traj):
        """Set parameters in Acados solver"""
        
        for i in range(self.N_horizon + 1):
            # Build parameter vector
            p_init = np.zeros(self.ocp.model.p.rows())
            
            # Use default weights
            p_init[self.id_weight_pos:self.id_weight_pos+3] = self.pval_weight_pos_init
            p_init[self.id_weight_rot] = self.pval_weight_rot_init
            p_init[self.id_weight_q:self.id_weight_q+9] = self.pval_weight_q_init
            p_init[self.id_weight_qp:self.id_weight_qp+9] = self.pval_weight_qp_init
            p_init[self.id_weight_theta] = self.pval_weight_theta_init
            p_init[self.id_weight_theta_dot] = self.pval_weight_theta_init_dot
            p_init[self.id_weight_u:self.id_weight_u+7] = self.pval_weight_u_init
            p_init[self.id_weight_v] = self.pval_weight_v_init
            
            # Inertia parameters
            p_init[self.id_mass_value] = self.pval_mass_value_init
            p_init[self.id_com_value:self.id_com_value+3] = self.pval_com_value_init
            
            # Path parameter reference
            p_init[self.id_theta_ref] = theta_traj[i]
            
            # Control points
            pval_ts = np.reshape(ctrl_pts_ts, (self.n_control_points * self.nts))
            pval_js = np.reshape(ctrl_pts_js, (self.n_control_points * self.nq))
            p_init[self.id_ctrl_pts_ts:self.id_ctrl_pts_js] = pval_ts
            p_init[self.id_ctrl_pts_js:] = pval_js
            
            self.solver.set(i, "p", p_init)


    def check_initialization_consistency(self):
        """Verify that initialization satisfies kinematic constraints"""
        
        print("Checking initialization consistency...")
        
        for i in range(min(5, self.N_horizon)):  # Check first 5 stages
            x = self.solver.get(i, "x")
            q = x[:self.nq]
            
            # Forward kinematics
            pin.forwardKinematics(self.pin_model, self.pin_data, q)
            pin.updateFramePlacements(self.pin_model, self.pin_data)
            
            pos_actual = self.pin_data.oMf[self.tool_frame_id].translation
            
            # Get reference from spline
            theta = x[-2]
            param = self.solver.get(i, "p")
            ctrl_pts_ts = param[self.id_ctrl_pts_ts:self.id_ctrl_pts_js]
            
            t_knots = bs.knot_vector(self.n_control_points, self.spline_order)
            c_ts = np.reshape(ctrl_pts_ts, (self.n_control_points, self.nts))
            pos_ref = bs.bspline(theta, t_knots, c_ts, self.spline_order)[:3]
            
            error = np.linalg.norm(pos_actual - pos_ref)
            print(f"Stage {i}: position error = {error:.6f} m")
            
            if error > 0.01:  # 1cm tolerance
                print(f"  WARNING: Large position error at stage {i}")


    # Usage example:
    def update_trajectory_with_taskspace_init(self, new_ctrl_pts_ts, new_ctrl_pts_js, 
                                            current_state, theta_start=0.0):
        """Update trajectory with consistent taskspace initialization"""
        
        q0 = current_state[:self.nq]
        qp0 = current_state[self.nq:2*self.nq]
        
        # Initialize with consistent trajectory
        self.initialize_taskspace_consistent(
            q0=q0,
            qp0=qp0, 
            theta0=theta_start,
            ctrl_pts_ts=new_ctrl_pts_ts,
            ctrl_pts_js=new_ctrl_pts_js,
            theta0_dot=0.1
        )
        
        # Verify consistency
        self.check_initialization_consistency()
        
        # Solve
        status = self.solver.solve()
        
        if status != 0:
            print(f"Solver failed with status {status} despite consistent initialization")
        
        return status


from ..utils.pinutil import print_frame_names
from ..utils.util import homtrans_to_pos_yaw
from ..utils.pinutil import get_frameSE3

def main(algorithm='RTI', as_rti_iter=1):
    mj_model, mj_data = load_mujoco_model(model_path)
    pin_model, pin_data = load_pinocchio_model(model_path)
    tool_frame_id = get_gripper_point_frame_id(pin_model)
    tool_body_id = get_tool_body_id(pin_model)

    # print_frame_names(pin_model)
    # exit()
    q0 = np.array([0, 0, 0, np.pi/2, 0, -np.pi/2, 0, 0, 0])
    joint_offset = np.array([np.pi/2, 0, 0, 0, 0, 0, 0, 0, 0])
    qd = q0 + joint_offset
    qw = q0 + 0.5*joint_offset

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
    ctr_pts_ts = np.array(ctr_pts_ts)
    ctr_pts_js = np.tile(q0, 3)

    print("q0: " + str(q0))
    print("qw: " + str(qw))
    print("qd: " + str(qd))

    qp0 = np.zeros(pin_model.nq)
    u0 = np.zeros(pin_model.nq - 2)
    theta0 = 0
    theta_dot0 = 0
    x0 = np.concatenate([q0, qp0, [theta0, theta_dot0]])

    dt = 0.05
    N_horizon = 50
    Tf = N_horizon * dt

    ocp = OCP(pin_model, pin_data, tool_frame_id, tool_body_id, N_horizon, Tf, q0, qp0, theta0, theta_dot0, algorithm, as_rti_iter, regenerate=True, ctr_pts_js=ctr_pts_js, ctr_pts_ts=ctr_pts_ts)

    print("ctr_pts_ts.shape: ", ctr_pts_ts.shape)
    nx = ocp.get_nx()
    nu = ocp.get_nu()

    Nsim = N_horizon*2
    simX = np.zeros((Nsim+1, nx))
    simU = np.zeros((Nsim, nu))

    simX[0,:] = x0

    if algorithm != "SQP":
        t_preparation = np.zeros((Nsim))
        t_feedback = np.zeros((Nsim))

    else:
        t = np.zeros((Nsim))


    ocp.initialize_taskspace_consistent(
            q0=q0,
            qp0=qp0, 
            theta0=0,
            ctrl_pts_ts=ctr_pts_ts,
            ctrl_pts_js=ctr_pts_js,
            theta0_dot=0.1
        )

    # closed loop
    for i in range(Nsim):

        if algorithm != "SQP":
            # preparation phase
            ocp.solver.options_set('rti_phase', 1)
            status = ocp.solver.solve()
            t_preparation[i] = ocp.solver.get_stats('time_tot')

            # set initial state
            ocp.solver.set(0, "lbx", simX[i, :])
            ocp.solver.set(0, "ubx", simX[i, :])

            # feedback phase
            ocp.solver.options_set('rti_phase', 2)
            status = ocp.solver.solve()
            t_feedback[i] = ocp.solver.get_stats('time_tot')

            simU[i, :] = ocp.solver.get(0, "u")

        else:
            # solve ocp and get next control input
            simU[i,:] = ocp.solver.solve_for_x0(x0_bar = simX[i, :])

            t[i] = ocp.solver.get_stats('time_tot')

        # simulate system
        simX[i+1, :] = ocp.integrator.simulate(x=simX[i, :], u=simU[i,:])

    # evaluate timings
    if algorithm != "SQP":
        # scale to milliseconds
        t_preparation *= 1000
        t_feedback *= 1000
        print(f'Computation time in preparation phase in ms: \
                min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}')
        print(f'Computation time in feedback phase in ms:    \
                min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}')
    else:
        # scale to milliseconds
        t *= 1000
        print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')

    # plot results
    plot_pfc_results(np.linspace(0, (Tf/N_horizon)*Nsim, Nsim+1), ocp.u_limit, simU, simX, title=algorithm)

    ocp.solver = None


if __name__ == '__main__':
    
    main(algorithm="RTI", as_rti_iter=1)

    # for algorithm in ["SQP", "RTI", "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]:
    #     main(algorithm=algorithm, as_rti_iter=1)
