import os
import sys

import casadi as ca
import numpy as np
import scipy.linalg
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

# import pinutil as pu
from colorama import Fore, Style, init
from pinocchio import casadi as cpin

from ..common.load_model import (
    get_gripper_point_frame_id,
    get_tool_body_id,
    load_mujoco_model,
    load_pinocchio_model,
    model_path,
)
from ..utils.plotter import plot_pfc_results, plot_results

# sys.path.insert(0, '../common')
# sys.path.insert(0, '../utils')
from .crane_model import export_robocrane_ode_model


class OCP:
    REAL_TIME_ALGORITHMS = ["RTI", "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]
    ALGORITHMS = ["SQP"] + REAL_TIME_ALGORITHMS

    def __init__(
        self,
        cpin_model,
        cpin_data,
        tool_frame_id,
        tool_body_id,
        N_horizon,
        Tf,
        q0,
        qp0,
        algorithm="SQP",
        as_rti_iter=1,
        regenerate=False,
        ctr_pts=None,
    ):
        # limits
        self.q_ub = np.array([170, 120, 170, 120, 170, 120, 175, 45, 45]) * np.pi / 180
        self.q_lb = -self.q_ub
        self.qp_ub = (
            np.array([85, 85, 100, 75, 130, 135, 135, 50000, 50000]) * np.pi / 180
        )
        self.qp_lb = -self.qp_ub
        self.u_limit = 3

        self.algorithm = algorithm
        self.as_rti_iter = as_rti_iter
        self.nq = 9

        ocp = AcadosOcp()

        # set model
        model, p_val = export_robocrane_ode_model(
            cpin_model, cpin_data, tool_frame_id, tool_body_id
        )
        ocp.model = model
        ocp.parameter_values = p_val

        nx = model.x.rows()
        nu = model.u.rows()
        ny = nx + nu
        ny_e = nx
        dof = int(cpin_model.nq)

        ocp.dims.N = int(N_horizon)

        ocp.solver_options.N_horizon = int(N_horizon)
        self.N_horizon = N_horizon

        print("q0: " + str(q0))
        print("qp0: " + str(qp0))
        x0 = np.concatenate([q0, qp0])

        ## Nonlinear LS

        # lagrange cost
        ocp.cost.cost_type = "NONLINEAR_LS"

        q = model.x[:dof]
        q_dot = model.x[dof:]
        cpin.forwardKinematics(cpin_model, cpin_data, q)
        cpin.updateFramePlacements(cpin_model, cpin_data)
        oMact = cpin_data.oMf[tool_frame_id].homogeneous

        pos_act = oMact[0:3, 3]
        rotmat_act = oMact[0:3, 0:3]
        r21 = rotmat_act[1, 0]
        r11 = rotmat_act[0, 0]
        yaw_act = ca.atan2(r21, r11)

        ocp.model.cost_y_expr = ca.vertcat(pos_act, yaw_act, q, q_dot, model.u)
        # ocp.model.cost_y_expr = ca.vertcat(pos_act, q_dot, model.u)
        p0 = np.zeros(3)
        yaw0 = np.zeros(1)

        print(Fore.GREEN + "p0.shape: {}".format(p0.shape) + Style.RESET_ALL)
        print(Fore.GREEN + "x0.shape: {}".format(x0.shape) + Style.RESET_ALL)
        ocp.cost.yref = np.concatenate(
            [p0, yaw0, np.zeros(dof), np.zeros(dof), np.zeros(7)]
        )
        # Q = np.eye(4+dof)
        weight_pos = 3 * [1]
        weight_rot = [0.5]
        weight_q = [0, 0, 0.1, 0, 0.1, 0, 0, 0.2, 0.2]
        weight_qdot = np.concatenate(
            [7 * [0.1], 2 * [2.0]]
        )  # np.concatenate([0.1*np.ones(2), 1.0*np.ones(1), 0.1*np.ones(1), 1.0*np.ones(1), 0.1*np.ones(2), 0.2*np.ones(2)])
        weight_u = 7 * [0.15]
        Q = np.diag(np.concatenate([weight_pos, weight_rot, weight_q, weight_qdot]))
        R = np.diag(weight_u)
        ocp.cost.W = scipy.linalg.block_diag(Q, R)

        # mayer cost
        # ocp.cost.cost_type_e = "NONLINEAR_LS"
        # # ocp.model.cost_y_expr_e = ca.vertcat(pos_act, yaw, q_dot)
        # ocp.model.cost_y_expr_e = ca.vertcat(pos_act, yaw_act, q, q_dot)
        # ocp.cost.yref_e = np.concatenate([p0, yaw0, np.zeros(dof), np.zeros(dof)])
        # ocp.cost.W_e = Q

        # set the constraints
        ocp.constraints.lbu = np.concatenate([-self.u_limit * np.ones(nu)])
        ocp.constraints.ubu = np.concatenate([self.u_limit * np.ones(nu)])
        ocp.constraints.x0 = x0
        ocp.constraints.idxbu = np.arange(nu)

        ocp.constraints.lbx = np.concatenate([self.q_lb, self.qp_lb])
        ocp.constraints.ubx = np.concatenate([self.q_ub, self.qp_ub])
        ocp.constraints.idxbx = np.arange(nx)

        ocp.solver_options.qp_solver = (
            "PARTIAL_CONDENSING_HPIPM"  # FULL_CONDENSING_QPOASES
        )
        ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
        # ocp.solver_options.hessian_approx = 'EXACT'
        ocp.solver_options.integrator_type = "IRK"

        ocp.solver_options.qp_solver_warm_start = 1

        # ocp.translate_nls_cost_to_conl()

        if algorithm in OCP.REAL_TIME_ALGORITHMS:
            ocp.solver_options.nlp_solver_type = "SQP_RTI"
        else:
            ocp.solver_options.nlp_solver_type = "SQP"

        if algorithm == "AS-RTI-A":
            ocp.solver_options.as_rti_iter = as_rti_iter
            ocp.solver_options.as_rti_level = 0
        elif algorithm == "AS-RTI-B":
            ocp.solver_options.as_rti_iter = as_rti_iter
            ocp.solver_options.as_rti_level = 1
        elif algorithm == "AS-RTI-C":
            ocp.solver_options.as_rti_iter = as_rti_iter
            ocp.solver_options.as_rti_level = 2
        elif algorithm == "AS-RTI-D":
            ocp.solver_options.as_rti_iter = as_rti_iter
            ocp.solver_options.as_rti_level = 3

        ocp.solver_options.qp_solver_cond_N = N_horizon

        # set prediction horizon
        ocp.solver_options.tf = Tf

        solver_json = "acados_ocp_" + model.name + ".json"
        try:
            acados_ocp_solver = AcadosOcpSolver(
                ocp, json_file=solver_json, build=regenerate, generate=regenerate
            )
        except OSError as e:
            print(Fore.RED + "Could not open {}".format(e) + Style.RESET_ALL)
            regenerate = True
            acados_ocp_solver = AcadosOcpSolver(
                ocp, json_file=solver_json, build=regenerate, generate=regenerate
            )

        # create an ocp.integrator with the same settings as used in the OCP solver.
        acados_integrator = AcadosSimSolver(ocp, json_file=solver_json)

        self.solver = acados_ocp_solver
        self.integrator = acados_integrator

        # return acados_ocp_solver, acados_integrator

    def get_nx(self):
        return self.solver.acados_ocp.dims.nx

    def get_nu(self):
        return self.solver.acados_ocp.dims.nu

    def set_ocp_param(self, mass, com):
        self.solver.set(0, "p", np.array([mass, com[0], com[1], com[2]]))
        for j in range(1, self.N_horizon + 1):
            self.solver.set(j, "p", np.array([mass, com[0], com[1], com[2]]))

    def cost_update_ref(self, p_ref, yaw_ref, q_ref, qp_ref, u_ref):
        yref = np.concatenate([p_ref, [yaw_ref], q_ref, qp_ref, u_ref])
        yref_e = np.concatenate([p_ref, [yaw_ref], q_ref, qp_ref])
        for j in range(self.N_horizon):
            self.solver.set(j, "yref", yref)

        # self.solver.set(self.N_horizon, "yref", yref_e)

    def cost_update_ref_full_horizon(
        self, p_ref_ls, yaw_ref_ls, q_ref_ls, qp_ref_ls, u_ref_ls
    ):
        if (
            len(p_ref_ls) != self.N_horizon
            or len(yaw_ref_ls) != self.N_horizon
            or len(q_ref_ls) != self.N_horizon
            or len(qp_ref_ls) != self.N_horizon
            or len(u_ref_ls) != self.N_horizon
        ):
            print(
                Fore.RED
                + "Error: Reference trajectory length should be equal to the horizon length"
                + Style.RESET_ALL
            )
            return

        for j in range(self.N_horizon):
            p_ref = p_ref_ls[j]
            yaw_ref = yaw_ref_ls[j]
            q_ref = q_ref_ls[j]
            qp_ref = qp_ref_ls[j]
            u_ref = u_ref_ls[j]
            yref = np.concatenate([p_ref, yaw_ref, q_ref, qp_ref, u_ref])
            self.solver.set(j, "yref", yref)

        # yref_e = np.concatenate([p_ref, yaw_ref, q_ref, q_ref, qp_ref])
        # self.solver.set(self.N_horizon, "yref", yref_e)

    def get_cost(self):
        cost = self.solver.get_cost()
        return cost


import pinocchio as pin


def get_pos_and_yaw(pin_model, pin_data, frame_id, q):
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    oMact = pin_data.oMf[frame_id].homogeneous

    pos = oMact[0:3, 3]
    rotmat = oMact[0:3, 0:3]
    r21 = rotmat[1, 0]
    r11 = rotmat[0, 0]
    yaw = np.array([np.arctan2(r21, r11)])

    # p =  np.concatenate([pos, yaw])
    return pos, yaw


def get_position(pin_model, pin_data, frame_id, q):
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    oMact = pin_data.oMf[frame_id].homogeneous

    pos = oMact[0:3, 3]
    return pos


def main(algorithm="RTI", as_rti_iter=1):
    mj_model, mj_data = load_mujoco_model(model_path)
    pin_model, pin_data = load_pinocchio_model(model_path)
    tool_frame_id = get_gripper_point_frame_id(pin_model)
    tool_body_id = get_tool_body_id(pin_model)

    cpin_model = cpin.Model(pin_model)
    cpin_data = cpin_model.createData()

    q0 = np.array([0, 0, 0, np.pi / 2, 0, -np.pi / 2, 0, 0, 0])
    joint_offset = np.array([np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0])
    qw = q0 + joint_offset
    qd = qw

    # p0 = get_pos_and_yaw(pin_model, pin_data, tool_frame_id, q0)
    p0 = get_position(pin_model, pin_data, tool_frame_id, q0)

    print("q0: " + str(q0))
    print("p0: " + str(p0))
    print("qw: " + str(qw))
    print("qd: " + str(qd))

    qp0 = np.zeros(pin_model.nq)
    u0 = np.zeros(pin_model.nq - 2)
    x0 = np.concatenate([q0, qp0])

    dt = 0.05
    N_horizon = 50
    Tf = N_horizon * dt

    ocp = OCP(
        cpin_model,
        cpin_data,
        tool_frame_id,
        tool_body_id,
        N_horizon,
        Tf,
        q0,
        qp0,
        algorithm,
        as_rti_iter,
        regenerate=True,
    )

    nx = ocp.get_nx()
    nu = ocp.get_nu()

    Nsim = N_horizon * 2
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))

    simX[0, :] = x0

    if algorithm != "SQP":
        t_preparation = np.zeros((Nsim))
        t_feedback = np.zeros((Nsim))

    else:
        t = np.zeros((Nsim))

    # closed loop
    for i in range(Nsim):
        if algorithm != "SQP":
            # preparation phase
            ocp.solver.options_set("rti_phase", 1)
            status = ocp.solver.solve()
            t_preparation[i] = ocp.solver.get_stats("time_tot")

            # set initial state
            ocp.solver.set(0, "lbx", simX[i, :])
            ocp.solver.set(0, "ubx", simX[i, :])

            # feedback phase
            ocp.solver.options_set("rti_phase", 2)
            status = ocp.solver.solve()
            t_feedback[i] = ocp.solver.get_stats("time_tot")

            simU[i, :] = ocp.solver.get(0, "u")

        else:
            # solve ocp and get next control input
            simU[i, :] = ocp.solver.solve_for_x0(x0_bar=simX[i, :])

            t[i] = ocp.solver.get_stats("time_tot")

        # simulate system
        simX[i + 1, :] = ocp.integrator.simulate(x=simX[i, :], u=simU[i, :])

    # evaluate timings
    if algorithm != "SQP":
        # scale to milliseconds
        t_preparation *= 1000
        t_feedback *= 1000
        print(
            f"Computation time in preparation phase in ms: \
                min {np.min(t_preparation):.3f} median {np.median(t_preparation):.3f} max {np.max(t_preparation):.3f}"
        )
        print(
            f"Computation time in feedback phase in ms:    \
                min {np.min(t_feedback):.3f} median {np.median(t_feedback):.3f} max {np.max(t_feedback):.3f}"
        )
    else:
        # scale to milliseconds
        t *= 1000
        print(
            f"Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}"
        )

    # plot results
    plot_results(
        np.linspace(0, (Tf / N_horizon) * Nsim, Nsim + 1),
        ocp.u_limit,
        simU,
        simX,
        title=algorithm,
    )

    ocp.solver = None
    input("press enter to exit..")


if __name__ == "__main__":
    main(algorithm="RTI", as_rti_iter=1)

    # for algorithm in ["SQP", "RTI", "AS-RTI-A", "AS-RTI-B", "AS-RTI-C", "AS-RTI-D"]:
    #     main(algorithm=algorithm, as_rti_iter=1)
