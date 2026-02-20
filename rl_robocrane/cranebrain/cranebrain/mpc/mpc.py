import os
import sys
from importlib.resources import files  # for accessing casadi data files

#!/usr/bin/env python
import numpy as np
import pinocchio as pin
from acados_template import (
    AcadosModel,
    AcadosOcp,
    AcadosOcpSolver,
    AcadosSim,
    AcadosSimSolver,
)
from casadi import *
from colorama import Fore, Style, init
from pinocchio import casadi as cpin
from scipy.interpolate import make_interp_spline
from scipy.linalg import block_diag

from ..common.load_model import (
    get_gripper_point_frame_id,
    get_tool_body_id,
    load_mujoco_model,
    load_pinocchio_iiwa_model,
    load_pinocchio_model,
    model_path,
)
from ..control.controller_pinocchio import CTcontrol, K0_joint, K1_joint, KI_joint
from ..utils.plotter import plot_pfc_results, plot_results

# sys.path.insert(0, '../common')
# sys.path.insert(0, '../utils')
from .ocp import OCP, get_pos_and_yaw, get_position


class MPC:
    def __init__(
        self,
        pin_model,
        pin_data,
        tool_frame_id,
        tool_body_id,
        dt,
        N_horizon,
        q0,
        qp0,
        regenerate=True,
    ):
        self.dof_u = 2
        self.dof = pin_model.nq
        self.dof_a = self.dof - self.dof_u

        # pinocchio casadi
        self.pin_model = pin_model
        self.pin_data = pin_data

        # pinocchio casadi
        self.cpin_model = cpin.Model(pin_model)
        self.cpin_data = self.cpin_model.createData()
        self.tool_frame_id = tool_frame_id
        self.tool_body_id = tool_body_id

        self.dt = dt
        self.N_horizon = N_horizon
        self.Tf = N_horizon * dt

        self.ocp = OCP(
            self.cpin_model,
            self.cpin_data,
            self.tool_frame_id,
            self.tool_body_id,
            N_horizon,
            self.Tf,
            q0,
            qp0,
            algorithm="RTI",
            as_rti_iter=1,
            regenerate=regenerate,
        )

        self.plot_horizon = False
        self.init = True

        self.t_preparation = 0
        self.t_feedback = 0

        inertia = self.pin_model.inertias[tool_body_id]
        self.pval_mass = np.array([inertia.mass])
        self.pval_com = np.array([inertia.lever]).flatten()

    def set_inertia(self, mass, com):
        self.pval_mass = np.array([mass])
        self.pval_com = np.array([com]).flatten()
        self.ocp.set_ocp_param(mass=self.pval_mass, com=self.pval_com)

    def get_opt_time(self):
        return self.t_preparation + self.t_feedback

    def reset(self, x0, x_ref):
        x_traj = np.linspace(x0, x_ref, self.N_horizon)
        for i in range(self.N_horizon):
            self.ocp.solver.set(i, "x", x_traj[i])

    def hard_reset(self, x0: np.ndarray):
        """
        Fully reset acados memories so a bad QP (e.g., NaNs) cannot poison later solves.
        Keeps generated code; recreates the solver only if needed.
        """
        s = self.ocp.solver
        s.reset()  # clears internal memories & warm starts

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
            self.set_ocp_param(theta0=x0[-2])  # your param packer
            s.set_p_global_and_precompute_dependencies()  # precompute (if present)
        except Exception:
            pass

        # 7) Make the MPC do a fresh horizon init on next iterate()
        self.init = True

    def set_reference(self, p_ref, yaw_ref, q_ref, qp_ref, u_ref):
        self.ocp.cost_update_ref(p_ref, yaw_ref, q_ref, qp_ref, u_ref)

    def set_reference_full_horizon(
        self, p_ref_ls, yaw_ref_ls, q_ref_ls, qp_ref_ls, u_ref_ls
    ):
        self.ocp.cost_update_ref_full_horizon(
            p_ref_ls, q_ref_ls, qp_ref_ls, yaw_ref_ls, u_ref_ls
        )

    def get_y_ref(self, pd, yawd, x, u):
        p_ref = np.concatenate([pd, [yawd]])
        q_dot_ref = np.zeros(self.pin_model.nq)
        qpp_a_ref = np.zeros(7)
        y_ref = np.hstack([p_ref, q_dot_ref, qpp_a_ref])
        return y_ref

    def get_y_act(self, x, u):
        q = x[: self.dof]
        q_dot = x[self.dof :]
        qpp_a = u[: self.dof_a]

        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        oMact = self.pin_data.oMf[self.tool_frame_id].homogeneous

        pos_act = oMact[0:3, 3]
        rotmat_act = oMact[0:3, 0:3]
        r21 = rotmat_act[1, 0]
        r11 = rotmat_act[0, 0]
        yaw_act = np.arctan2(r21, r11)
        # p_act = np.concatenate([pos_act, [yaw_act]])
        p_act = pos_act

        y_act = np.hstack([p_act, yaw_act, q_dot, qpp_a])
        return y_act

    def get_cost(self):
        return self.ocp.get_cost()

    def predict(self, t0, x0, u0):
        t1 = t0 + self.dt
        x1 = self.ocp.integrator.simulate(x=x0, u=u0)
        return t1, x1

    def ocp_solve(self, t1, x1):
        if self.ocp.algorithm != "SQP":
            # preparation phase
            self.ocp.solver.options_set("rti_phase", 1)
            status = self.ocp.solver.solve()
            t_preparation = self.ocp.solver.get_stats("time_tot")

            # set initial state
            self.ocp.solver.set(0, "lbx", x1)
            self.ocp.solver.set(0, "ubx", x1)

            # feedback phase
            self.ocp.solver.options_set("rti_phase", 2)
            status = self.ocp.solver.solve()
            t_feedback = self.ocp.solver.get_stats("time_tot")
            u1 = self.ocp.solver.get(0, "u")

        else:
            # solve ocp and get next control input
            # print("x1: ", x1)
            u1 = self.ocp.solver.solve_for_x0(x0_bar=x1)
            t_preparation = 0
            t_feedback = self.ocp.solver.get_stats("time_tot")

        return u1, t_preparation, t_feedback, status

    def get_trajectory(self, t_horizon, x_horizon, u_horizon):
        t_traj = [(t) for t in t_horizon]
        q_traj = [x_horizon[i][: self.dof] for i in range(self.N_horizon)]
        qp_traj = [x_horizon[i][self.dof :] for i in range(self.N_horizon)]
        u_traj = [u_horizon[i][: self.dof_a] for i in range(self.N_horizon)]
        return t_traj, q_traj, qp_traj, u_traj

    def get_result(self, x_horizon, u_horizon):
        # Extract the first value of x and u in the horizon
        x_first = x_horizon[1]
        u_first = u_horizon[1]
        return u_first, x_first

    def iterate(
        self,
        t0,
        q0,
        qp0,
        u0,
        p_ref=None,
        yaw_ref=None,
        q_ref=None,
        qp_ref=None,
        u_ref=None,
    ):
        x0 = np.concatenate([q0, qp0])
        # xf = np.concatenate([qd, np.zeros(self.dof)])
        if p_ref is None:
            raise Exception("p_ref not set!")
        if yaw_ref is None:
            raise Exception("yaw_ref not set!")
        if q_ref is None:
            q_ref = np.zeros(self.dof)
        if qp_ref is None:
            qp_ref = np.zeros(self.dof)
        if u_ref is None:
            u_ref = np.zeros(self.dof_a)

        if isinstance(p_ref, list):
            self.set_reference_full_horizon(p_ref, yaw_ref, q_ref, qp_ref, u_ref)
        else:
            self.set_reference(p_ref, yaw_ref, q_ref, qp_ref, u_ref)

        if self.init:
            x0 = np.concatenate([q0, qp0])
            self.reset(x0, x0)
            self.init = False

        t0_post = t0
        x0_post = x0
        u0, self.t_preparation, self.t_feedback, self.status = self.ocp_solve(
            t0_post, x0_post
        )
        t_horizon, x_horizon, u_horizon = self.get_ocp_result()
        t_traj, q_traj, qp_traj, u_traj = self.get_trajectory(
            t_horizon, x_horizon, u_horizon
        )
        u1, x1 = self.get_result(x_horizon, u_horizon)

        if self.plot_horizon:
            u_horizon = u_horizon[:-1]
            plot_results(
                t_horizon,
                self.ocp.u_limit,
                np.array(u_horizon),
                np.array(x_horizon),
                title=self.ocp.algorithm,
                plt_show=True,
            )
            input("Wait for user input...")

        return t_traj, q_traj, qp_traj, u_traj, u0, x1

    def simulate(self, x0, u0):
        x1 = self.ocp.integrator.simulate(x=x0, u=u0)
        return x1

    def get_ocp_result(self):
        # Get the solution
        t = np.linspace(0, self.Tf, self.N_horizon)
        x = []  # q, qp, theta
        u = []  # qpp (u), theta_dot (v)
        for i in range(self.N_horizon):
            x.append(self.ocp.solver.get(i, "x"))
            u.append(self.ocp.solver.get(i, "u"))

        return t, x, u


import time

import mujoco
import mujoco.viewer


def main(algorithm, as_rti_iter):
    mj_model, mj_data = load_mujoco_model(model_path)
    pin_model, pin_data = load_pinocchio_model(model_path)
    pin_iiwa_model, pin_iiwa_data = load_pinocchio_iiwa_model(model_path)
    tool_frame_id = get_gripper_point_frame_id(pin_model)
    tool_body_id = get_tool_body_id(pin_model)

    q0 = np.array([0, 0, 0, np.pi / 2, 0, -np.pi / 2, 0, 0, 0])
    joint_offset = np.array([np.pi / 2, 0, 0, 0, 0, 0, 0, 0, 0])
    qw = q0 + joint_offset
    qd = qw

    print("q0: " + str(q0))
    print("qw: " + str(qw))
    print("qd: " + str(qd))

    qp0 = np.zeros(pin_model.nq)
    u0 = np.zeros(pin_model.nq - 2)  # qpp_a (u)
    x0 = np.concatenate([q0, qp0])

    dt = 0.03
    Ts = 1e-3
    N_horizon = 30
    Tf = N_horizon * dt

    controller = MPC(
        pin_model,
        pin_data,
        tool_frame_id,
        tool_body_id,
        dt,
        N_horizon,
        q0,
        qp0,
        regenerate=True,
    )
    ct_controller = CTcontrol(pin_iiwa_model, dt, K0_joint, K1_joint, KI_joint)

    t = np.linspace(0, 30 * Tf, 30 * N_horizon)
    frameskip = int(dt / Ts)

    t_exec = []

    x_res = []
    u_res = []

    m = mj_model
    d = mj_data
    with mujoco.viewer.launch_passive(m, d) as viewer:
        d.qpos[: pin_model.nq] = q0
        qp0 = np.zeros_like(q0)

        q_d = q0[:7].copy()
        q_dot_d = np.zeros_like(q_d)

        pd, yawd = get_pos_and_yaw(pin_model, pin_data, tool_frame_id, qd)

        for t0 in t:
            if not viewer.is_running():
                break

            start = time.time()
            t_traj, q_traj, qp_traj, u_traj, u1, x1 = controller.iterate(
                t0, q0, qp0, u0, pd, yawd
            )
            end = time.time()
            t_exec.append(end - start)

            for i in range(frameskip):
                q_dotdot_d = u1[:7]
                q_dot_d = q_dot_d + q_dotdot_d * Ts
                q_d = q_d + q_dot_d * Ts

                tau = ct_controller.update(q0[:7], qp0[:7], q_d, q_dot_d, q_dotdot_d)
                d.ctrl[: pin_model.nq - 2] = tau
                mujoco.mj_step(m, d)
                q0 = d.qpos[: pin_model.nq].copy()
                qp0 = d.qvel[: pin_model.nq].copy()

            y_act = controller.get_y_act(x1, u1)
            y_ref = controller.get_y_ref(pd, yawd, x1, u1)
            # print("y_act: ", y_act)
            # print("y_ref: ", y_ref)

            x_res.append(x1)
            u_res.append(u1)

            # q0 = x1[:9]
            # qp0 = x1[9:]
            # u0 = u1[:]

            viewer.sync()

            print("---")
            print("t0: " + str(t0))
            print("q0: ", np.round(q0, 2).T)
            print("qp0: ", np.round(qp0, 2).T)
            print("u0: ", np.round(u0, 2).T)
            # print("q_cost_scale: ", np.round(q_scaler,2))

    print("Average execution time: " + str(np.mean(t_exec)))
    print("std execution time: " + str(np.std(t_exec)))
    print("max execution time: " + str(np.max(t_exec)))
    print("min execution time: " + str(np.min(t_exec)))

    print("plot results...")
    u_res = np.array(u_res)
    x_res.append(x_res[-1])  # workaround: x 1 longer than u for plotting
    x_res = np.array(x_res)
    t_res = np.append(t, t[-1])  # np.linspace(0, (Tf/N_horizon)*N_horizon, N_horizon+1)
    print("u_res.shape: ", u_res.shape)
    print("x_res.shape: ", x_res.shape)
    print("t_res.shape: ", t_res.shape)
    plot_results(t_res, controller.ocp.u_limit, u_res, x_res, title=algorithm)

    input("press enter to exit..")


if __name__ == "__main__":
    main(algorithm="RTI", as_rti_iter=1)
