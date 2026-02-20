# cranebrain/imitation/mpc_expert.py
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Optional SB3 type aliases
try:
    from stable_baselines3.common.type_aliases import GymAct, GymObs
except Exception:
    GymObs = Any
    GymAct = Any

from cranebrain.common.load_model import (
    get_gripper_point_frame_id,
    get_tool_body_id,
    load_pinocchio_model,
)
from cranebrain.mpc.mpc import MPC

# Utilities
from cranebrain.utils.pinutil import get_frameSE3
from cranebrain.utils.util import homtrans_to_pos_yaw


class MPCExpertPolicy:
    def __init__(
        self,
        model_path: str,
        env,
        Ts: Optional[float] = None,
        N_horizon: int = 60,
        regenerate: bool = True,
        N_path_pts=50,
    ):
        self.env = env
        self.dt = float(env.dt if Ts is None else Ts)
        self.N_horizon = int(N_horizon)
        self.Tf = self.N_horizon * self.dt

        # Pinocchio model for MPC
        self.pin_model, self.pin_data = load_pinocchio_model(model_path)
        self.tool_frame_id = get_gripper_point_frame_id(self.pin_model)
        self.tool_body_id = get_tool_body_id(self.pin_model)

        # Initial placeholders; real state comes from obs
        q0_zeros = np.zeros(self.pin_model.nq)
        qp0_zeros = np.zeros(self.pin_model.nq)

        # MPC instance
        self.controller = MPC(
            pin_model=self.pin_model,
            pin_data=self.pin_data,
            tool_frame_id=self.tool_frame_id,
            tool_body_id=self.tool_body_id,
            dt=self.dt,
            N_horizon=self.N_horizon,
            q0=q0_zeros,
            qp0=qp0_zeros,
            regenerate=regenerate,
        )

        # Warm-start container for [qpp_a(7), v]
        self._last_u = np.zeros(8)
        self._init_done = False
        self.N_path = N_path_pts
        self.i = 0

    # ------------------------------------------------------------------
    # SB3-like API
    # ------------------------------------------------------------------
    def predict(
        self, observation: GymObs, deterministic: bool = True
    ) -> Tuple[GymAct, Optional[Any]]:
        """
        Returns action = [qpp_a(7), v]  (shape (8,)).
        """
        q, qdot = self._obs_to_state(observation)

        # use local mpc state (theta, theta_dot) as environment is not reset with new path
        x0 = np.concatenate([q, qdot])

        # Push current theta to OCP params
        # self.controller.set_ocp_param(mass, com))

        # First call: reset/warm-start around current state
        if not self._init_done:
            self.controller.reset(x0, x0)
            self._init_done = True

        if self._last_u is None or self._last_u.shape[0] != 8:
            self._last_u = np.zeros(8)

        # One MPC iteration
        u1, info, dq = self._solve_mpc_step(x0)
        self._last_u = u1.copy()

        action = dq[1]

        return action, info, dq

    def __call__(self, observation: GymObs) -> GymAct:
        act, _ = self.predict(observation, deterministic=True)
        return act

    def reset(self, q_init: np.ndarray, qd_init: np.ndarray):
        """External reset (e.g., episode start)."""
        self._init_done = False
        self._last_u = np.zeros(8)
        self.q_init = q_init.copy()
        self.qd_init = qd_init.copy()
        self.i = 0

        T0 = get_frameSE3(
            self.pin_model, self.pin_data, self.q_init, self.tool_frame_id
        ).homogeneous
        ts_start = homtrans_to_pos_yaw(T0)
        self.p_init = ts_start[:3]
        self.yaw_init = ts_start[3]
        self.p_goal = self.p_init
        self.yaw_goal = self.yaw_init
        return ts_start

    # ------------------------------------------------------------------
    # Task / ctrl-points management
    # ------------------------------------------------------------------
    def update_task(
        self,
        pos: np.ndarray,
        yaw: np.ndarray,
    ):
        self.p_goal = pos
        self.yaw_goal = yaw
        self.i = 0

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _obs_to_state(self, obs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # q_norm = obs.q_norm
        # q_dot_norm = obs.q_dot_norm
        # q = self.env.denormalize(
        #     q_norm, self.env.qpos_min, self.env.qpos_max
        # )
        # qdot = self.env.denormalize(
        #     q_dot_norm, self.env.qvel_min, self.env.qvel_max
        # )
        q = self.env.mj_data.qpos[: self.env.dof]
        qdot = self.env.mj_data.qvel[: self.env.dof]

        return q, qdot

    def _solve_mpc_step(self, x0: np.ndarray):
        q0 = x0[:9]
        qp0 = x0[9:18]
        u0 = self._last_u[:7] if self._last_u is not None else np.zeros(7)

        if self.N_path > 0:
            self.p_ref = self.p_init + (self.p_goal - self.p_init) * float(
                self.i / self.N_path
            )
            self.yaw_ref = self.yaw_init + (self.yaw_goal - self.yaw_init) * float(
                self.i / self.N_path
            )
            self.i = np.clip(self.i + 1, 0, self.N_path)
        else:
            self.p_ref = self.p_goal
            self.yaw_ref = self.yaw_goal

        _t, _q, dq, _u, u1, x1 = self.controller.iterate(
            t0=0.0,
            q0=q0,
            qp0=qp0,
            u0=u0,
            p_ref=self.p_ref,
            yaw_ref=self.yaw_ref,
            q_ref=None,
            qp_ref=None,
            u_ref=None,
        )
        status = self.controller.status
        x0 = np.concatenate([q0, qp0])
        y_ref = self.controller.get_y_ref(self.p_ref, self.yaw_ref, x0, u0)
        y_act = self.controller.get_y_act(x0, u0)

        p_err = y_act[:3] - y_ref[:3]
        yaw_err = y_act[3] - y_ref[3]

        t_prep = getattr(self.controller, "t_preparation", 0.0)
        t_fb = getattr(self.controller, "t_feedback", 0.0)
        info = {
            "t_preparation": t_prep,
            "t_feedback": t_fb,
            "status": status,
            # "p_ref": self.p_ref,
            # "yaw_ref": self.yaw_ref,
            "y_act": y_act,
            "y_ref": y_ref,
            "pos_err": p_err,
            "yaw_err": yaw_err,
        }
        return u1, info, dq
