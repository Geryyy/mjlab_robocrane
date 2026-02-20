# robocrane_env_min.py
from collections import deque

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np
import pinocchio as pin
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

from BaseRobocraneEnv import BaseRobocraneEnv
from cranebrain.common.load_model import (
    get_gripper_mj_geom_id,
    get_gripper_point_frame_id,
    load_pinocchio_iiwa_model,
    load_pinocchio_model,
)

from JointRobocraneEnv import JointRobocraneEnv


class RealJointRobocraneEnv(JointRobocraneEnv):
    """
    Real-robot version of the JointRobocraneEnv:
    - No physics stepping
    - No torque application
    - MuJoCo acts as analytical model fed by real robot state
    - build_obs() and compute_reward() remain identical
    """

    def __init__(self, *args, **kwargs):
        # No randomization in hardware
        kwargs.update(dict(
            randomize_body_com=False,
            randomize_hfield=False,
            penalize_limits=False,
        ))
        super().__init__(*args, **kwargs)

        self.step_count = 0
        self._prev_action = np.zeros(self.dof_act, dtype=np.float32)

    # ----------------------------------------------------
    # Real robot → env state injection
    # ----------------------------------------------------
    def update_from_robot(self, q, q_dot, tau_meas):
        """
        Inject real robot state into MuJoCo.
        After this, build_obs() will produce the same obs as in sim.
        """

        q = np.asarray(q, dtype=np.float64)
        q_dot = np.asarray(q_dot, dtype=np.float64)
        tau_meas = np.asarray(tau_meas, dtype=np.float64)

        # Accept q of length 7 or 9
        if q.shape[0] == 7:
            q_full = np.zeros(self.dof, dtype=np.float64)
            q_full[:7] = q
        else:
            q_full = q[:self.dof]

        if q_dot.shape[0] == 7:
            qdot_full = np.zeros(self.dof, dtype=np.float64)
            qdot_full[:7] = q_dot
        else:
            qdot_full = q_dot[:self.dof]

        # Inject state
        self.mj_data.qpos[:self.dof] = q_full
        self.mj_data.qvel[:self.dof] = qdot_full

        # Inject measured torques (for contact estimation)
        self.mj_data.qfrc_actuator[:7] = tau_meas

        # Update derived quantities
        mujoco.mj_forward(self.mj_model, self.mj_data)

    # ----------------------------------------------------
    # Wrapper to get obs for the policy
    # ----------------------------------------------------
    def get_obs_for_policy(self):
        """
        Call this after update_from_robot():
            obs = env.get_obs_for_policy()

        This ensures build_obs() (inherited) is used unmodified.
        """
        return self.build_obs()

    # ----------------------------------------------------
    # RL step wrapper (no simulation)
    # ----------------------------------------------------
    def step_from_robot(self, a_norm):
        """
        Performs a 'logical' RL step:
        - no mj_step()
        - uses real robot state
        - computes obs, reward, and info

        a_norm = action sent to robot by the RL policy (normalized)
        """

        self.step_count += 1

        # jerk metric same as sim
        a_norm = np.asarray(a_norm, dtype=np.float32)
        jerk = float(np.linalg.norm(a_norm - self._prev_action)**2)
        self._prev_action = a_norm.copy()

        # obs on real state
        obs = self.get_obs_for_policy()

        # reward identical to simulation
        r, info = self.compute_reward(jerk)

        terminated = False
        truncated = self.step_count >= self.max_episode_steps

        info.update({
            "terminated": terminated,
            "truncated": truncated,
            "step_count": self.step_count,
            "mode": "real",
        })

        return obs, r, terminated, truncated, info

    # ----------------------------------------------------
    # Disable sim-only functions
    # ----------------------------------------------------
    def step(self, action):
        raise RuntimeError("Use step_from_robot() in real mode, not step().")

    def action_to_command(self, action):
        raise RuntimeError("Real robot applies torques itself.")

    def reset_state(self, *a, **kw):
        pass    # hardware sets its own state