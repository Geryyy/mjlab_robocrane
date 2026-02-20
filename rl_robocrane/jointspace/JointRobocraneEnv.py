# jointspace_env_fixed.py

import os
import sys
from dataclasses import dataclass, field, fields

import mujoco
import numpy as np
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.insert(0, parent_dir)
from cranebrain.control.controller_pinocchio import (
    CTcontrol,
    K0_joint,
    K1_joint,
    KI_joint,
)

from BaseRobocraneEnv import BaseRobocraneEnv


def z(n):
    """Convenience: zero vector of length n."""
    return np.zeros(n, dtype=np.float32)


def add_noise(x, sigma):
    return x + np.random.normal(0, sigma, size=x.shape).astype(np.float32)


@dataclass
class ObsStruct:
    # Task-space error (3D position + yaw)
    e_pos: np.ndarray = field(default_factory=lambda: z(3))
    e_yaw: float = 0.0

    # gripper pose
    pos: np.ndarray = field(default_factory=lambda: z(3))
    yaw: float = 0.0

    # goal
    goal_pos: np.ndarray = field(default_factory=lambda: z(3))
    goal_yaw: float = 0.0

    # End-effector pose
    # pos_ef: np.ndarray = z(3)
    # yaw_ef: float      = 0.0

    # Previous action (7 DoF actuated joints)
    a_prev: np.ndarray = field(default_factory=lambda: z(7))

    # joint states (2 DoF)
    q_norm: np.ndarray = field(default_factory=lambda: z(9))
    qdot_norm: np.ndarray = field(default_factory=lambda: z(9))
    # qddot_norm: np.ndarray = field(default_factory=lambda: z(9))

    # qdot_hint_norm: np.ndarray = field(default_factory=lambda: z(7))

    # --------------------------------------------------

    def flatten(self) -> np.ndarray:
        """Return a 1D vector, deterministic field order."""
        parts = []
        for f in fields(self):
            v = getattr(self, f.name)
            if np.isscalar(v):
                parts.append(np.array([v], dtype=np.float32))
            else:
                parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
        return np.concatenate(parts)

    @classmethod
    def flat_dim(cls) -> int:
        """Compute flattened dimension from default values."""
        dummy = cls()  # uses all default values
        return dummy.flatten().size


@dataclass
class ObsStructForce:
    # Task-space error (3D position + yaw)
    e_pos: np.ndarray = field(default_factory=lambda: z(3))
    e_yaw: float = 0.0

    # gripper pose
    pos: np.ndarray = field(default_factory=lambda: z(3))
    yaw: float = 0.0

    # goal
    goal_pos: np.ndarray = field(default_factory=lambda: z(3))
    goal_yaw: float = 0.0

    # End-effector pose
    # pos_ef: np.ndarray = z(3)
    # yaw_ef: float      = 0.0

    # Previous action (7 DoF actuated joints)
    a_prev: np.ndarray = field(default_factory=lambda: z(7))

    # joint states (2 DoF)
    q_norm: np.ndarray = field(default_factory=lambda: z(9))
    qdot_norm: np.ndarray = field(default_factory=lambda: z(9))
    # qddot_norm: np.ndarray = field(default_factory=lambda: z(9))

    # qdot_hint_norm: np.ndarray = field(default_factory=lambda: z(7))

    # # Terrain/contact
    # surface_normal: np.ndarray = field(default_factory=lambda: z(3))
    # z_axis: np.ndarray = field(default_factory=lambda: z(3))

    # External forces
    force_ext: np.ndarray = field(default_factory=lambda: z(3))
    wrench_ext: np.ndarray = field(default_factory=lambda: z(3))
    z_axis: np.ndarray = field(default_factory=lambda: z(3))

    # ground_dist: float = 0.0
    # # Normalized magnitudes
    # force_norm: float = 0.0
    # tau_res_norm: float     = 0.0

    # --------------------------------------------------

    def flatten(self) -> np.ndarray:
        """Return a 1D vector, deterministic field order."""
        parts = []
        for f in fields(self):
            v = getattr(self, f.name)
            if np.isscalar(v):
                parts.append(np.array([v], dtype=np.float32))
            else:
                parts.append(np.asarray(v, dtype=np.float32).reshape(-1))
        return np.concatenate(parts)

    @classmethod
    def flat_dim(cls) -> int:
        """Compute flattened dimension from default values."""
        dummy = cls()  # uses all default values
        return dummy.flatten().size


class JointRobocraneEnv(BaseRobocraneEnv):
    metadata = {"render_modes": []}

    def __init__(
        self,
        mj_model_path="./../robocrane/robocrane_contact.xml",
        pin_model_path="./../robocrane/robocrane_contact_pin.xml",
        max_episode_steps=1024,
        control_dt=0.03,
        randomize_body_com=False,
        com_range=0.005,
        randomize_hfield=False,
        N_path_pts=0,
        expert=False,
        use_force=False,
    ):
        super().__init__(
            mj_model_path=mj_model_path,
            pin_model_path=pin_model_path,
            max_episode_steps=max_episode_steps,
            control_dt=control_dt,
            randomize_body_com=randomize_body_com,
            com_range=com_range,
            randomize_hfield=randomize_hfield,
            penalize_limits=True,
            only_positive_rewards=True,
        )
        self.expert = expert
        self.use_force = use_force
        self.force_normalizer = 40.0
        self.wrench_normalizer = 2.0
        self.max_force = 40 / self.force_normalizer
        # Gaussian force shaping
        self.f_target = 25 / self.force_normalizer
        self.f_sigma = 10 / self.force_normalizer

        # Noise
        self.difficulty = 0.0
        self.meas_sigma_q = 0.01
        self.meas_sigma_dq = 0.01
        self.init_bias_sigma_q = 1e-4
        self.init_bias_sigma_dq = 1e-3
        self.bias_drift_sigma_q = 5e-4
        self.bias_drift_sigma_dq = 5e-3
        self.init_bias_sigma_force = 2.5
        self.init_bias_sigma_wrench = 1
        self.bias_drift_sigma_force = 0.1
        self.bias_drift_sigma_wrench = 0.02
        self.meas_sigma_force = 0.2
        self.meas_sigma_wrench = 0.05
        self._real_robot = False

        # ----------------------------------------------
        # Controller
        # ----------------------------------------------
        print("--------")
        print("Initializing controller with parameters:")
        print(f"K0_joint: {K0_joint}")
        print(f"K1_joint: {K1_joint}")
        print(f"KI_joint: {KI_joint}")
        print("--------")
        self.controller = CTcontrol(
            self.iiwa_pin_model, self.sim_dt, K0_joint, K1_joint, KI_joint
        )

        # ----------------------------------------------
        # Scaled joint velocity limit (INSTEAD of denormalization)
        # ----------------------------------------------
        self.qdot_max_user = 1.0 * np.ones(self.dof_act, dtype=np.float32)

        # ----------------------------------------------
        # Action space (normalized → scaled joint velocities)
        # ----------------------------------------------
        self.action_space = spaces.Box(
            low=-1.0 * np.ones(self.dof_act, dtype=np.float32),
            high=1.0 * np.ones(self.dof_act, dtype=np.float32),
            dtype=np.float32,
        )

        if self.use_force:
            obs_dim = ObsStructForce.flat_dim()
        else:
            obs_dim = ObsStruct.flat_dim()
        self.observation_space = spaces.Box(
            -np.inf * np.ones(obs_dim, dtype=np.float32),
            +np.inf * np.ones(obs_dim, dtype=np.float32),
        )

        # Running EMA for smoothing
        self.pos_err_avg = 0.0
        self.yaw_err_avg = 0.0
        self.sway_err_avg = 0.0
        self.act_mag_avg = 0.0
        self.jerk_avg = 0.0

        self.N_path = N_path_pts
        self.i = 0

    # ==========================================================
    # Difficulty for curriculum
    # ==========================================================
    def set_difficulty(self, difficulty: float):
        self.difficulty = difficulty
        print(f"[Curriculum] Difficulty updated to {difficulty}")

    # ==========================================================
    # RESET
    # ==========================================================
    def sub_reset(self):
        self.bias_q = np.random.normal(0, self.init_bias_sigma_q, size=2)
        self.bias_dq = np.random.normal(0, self.init_bias_sigma_dq, size=2)
        # self.bias_force = np.random.normal(0, self.init_bias_sigma_force, size=3)
        # self.bias_wrench = np.random.normal(0, self.init_bias_sigma_wrench, size=3)
        self.q_d = self.mj_data.qpos[: self.dof_act].copy()
        self.qdot_d = np.zeros_like(self.q_d)
        self.qddot_d = np.zeros_like(self.q_d)
        self._prev_qddot_d = np.zeros_like(self.q_d)
        self._prev_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
        self.a_filtered = np.zeros(self.dof_act)
        self.gripper_z_axis = self.get_gripper_z_axis()

        self.p_init = self.get_gripper_position()
        self.yaw_init = float(self.get_gripper_yaw())
        self.i = 0

        self.viscous_coeff = self.init_viscous_coeff * np.random.uniform(0.5, 1.5)  # add some randomness to friction
        self.static_coeff = self.init_static_coeff * np.random.uniform(0.5, 1.5)
        # print("---------- reset ---------")

    # ==========================================================
    # ACTION → TORQUE
    # ==========================================================
    def friction_model(self, qdot):
        if not hasattr(self, "_sticking"):
            self._sticking = np.zeros_like(qdot, dtype=bool)

        v = np.abs(qdot)

        # hysteresis (stick / release)
        self._sticking = np.where(v < 0.01, True, self._sticking)
        self._sticking = np.where(v > 0.02, False, self._sticking)

        # smooth static friction
        static = self.static_coeff * np.tanh(50.0 * qdot)
        viscous = self.viscous_coeff * qdot

        return np.where(self._sticking,
                        -static,
                        -(static + viscous))
            
    

    def action_to_command(self, a_norm):
        # Scaled, safe joint velocities
        if not self.expert:
            a_scaled = a_norm * self.qdot_max_user
        else:
            a_scaled = a_norm

        # Low pass filter the action
        self.a_filtered = a_scaled

        # # qdot_d = self.denormalize(self.a_filtered, self.qvel_min[:7], self.qvel_max[:7])
        # qdot_d = self.denormalize(self.a_filtered, self.qvel_min[:7], self.qvel_max[:7])
        #
        # # Limit rate of change of desired joint velocities
        # qddot_d = (qdot_d - self._prev_qdot_d) / self.sim_dt
        # qddot_d = np.clip(
        #     qddot_d,
        #     -self.max_acceleration_action,
        #     self.max_acceleration_action,
        # )
        # qdot_d = self._prev_qdot_d + self.sim_dt * qddot_d

        qddot_d = self.denormalize(
            self.a_filtered, self.qacc_min[:7], self.qacc_max[:7]
        )
        # qdddot_d = qddot_d - self._prev_qddot_d
        # qddot_d = self._prev_qddot_d + 1 / self.frameskip * qdddot_d

        # Commanded acceleration
        # qddot_d = (qdot_d - self._prev_qdot_d) / self.dt
        self.qddot_d = qddot_d

        self.qdot_d += self.sim_dt * self.qddot_d
        self.qdot_d = np.clip(self.qdot_d, self.qvel_min[:7], self.qvel_max[:7])

        # Integrate desired joint positions
        self.q_d += self.sim_dt * self.qdot_d
        self.q_d = np.clip(self.q_d, self.qpos_min[:7], self.qpos_max[:7])

        tau = self.controller.update(
            self.mj_data.qpos[:7],
            self.mj_data.qvel[:7],
            self.q_d,
            self.qdot_d,
            self.qddot_d,
        )

        tau_friction = self.friction_model(self.mj_data.qvel[:7])
        tau += tau_friction

        self._prev_qddot_d = self.qddot_d

        return np.clip(tau, self.tau_low, self.tau_high)

    # ==========================================================
    # OBSERVATION
    # ==========================================================
    def use_real_robot(self, real=True):
        self._real_robot = real
        print(f"Using {'REAL' if real else 'SIMULATED'} robot for observations.")

    def build_obs(self):
        if self.use_force:
            obs = ObsStructForce()
        else:
            obs = ObsStruct()

        # Joint states
        q = self.mj_data.qpos[: self.dof].copy()
        dq = self.mj_data.qvel[: self.dof].copy()
        # ddq = self.mj_data.qacc[: self.dof]
        

        if not self._real_robot:
            q[:7] = add_noise(q[:7], sigma=0.001)
            dq[:7] = add_noise(dq[:7], sigma=0.01)
            self.bias_q += np.random.normal(0, self.bias_drift_sigma_q, size=2)
            self.bias_dq += np.random.normal(0, self.bias_drift_sigma_dq, size=2)
            q[7:9] = q[7:9] + self.bias_q + np.random.normal(0, self.meas_sigma_q, size=2)
            dq[7:9] = (
                dq[7:9] + self.bias_dq + np.random.normal(0, self.meas_sigma_dq, size=2)
            )

        if self.N_path > 0:
            self.p_ref = self.p_init + (self.goal_pos - self.p_init) * float(
                self.i / self.N_path
            )
            self.yaw_ref = self.yaw_init + (self.goal_yaw - self.yaw_init) * float(
                self.i / self.N_path
            )
            self.i = np.clip(self.i + 1, 0, self.N_path)
        else:
            self.p_ref = self.goal_pos
            self.yaw_ref = self.goal_yaw
        # ---------------------------------------
        # Position, yaw & errors
        # ---------------------------------------
        # pos = self.get_gripper_position()
        # yaw = float(self.get_gripper_yaw())
        # pose, _ = self.get_gripper_pose_fixed(q)
        # compute forward kinematics
        pose = self.get_gripper_pose_yaw(q)
        pos = pose[:3]
        yaw = pose[3]

        obs.goal_pos = self.p_ref
        obs.goal_yaw = float(self.yaw_ref)

        # obs.e_pos = pos - self.p_ref
        # obs.e_yaw = yaw - float(self.yaw_ref)
        obs.e_pos = pos - self.p_ref
        obs.e_yaw = yaw - float(self.yaw_ref)

        # obs.pos_ef = self.get_iiwa_ef_position()
        # obs.yaw_ef = float(self.get_iiwa_ef_yaw())
        obs.pos_ef = pos
        obs.yaw_ef = yaw

        # ----------------------------------------
        # Joint states (normalized)
        # ----------------------------------------
        obs.q_norm = self.normalize(q, self.qpos_min, self.qpos_max)
        obs.qdot_norm = self.normalize(dq, self.qvel_min, self.qvel_max)
        # obs.qddot_norm = self.normalize(ddq, self.qacc_min, self.qacc_max)

        # qdot_hint = self.compute_qdot_hint(obs.e_pos)
        # obs.qdot_hint_norm = self.normalize(
        #     qdot_hint, self.qvel_min[:7], self.qvel_max[:7]
        # )
        # print("obs.qdot_hint_norm: ", obs.qdot_hint_norm)
        # print("goal_pos: ", obs.goal_pos)
        # print("e_pos: ", obs.e_pos)
        # ----------------------------------------
        # Orientation vectors
        # ----------------------------------------
        # obs.surface_normal = self.surface_normal
        # obs.z_axis = self.get_gripper_z_axis()

        # ----------------------------------------
        # Contact forces
        # ----------------------------------------
        if self.use_force:
            if not hasattr(self, "tau_filtered"):
                self.tau_filtered = self.tau
            alpha = 0.9
            self.tau_filtered = (1 - alpha) * self.tau + alpha * self.tau_filtered
            # tau = self.mj_data.qfrc_applied[:7]

            # print(f"tau: {self.tau_filtered}")
            contact = self.contact_metrics(self.tau_filtered)
            if not self._real_robot:
                self.bias_force += np.random.normal(0, self.bias_drift_sigma_force, size=3)
                self.bias_wrench += np.random.normal(
                    0, self.bias_drift_sigma_wrench, size=3
                )
                self.unnoisy_force = contact["force_ext"].copy()
                force = (
                    self.unnoisy_force
                    + self.bias_force
                    + np.random.normal(0, self.meas_sigma_force, size=3)
                )
                wrench = (
                    contact["wrench_ext"].copy()
                    + self.bias_wrench
                    + np.random.normal(0, self.meas_sigma_wrench, size=3)
                )
            else:
                force = contact["force_ext"]
                wrench = contact["wrench_ext"]

            obs.force_ext = force / self.force_normalizer
            obs.wrench_ext = wrench / self.wrench_normalizer
            obs.z_axis = self.get_gripper_z_axis()
            # obs.force_norm = float(contact["force_norm"])
            # obs.tau_res_norm = float(contact["tau_res_norm"])

            obs.e_pos[2] = 0.0
            obs.goal_pos[2] = 0.3

        # ----------------------------------------
        # Previous action
        # ----------------------------------------
        obs.a_prev = self._prev_action

        # distance between gripper and height field
        # if self.use_force:
        #     obs.ground_dist = pos[2] - self.get_terrain_height(
        #         pos[0:2], res=self.heightfield_resolution
        #     )
        #     # print("ground_dist: ", ground_dist)

        self._obs = obs
        return obs.flatten()

    # ==========================================================
    # REWARD
    # ==========================================================
    def joint_limit_penalty(self, q_norm, margin=0.8):
        """
        q_norm: np.ndarray of shape (n_joints,), normalized to [-1, 1]
        """
        q_abs = np.abs(q_norm)
        # how far beyond the safe margin we are
        excess = np.clip(q_abs - margin, 0.0, None)  # 0 inside safe zone
        # normalize so that penalty is 1.0 if we sit exactly at the limit
        scale = 1.0 - margin
        if scale <= 0:
            raise ValueError("margin must be < 1.0")
        penalty_per_joint = (excess / scale) ** 2  # quadratic
        # average over joints (or sum, depending on your taste)
        return penalty_per_joint.mean()

    def compute_reward(self, jerk):
        obs = self._obs

        # ------------------------------------------
        # Errors
        # ------------------------------------------
        if self.use_force:
            pos_err = np.linalg.norm(obs.e_pos[:2])  #  only x y
        else:
            pos_err = np.linalg.norm(obs.e_pos)
        yaw_err = abs(obs.e_yaw)
        sway_err = np.linalg.norm(obs.qdot_norm[7:9])
        act_mag = np.linalg.norm(obs.a_prev)

        # alignment = np.dot(obs.surface_normal, -obs.z_axis)

        # ------------------------------------------
        # Soft redundancy (q3,q5 → 0)
        # ------------------------------------------
        q3_pen = obs.q_norm[2] ** 2
        q5_pen = obs.q_norm[4] ** 2
        q_pas_pen = np.linalg.norm(obs.q_norm[7:9]) ** 2

        # ------------------------------------------
        # Smoothing / EMA
        # ------------------------------------------
        alpha = 1.0
        self.pos_err_avg = (1 - alpha) * self.pos_err_avg + alpha * pos_err
        self.yaw_err_avg = (1 - alpha) * self.yaw_err_avg + alpha * yaw_err
        self.sway_err_avg = (1 - alpha) * self.sway_err_avg + alpha * sway_err
        self.jerk_avg = (1 - alpha) * self.jerk_avg + alpha * jerk
        self.act_mag_avg = (1 - alpha) * self.act_mag_avg + alpha * act_mag

        # ------------------------------------------
        # Reward components — purely exponential
        # ------------------------------------------
        r_pos1 = 1.0 * np.exp(-20 * self.pos_err_avg)
        r_pos2 = 1.0 * np.exp(-10 * self.pos_err_avg)
        r_pos3 = 1.0 * np.exp(-5 * self.pos_err_avg)

        hfield_dist_d = 0.05 * (1 - r_pos1)
        if self.use_force:
            ground_dist = obs.pos_ef[2] - self.get_terrain_height(
                obs.pos_ef[0:2], res=self.heightfield_resolution
            )
            r_dist = 2.0 * np.exp(-10 * abs(ground_dist - hfield_dist_d))
        else:
            r_dist = 0.0
        # r_pos4 = np.exp(-1 * self.pos_err_avg)
        # r_pos5 = np.exp(-0.2 * self.pos_err_avg)

        r_yaw1 = np.exp(-16 * self.yaw_err_avg)
        r_yaw2 = np.exp(-8 * self.yaw_err_avg)
        r_yaw3 = np.exp(-2 * self.yaw_err_avg)

        # r_sway = 0.05 * np.exp(-self.sway_err_avg)
        # r_smooth = 0.03 * np.exp(-2 * self.jerk_avg)
        # r_effort = 0.0 * np.exp(-0.5 * self.act_mag_avg)
        r_sway = -0.5 * self.sway_err_avg**2
        # r_smooth = -0.05 * self.jerk_avg
        r_smooth = -0.2 * np.linalg.norm(self.qdot_d) ** 2
        # r_effort = -0.25 * self.act_mag_avg**2
        r_effort = -0.5 * self.act_mag_avg**2

        # r_force = (
        #     1.0 * np.exp(-((obs.force_norm - f_target) ** 2) / (2 * sigma**2))
        #     if pos_err < 0.02 and self.i >= self.N_path
        #     else 0.0
        # )
        r_force = 0.0
        r_align = 0.0
        if self.use_force:
            # Use unnoisy force
            force_ext = self.unnoisy_force / self.force_normalizer
            # force_ext = obs.force_ext.copy()

            z_axis = obs.z_axis
            force_norm = np.linalg.norm(force_ext)
            pos_tol = 0.05

            if pos_err < pos_tol and self.i >= self.N_path:
                if force_norm > 0.001 and self.mj_data.ncon > 0:
                    cosine_similarity = -np.dot(self.surface_normal, z_axis) / (
                        np.linalg.norm(self.surface_normal) * np.linalg.norm(z_axis)
                    )
                    angle1 = np.exp(-15 * (1 - np.clip(cosine_similarity, 0.0, 1.0)))
                    angle2 = np.exp(-30 * (1 - np.clip(cosine_similarity, 0.0, 1.0)))
                    angle3 = np.exp(-45 * (1 - np.clip(cosine_similarity, 0.0, 1.0)))
                    angle = angle1 + angle2 + angle3

                    r_force = 1 + (
                        3
                        * angle3
                        * np.exp(
                            -10
                            * ((force_norm - self.f_target) ** 2)
                            / (2 * self.f_sigma**2)
                        )
                        + 3
                        * angle2
                        * np.exp(
                            -5
                            * ((force_norm - self.f_target) ** 2)
                            / (2 * self.f_sigma**2)
                        )
                        + 3
                        * angle1
                        * np.exp(
                            -2
                            * ((force_norm - self.f_target) ** 2)
                            / (2 * self.f_sigma**2)
                        )
                    )
                else:
                    angle1 = 0.0
                    angle2 = 0.0
                    angle3 = 0.0
                    angle = 0.0
                    r_force = 0.0
                r_align = 0.0 * angle

            # Add penalty for excessive force
            if force_norm > self.max_force and self.mj_data.ncon > 0:
                force_penalty = 2 * (force_norm - self.max_force) ** 2
                r_force -= force_penalty
                r_align -= force_penalty * 0

        # Redundancy
        r_redundancy = 0.03 * np.exp(-(q3_pen + q5_pen))
        r_pas_jnt = (1 / 3) * (
            np.exp(-10 * q_pas_pen) + np.exp(-5 * q_pas_pen) + np.exp(-1 * q_pas_pen)
        )

        # Success
        if self.use_force and pos_err < pos_tol and self.i >= self.N_path:
            r_success = 10.0 * np.exp(-self.sway_err_avg)
            r_success *= r_force
        elif not self.use_force:
            r_success = (
                20.0 * np.exp(-self.sway_err_avg)
                if pos_err < 0.02 and self.i >= self.N_path
                else 0.0
            )
        else:
            r_success = 0.0

        # ------------------------------------------
        # joint limit penalty
        # ------------------------------------------
        p_joint = 5 * self.joint_limit_penalty(obs.q_norm, margin=0.9)

        # ------------------------------------------
        # Final score
        # ------------------------------------------
        r_total = (
            r_success
            + r_pos1 * r_yaw1
            + r_pos2 * r_yaw2
            + r_pos3 * r_yaw3
            + r_dist
            # + r_yaw1
            # + r_yaw2
            # + r_yaw3
            + r_sway
            + r_smooth
            + r_effort
            + r_force
            + r_redundancy
            + r_pas_jnt
            # + r_align
            + 1
            - p_joint
        )
        r_total = np.max([r_total, 0.01])

        info = {
            "r_total": float(r_total),
            "r_success": float(r_success),
            "r_pos1": float(r_pos1),
            "r_pos2": float(r_pos2),
            "r_pos3": float(r_pos3),
            "r_yaw1": float(r_yaw1),
            "r_yaw2": float(r_yaw2),
            "r_yaw3": float(r_yaw3),
            "r_sway": float(r_sway),
            "r_smooth": float(r_smooth),
            "r_effort": float(r_effort),
            "r_force": float(r_force),
            "r_redundancy": float(r_redundancy),
            "r_pas_jnt": float(r_pas_jnt),
            "r_align": float(r_align),
            "r_dist": float(r_dist),
            # "p_joint": float(p_joint),
        }

        return float(r_total), info

    def compute_qdot_hint(self, e_pos):
        """
        Compute IK-based joint velocity hint (translation only)
        using MuJoCo Jacobian for the gripper site.
        """

        jacp = np.zeros((3, self.mj_model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.mj_model.nv), dtype=np.float64)

        # Compute positional/rotational Jacobian at gripper site
        site_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ef_site_name
        )

        mujoco.mj_jacSite(
            self.mj_model,
            self.mj_data,
            jacp,
            jacr,
            site_id,
        )

        # extract the part corresponding to actuated joints
        Jpos = jacp[:, : self.dof_act]  # shape (3, 7)

        # virtual cartesian velocity
        v_ref = 1.0 * e_pos  # or Kp * e_pos

        # damped least squares pseudo-inverse
        lam = 0.05
        J_pinv = Jpos.T @ np.linalg.inv(Jpos @ Jpos.T + lam * np.eye(3))

        qdot_hint = J_pinv @ v_ref
        return qdot_hint.astype(np.float32)
