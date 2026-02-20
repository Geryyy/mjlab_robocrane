import gymnasium as gym
import mujoco
import numpy as np
import pinocchio as pin
from cranebrain.common.load_model import (
    get_gripper_mj_body_id,
    get_gripper_point_frame_id,
    load_pinocchio_iiwa_model,
    load_pinocchio_model,
)
from cranebrain.common.SteadyState import SteadyState
from scipy.spatial.transform import Rotation as R


class BaseRobocraneEnv(gym.Env):
    """9-DoF Robocrane Environment (7 actuated + 2 passive)."""

    def __init__(
        self,
        mj_model_path: str = "./../robocrane/robocrane_contact.xml",
        pin_model_path: str = "./../robocrane/robocrane_contact_pin.xml",
        max_episode_steps: int = 1024,
        control_dt: float | None = 0.03,
        randomize_body_com: bool = False,
        com_range: float = 0.005,
        randomize_hfield: bool = False,
        penalize_limits: bool = False,
        only_positive_rewards: bool = False,
    ):
        super().__init__()

        # --- Load model ---
        # mujoco
        self.mj_model = mujoco.MjModel.from_xml_path(mj_model_path)
        self.mj_data = mujoco.MjData(self.mj_model)
        self.tool_body_id = get_gripper_mj_body_id(self.mj_model)

        # pinocchio
        self.iiwa_pin_model, _ = load_pinocchio_iiwa_model(pin_model_path)
        self.iiwa_pin_data = self.iiwa_pin_model.createData()
        self.tool_frame_id = get_gripper_point_frame_id(self.iiwa_pin_model)
        self.ef_site_name = "lab/iiwa/gripper_attachment"
        self.iiwa_link7_name = "lab/iiwa/iiwa_link_7"

        # Steady state
        self.pin_model, _ = load_pinocchio_model(pin_model_path)
        self.pin_data = self.pin_model.createData()
        self.steady_state = SteadyState(
            self.pin_model, self.pin_data, self.tool_frame_id
        )

        # --- Timing ---
        self.sim_dt = float(self.mj_model.opt.timestep)
        desired_dt = self.sim_dt if control_dt is None else float(control_dt)
        self.frameskip = max(1, int(np.round(desired_dt / self.sim_dt)))
        self.dt = self.frameskip * self.sim_dt
        self.max_episode_steps = int(max_episode_steps)

        # --- DoFs ---
        self.dof, self.dof_act = 9, 7
        self.dof_pas = self.dof - self.dof_act
        self.controlled_joints = np.arange(self.dof_act)
        self.tau = np.zeros(self.dof_act)

        # --- Limits ---
        self.qpos_min, self.qpos_max = self.mj_model.jnt_range[:9].T
        max_vel_deg = np.array([85, 85, 100, 75, 130, 135, 135, 50, 50])
        self.qvel_max = np.deg2rad(max_vel_deg[:9])
        self.qvel_min = -self.qvel_max
        act_cr = self.mj_model.actuator_ctrlrange[: self.dof_act]
        self.tau_low, self.tau_high = act_cr[:, 0], act_cr[:, 1]
        # Acceleration limits
        max_acc_deg = np.array([5, 5, 5, 5, 5, 5, 5, 10, 10])
        self.qacc_max = max_acc_deg[: self.dof]
        self.qacc_min = -self.qacc_max

        self.max_acceleration_action = 2.5

        # --- Defaults & workspace ---
        self._default_qpos = np.zeros(self.dof)
        if self.mj_model.nkey > 0:
            mujoco.mj_resetDataKeyframe(self.mj_model, self.mj_data, 0)
            self._default_qpos[:] = self.mj_data.qpos[: self.dof]
        else:
            self._default_qpos[:7] = [0, 0, 0, -np.pi / 2, 0, np.pi / 2, 0]

        # alter CoM of gripper base body
        base_body_name = "lab/iiwa/cardan_joint/ur_gripper/base"  # only this body affects subtree com of gripper
        self.block_body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, base_body_name
        )
        print("block body id:", self.block_body_id)
        self.randomize_body_com_flag = randomize_body_com
        self.com_range = com_range
        self.randomize_hfield = randomize_hfield

        self.terrain_geom_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, "sandkasten"
        )
        self.penalize_limits = penalize_limits
        self.only_positive_rewards = only_positive_rewards

    def enable_randomize_body_com(self, enable: bool):
        self.randomize_body_com_flag = enable

    def enable_randomize_hfield(self, enable: bool):
        self.randomize_hfield = enable

    # ----------------------------------------------------------
    # Normalization helpers
    # ----------------------------------------------------------
    def normalize(self, x, x_min, x_max):
        return 2 * (x - x_min) / (x_max - x_min) - 1

    def denormalize(self, x_norm, x_min, x_max):
        return 0.5 * (x_norm + 1) * (x_max - x_min) + x_min

    # ----------------------------------------------------------
    # Reset / sampling
    # ----------------------------------------------------------
    def sample_goal_pos(self, base_pose=np.zeros(3, dtype=np.float32)):
        """
        Samples a 3D goal position for the end-effector inside the reachable shell
        of the KUKA iiwa R820, while avoiding the floor.
        The workspace is modeled as a spherical shell:
            r_min = 0.42 m
            r_max = 0.82 m
        With the sphere center elevated at z_center = 0.36.
        """

        # ---- workspace model - values from iiwa datasheet ----
        r_min = 0.32
        r_max = 0.82 - 0.2  # leave margin due to ef attachments
        z_center = 0.36
        z_min = 0.12
        z_max = 0.55

        # Sphere center in world coordinates (usually robot base position)
        # If base_pose = [x,y,z] of iiwa base:
        cx, cy, cz = base_pose[0], base_pose[1], z_center + base_pose[2]

        # ---- sample radius uniformly in volume (correct distribution) ----
        r = np.random.uniform(r_min, r_max)

        # ---- sample direction uniformly on sphere ----
        theta = np.random.uniform(-np.pi / 6, np.pi / 6)
        phi = np.pi / 2 + np.random.uniform(-np.pi / 6, np.pi / 6)

        # Convert spherical → Cartesian (unit vector)
        # (u = cos(phi))
        nx = np.sin(phi) * np.cos(theta)
        ny = np.sin(phi) * np.sin(theta)
        nz = np.cos(phi)

        # ---- construct world position ----
        x = cx + r * nx
        y = cy + r * ny
        z = cz + r * nz

        # ---- enforce minimal height ----
        z = max(z, z_min)
        z = min(z, z_max)

        return np.array([x, y, z], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        if self.randomize_body_com_flag:
            self.randomize_body_com(self.block_body_id, xyz_range=self.com_range)

        self.reset_state(self._default_qpos)

        if self.randomize_hfield:
            hfield_name = "terrain"
            H_norm, res = self.sample_heightfield(hfield_name)
            mujoco.mj_forward(self.mj_model, self.mj_data)

            point, (px, py), scale = self.sample_point_on_heightfield(
                H_norm, hfield_name, res
            )

            # Save goal position
            self.goal_pos = point + np.array([0, 0, 0.01])  # 1 cm offset
            # Surface normal
            self.surface_normal = self.compute_surface_normal(H_norm, px, py, scale)
            self.heightfield_array = H_norm
            self.heightfield_resolution = res
        else:
            self.goal_pos = self.sample_goal_pos() + np.array([0, 0, 0.01])
            self.surface_normal = np.array([0, 0, 1], dtype=np.float32)

        self.goal_yaw = -np.pi / 2 + np.random.uniform(-np.pi / 2, np.pi / 2)

        self.sub_reset()

        return self.build_obs(), {}

    def set_goal(self, goal_pose):
        self.goal_pos = goal_pose[:3]
        self.goal_yaw = goal_pose[3]
        self.surface_normal = np.array([0, 0, 1], dtype=np.float32)

    def reset_state(self, q_init):
        self.mj_data.qpos[: self.dof] = q_init
        self.mj_data.qvel[: self.dof] = 0
        self.mj_data.ctrl[:] = 0
        mujoco.mj_forward(self.mj_model, self.mj_data)

        self.pos_d = self.get_gripper_position()
        self.quat_d = self.get_gripper_quaternion()
        self.wrench_filtered = None

    def sub_reset(self):
        pass

    # ----------------------------------------------------------
    # Core logic
    # ----------------------------------------------------------
    def action_to_command(self, action):
        raise NotImplementedError

    def step(self, action):
        # -----------------------------------------
        # 1) Step counter
        # -----------------------------------------
        self.step_count += 1

        # -----------------------------------------
        # 2) Clip & store normalized action
        # -----------------------------------------
        a_norm = np.clip(action, -1, 1).astype(np.float32)

        # -----------------------------------------
        # 3) Jerk metric (common across envs)
        # -----------------------------------------
        if not hasattr(self, "_prev_action"):
            self._prev_action = np.zeros_like(a_norm)

        jerk = float(np.linalg.norm(a_norm - self._prev_action) ** 2)
        # jerk = float(np.linalg.norm(self.mj_data.qacc[: self.dof_act]) ** 2)
        self._prev_action = a_norm

        # -----------------------------------------
        # 4) Frameskip loop
        # -----------------------------------------
        reward_penalty = 0.0
        terminated = False

        for _ in range(self.frameskip):
            # -------------------------------------
            # 4a) Mode-specific control:
            #     Subclass converts action → torque
            # -------------------------------------
            self.tau = self.action_to_command(a_norm)

            # apply torques
            self.mj_data.ctrl[: self.dof_act] = self.tau

            # advance simulation
            mujoco.mj_step(self.mj_model, self.mj_data)

            # -------------------------------------
            # 4c) Contact penalty (shared)
            # -------------------------------------
            penalty = 0.0
            for k in range(self.mj_data.ncon):
                con = self.mj_data.contact[k]
                g1, g2 = con.geom1, con.geom2

                # ignore sandkasten ground contact
                if g1 == self.terrain_geom_id or g2 == self.terrain_geom_id:
                    continue

                penalty = 5.0
                terminated = True
                break
            reward_penalty -= penalty

            # -------------------------------------
            # 4d) Limits penalty
            # -------------------------------------
            if self.penalize_limits:
                if np.any(self.mj_data.qpos[:9] < self.qpos_min[:9]) or np.any(
                    self.mj_data.qpos[:9] > self.qpos_max[:9]
                ):
                    penalty = 5.0
                    terminated = True
                reward_penalty -= penalty

            if self.use_force:
                force_ext = self.contact_metrics()["force_ext"]
                if np.linalg.norm(force_ext) > 100:
                    penalty = 1.0 * self.frameskip
                    terminated = True
                    reward_penalty -= penalty

            if terminated:
                break

        # -----------------------------------------
        # 5) Episode done?
        # -----------------------------------------
        truncated = self.step_count >= self.max_episode_steps

        # -----------------------------------------
        # 6) Build observation (subclass-specific)
        # -----------------------------------------
        obs = self.build_obs()

        obs = self.build_obs()
        r_step, info_terms = self.compute_reward(jerk)
        reward_acc = r_step + reward_penalty
        if self.only_positive_rewards:
            reward_acc = max(reward_acc, 0)

        # -----------------------------------------
        # 7) Append info
        # -----------------------------------------
        info_terms.update(
            {
                "terminated": float(terminated),
                "step_count": self.step_count,
            }
        )

        return obs, reward_acc, terminated, truncated, info_terms

    # ----------------------------------------------------------
    # Observations
    # ----------------------------------------------------------

    def build_obs(self):
        raise NotImplementedError

    def get_iiwa_ef_position(self):
        # site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ef_site_name)
        # pos = self.mj_data.site_xpos[site_id]
        body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.iiwa_link7_name
        )
        pos = self.mj_data.xpos[body_id]
        return np.array([pos[0], pos[1], pos[2]], np.float32)

    def get_iiwa_ef_quat(self):
        # site_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_SITE, self.ef_site_name)
        # rotmat = self.mj_data.site_xmat[site_id].reshape(3,3)
        body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.iiwa_link7_name
        )
        rotmat = self.mj_data.xmat[body_id].reshape(3, 3)
        quat_xyzw = R.from_matrix(rotmat).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=np.float32
        )
        return quat_wxyz

    def get_iiwa_ef_yaw(self):
        body_id = mujoco.mj_name2id(
            self.mj_model, mujoco.mjtObj.mjOBJ_BODY, self.iiwa_link7_name
        )
        yaw = np.arctan2(
            self.mj_data.xmat[body_id, 3],
            self.mj_data.xmat[body_id, 0],
        )
        return np.array([yaw], np.float32)

    def get_gripper_position(self):
        pos = self.mj_data.xpos[self.tool_body_id]
        return np.array([pos[0], pos[1], pos[2]], np.float32)

    def get_gripper_z_axis(self):
        R = self.mj_data.xmat[self.tool_body_id].reshape(3, 3)
        z_axis = R[:, 2]  # third column = body z-axis in world frame
        return z_axis.astype(np.float32)

    def get_gripper_quaternion(self):
        q_mj = self.mj_data.xquat[self.tool_body_id]
        q = np.array([q_mj[3], q_mj[0], q_mj[1], q_mj[2]], dtype=np.float32)
        return q

    def get_gripper_yaw(self):
        yaw = np.arctan2(
            self.mj_data.xmat[self.tool_body_id, 3],
            self.mj_data.xmat[self.tool_body_id, 0],
        )
        return np.array([yaw], np.float32)

    def get_gripper_pose_yaw(self, q):
        # compute forward kinematics with q
        pin.forwardKinematics(self.pin_model, self.pin_data, q, 0 * q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)
        pos = self.mj_data.xpos[self.tool_body_id]
        yaw = np.arctan2(
            self.mj_data.xmat[self.tool_body_id, 3],
            self.mj_data.xmat[self.tool_body_id, 0],
        )
        # quat order: w x y z
        return np.array([pos[0], pos[1], pos[2], yaw], np.float32)

    def get_gripper_pose_fixed(self, q=None):
        """
        Return (x, y, z, yaw) of the gripper in world frame.
        """
        if q is None:
            q = self.mj_data.qpos[: self.dof]
        q_ss = self.steady_state.find_steady_state(q)
        pin.forwardKinematics(
            self.steady_state.pin_model, self.steady_state.pin_data, q_ss, 0 * q_ss
        )
        pin.updateFramePlacements(
            self.steady_state.pin_model, self.steady_state.pin_data
        )
        oMf = self.steady_state.pin_data.oMf[self.tool_frame_id]
        pose, rotation = oMf.translation, oMf.rotation
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
        pose = np.concatenate((pose, [yaw]))
        return pose, rotation

    # ==========================================================
    #  DEPRECATED
    # ==========================================================
    def get_ef_position(self):
        q_iiwa = self.mj_data.qpos[:7]
        pin.forwardKinematics(self.iiwa_pin_model, self.iiwa_pin_data, q_iiwa)
        frame_id = self.controller.frame_id
        pin.updateFramePlacements(self.iiwa_pin_model, self.iiwa_pin_data)
        pos = self.iiwa_pin_data.oMf[frame_id].translation
        return np.array([pos[0], pos[1], pos[2]], np.float32)

    def get_gripper_pose(self):
        pos = self.mj_data.xpos[self.tool_body_id]
        rotmat = self.mj_data.xmat[self.tool_body_id].reshape(3, 3)
        quat_xyzw = R.from_matrix(rotmat).as_quat()  # gives (x,y,z,w)
        quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

        pose = np.concatenate([np.array(pos), quat_wxyz])
        return pose

    # ==========================================================
    #  CONTACT ESTIMATION (REAL-ROBOT COMPATIBLE)
    # ==========================================================

    def compute_model_torque(self, qacc_des=None):
        """
        Compute model-based torque from MuJoCo dynamics:
            tau_model = M*qacc_des + bias
        where:
            bias = C(q,qd) + g(q)      (computed automatically by MuJoCo)

        Default:
            qacc_des = 0  (robot wants to maintain current velocity)

        Returns tau_model[7]
        """
        if qacc_des is None:
            qacc_des = np.zeros(7, dtype=np.float32)

        # Extract full inertia matrix (nq x nq)
        M_full = np.zeros((self.mj_model.nv, self.mj_model.nv), dtype=np.float64)
        mujoco.mj_fullM(self.mj_model, M_full, self.mj_data.qM)

        M = M_full[:7, :7]  # actuated part

        bias = self.mj_data.qfrc_bias[:7].copy()  # C + G

        tau_model = M @ qacc_des + bias
        return tau_model.astype(np.float32)

    def compute_torque_residual(self, torque_actual=None, qacc_des=None):
        """
        Compute torque residual:
            tau_res = tau_actual - tau_model

        Args:
            torque_actual: measured torque (real robot)
                            if None → use MuJoCo actuator forces
            qacc_des: desired joint accelerations (optional)
        """
        if torque_actual is None:
            tau_meas = self.mj_data.qfrc_actuator[:7].copy().astype(np.float32)
        else:
            tau_meas = np.asarray(torque_actual, dtype=np.float32)

        if qacc_des is None:
            qacc_des = self.qddot_d

        tau_model = self.compute_model_torque(qacc_des=qacc_des)
        tau_res = tau_meas - tau_model
        return tau_res.astype(np.float32), float(np.linalg.norm(tau_res))

    def compute_jacobian_world(self):
        """
        Compute 6×7 geometric Jacobian for the tool in world frame:
            [Jv; Jw]
        """
        Jp = np.zeros((3, self.mj_model.nv))
        Jr = np.zeros((3, self.mj_model.nv))

        mujoco.mj_jacBody(self.mj_model, self.mj_data, Jp, Jr, self.tool_body_id)

        J6 = np.vstack([Jp, Jr])[:, :7]
        return J6.astype(np.float32)

    def compute_external_wrench(self, tau_res):
        """
        Estimate external wrench using:
            tau_res = J^T f_ext       →      f_ext = (J^T)^+ tau_res

        Returns 6D wrench: [Fx, Fy, Fz, Mx, My, Mz]
        """
        J = self.compute_jacobian_world()  # (6×7)
        JT = J.T  # (7×6)

        # Damped least-squares inverse of J^T
        lam = 1e-4
        A = JT @ JT.T + lam * np.eye(7)  # (7×7)
        J_pinv_T = np.linalg.solve(A, JT)  # (7×7)^(-1) * (7×6) = (7×6)

        f_ext_6D = J_pinv_T.T @ tau_res  # (6×7) @ (7,) = (6,)
        return f_ext_6D.astype(np.float32)

    def contact_metrics(self, torque_actual=None):
        tau_res, tau_res_norm = self.compute_torque_residual(torque_actual)
        wrench = self.compute_external_wrench(tau_res)
        self.wrench_filtered = wrench

        return {
            "tau_res": tau_res,
            "tau_res_norm": tau_res_norm,
            "force_ext": self.wrench_filtered[:3],
            "wrench_ext": self.wrench_filtered[3:],
            "force_norm": float(np.linalg.norm(wrench[:3])),
        }

    def compute_reward(self, jerk):
        raise NotImplementedError

    def randomize_body_com(self, body_id: int, xyz_range: float = 0.02):
        original_pos = self.mj_model.body_ipos[body_id].copy()
        delta = np.random.uniform(-xyz_range, xyz_range, size=3)
        self.mj_model.body_ipos[body_id] = original_pos + delta
        return delta

    def sample_heightfield(self, hfield_name: str = "terrain"):
        mj_model = self.mj_model

        hf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)
        if hf_id < 0:
            raise ValueError(f"Heightfield '{hfield_name}' not found in model.")

        # --- infer resolution from MJCF ---
        nrow = mj_model.hfield_nrow[hf_id]
        ncol = mj_model.hfield_ncol[hf_id]

        if nrow != ncol:
            raise ValueError("Heightfield must be square for this sampler.")

        res = nrow  # final resolution

        # --- generate your heightfield ---
        from generate_height_field import generate_smooth_hills_data

        H = generate_smooth_hills_data(
            res=res,
            num_hills=7,
            hill_height=0.40,
            hill_sigma=0.2,
            global_tilt=0.20,
            small_noise=0.005,
            only_tilt=True,
        ).astype(np.float32)

        hmin, hmax = float(H.min()), float(H.max())
        H_norm = (H - hmin) / max((hmax - hmin), 1e-6)

        # --- write into MuJoCo buffer ---
        start = mj_model.hfield_adr[hf_id]
        mj_model.hfield_data[start : start + res * res] = H_norm.flatten()

        return H_norm, res

    def get_terrain_height(self, point_xy, hfield_name="terrain", res=32):
        """
        Returns the heightfield data value at given world (x,y) position.
        """
        mj_model, mj_data = self.mj_model, self.mj_data

        hf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)

        # Hfield geom pos
        hx, hy, hz = mj_data.geom("sandkasten").xpos
        sx, sy, sz, z0 = mj_model.hfield_size[hf_id]

        # World <-> pixel scale
        dx = (2 * sx) / res
        dy = (2 * sy) / res

        # Convert world → pixel
        # print("point_xy:", point_xy)
        px = int(np.clip((point_xy[0] - (hx - sx)) / dx, 0, res - 1))
        py = int(np.clip((point_xy[1] - (hy - sy)) / dy, 0, res - 1))

        h = float(self.heightfield_array[py, px])
        z_world = hz + z0 + h * sz
        return z_world

    def sample_point_on_heightfield(
        self,
        H_norm,
        hfield_name="terrain",
        res=64,
        r_min=0.32,
        r_max=0.7,
        theta_min=-np.pi / 4,
        theta_max=np.pi / 4,
    ):
        """
        Samples a point in an annulus around the heightfield center.
        Returns:
        point_world: np.array([x, y, z])
        (px, py): pixel indices
        scale: dict with dx, dy, sz
        """
        mj_model, mj_data = self.mj_model, self.mj_data

        hf_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_HFIELD, hfield_name)

        # Hfield geom pos
        hx, hy, hz = mj_data.geom("sandkasten").xpos
        sx, sy, sz, z0 = mj_model.hfield_size[hf_id]

        # World <-> pixel scale
        dx = (2 * sx) / res
        dy = (2 * sy) / res

        # ---- Sample (x,y) in an annulus ----
        radius = np.sqrt(np.random.uniform(r_min**2, r_max**2))
        theta = np.random.uniform(theta_min, theta_max)

        x_world = hx + radius * np.cos(theta)
        y_world = hy + radius * np.sin(theta)

        # Convert world → pixel
        px = int(np.clip((x_world - (hx - sx)) / dx, 0, res - 1))
        py = int(np.clip((y_world - (hy - sy)) / dy, 0, res - 1))

        # Height
        h = float(H_norm[py, px])
        z_world = hz + z0 + h * sz

        point_world = np.array([x_world, y_world, z_world], dtype=np.float32)

        scale = dict(dx=dx, dy=dy, sz=sz)

        return point_world, (px, py), scale

    def compute_surface_normal(self, H_norm, px, py, scale):
        """
        Computes the surface normal using central differences.
        Uses scale (dx, dy, sz) from sampling function.
        Returns: np.array([nx, ny, nz])
        """
        dx, dy, sz = scale["dx"], scale["dy"], scale["sz"]
        res = H_norm.shape[0]

        i, j = py, px

        im1, ip1 = max(i - 1, 0), min(i + 1, res - 1)
        jm1, jp1 = max(j - 1, 0), min(j + 1, res - 1)

        dzdx = sz * (H_norm[i, jp1] - H_norm[i, jm1]) / (2 * dx)
        dzdy = sz * (H_norm[ip1, j] - H_norm[im1, j]) / (2 * dy)

        n = np.array([-dzdx, -dzdy, 1.0], dtype=np.float32)
        n /= np.linalg.norm(n)
        return n


# ------------------------------------------------------------------
# Smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    env = BaseRobocraneEnv(
        mj_model_path="./robocrane/robocrane_simplified.xml",
        pin_model_path="./robocrane/robocrane_simplified.xml",
        max_episode_steps=256,
    )
