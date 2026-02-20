import numpy as np
from .sample_workspace import sample_cylindrical_workspace_segment
from .SteadyState_rl import SteadyState
from .load_model import load_pinocchio_model, get_gripper_point_frame_id
from ..utils.util import pos_rpy_to_homtrans

class IKWorkspaceTaskSampler:
    def __init__(self, model_path, r_min=0.3, r_max=1.0, z_min=0.0, z_max=1.2,
                 grid_spacing=0.2, theta_min=0.0, theta_max=2*np.pi, rng=None):
        self.rng = np.random.default_rng() if rng is None else rng

        self.pin_model, self.pin_data = load_pinocchio_model(model_path)
        self.tool_frame_id = get_gripper_point_frame_id(self.pin_model)
        self.ss_solver = SteadyState(self.pin_model, self.pin_data, self.tool_frame_id)

        # pre-sample workspace once
        self.samples = sample_cylindrical_workspace_segment(
            r_min=r_min,
            r_max=r_max,
            z_min=z_min,
            z_max=z_max,
            grid_spacing=grid_spacing,
            theta_min=theta_min,
            theta_max=theta_max,
        )

    def sample(self, n_ctrl_pts=3, max_tries=50):
        """Pick n_ctrl_pts feasible [task-space, joint-space] ctrl points.
        Uses SteadyState.inverse_kinematics(pos_yaw=[x,y,z,yaw], q0=...).
        Ensures tool Z-axis points down at each ctrl point.
        """
        ctr_pts_ts = []
        ctr_pts_js = []

        tries = 0
        while len(ctr_pts_ts) < n_ctrl_pts and tries < max_tries:
            tries += 1
            x, y, z, yaw = self.samples[self.rng.integers(len(self.samples))]

            # IK initial guess (9 DoF incl. 2 passive)
            q_init = np.array([0.0, 0.0, 0.0, -1.57, 0.0, 1.57, 0.0, 0.0, 0.0], dtype=float)

            # NEW API: pass [x,y,z,yaw], no q_des, no H
            q_res, succ = self.ss_solver.inverse_kinematics([x, y, z, yaw], q0=q_init)

            # Enforce tool-down orientation
            if succ and self.ss_solver._verify_tool_orientation(q_res):
                ctr_pts_ts.append([x, y, z, yaw])
                ctr_pts_js.append(q_res)

        if len(ctr_pts_ts) < n_ctrl_pts:
            raise RuntimeError(f"Could not find {n_ctrl_pts} feasible IK samples after {max_tries} tries")

        return {
            "ctr_pts_ts": np.array(ctr_pts_ts, dtype=float),
            "ctr_pts_js": np.array(ctr_pts_js, dtype=float),
            "theta0": 0.0,
            "theta_dot0": 0.1,
        }
