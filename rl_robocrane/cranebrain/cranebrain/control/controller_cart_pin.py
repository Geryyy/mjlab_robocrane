import numpy as np
import pinocchio as pin

# Gains
K0_cart = np.diag(np.ones(6) * 550.0)
K1_cart = np.diag(np.ones(6) * 120.0)
K0_N_cart = np.diag(np.ones(7) * 50.0)
K1_N_cart = np.diag(np.ones(7) * 20.0)


class CartesianCTControl:
    """Minimal Cartesian Computed-Torque Operational Space Controller."""

    def __init__(self, model, frame_name, Ts, K0, K1, K0N, K1N, tcp_frame=False, q_NS = np.zeros(7)):
        self.model = model
        self.data = model.createData()
        self.frame_id = model.getFrameId(frame_name, pin.FrameType.BODY)
        self.Ts = Ts
        self.K0, self.K1 = K0, K1
        self.K0N, self.K1N = K0N, K1N
        self.tcp_frame = tcp_frame
        self.q_NS = np.zeros(model.nq)
        self.J = None
        self.Jdot = None

    # ------------------------------------------------------------------
    def update(self, q, q_vel, pose_d, xdot_d, xddot_d):
        self._compute_model(q, q_vel)
        return self._control_law(q, q_vel, pose_d, xdot_d, xddot_d)

    # ------------------------------------------------------------------
    def _compute_model(self, q, q_vel):
        pin.forwardKinematics(self.model, self.data, q, q_vel)
        pin.updateFramePlacements(self.model, self.data)

        self.M = pin.crba(self.model, self.data, q)
        self.nle = pin.nonLinearEffects(self.model, self.data, q, q_vel)

        # Jacobian (LOCAL frame)
        self.J = np.array(
            pin.computeFrameJacobian(self.model, self.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED)
        )

        # Jdot (LOCAL frame) — note: this is J̇, not J̇q̇
        pin.computeJointJacobiansTimeVariation(self.model, self.data, q, q_vel)
        self.Jdot = np.array(
            pin.getFrameJacobianTimeVariation(self.model, self.data, self.frame_id, pin.LOCAL_WORLD_ALIGNED)
        )

        # Operational-space acceleration
        self.xdot = self.J @ q_vel
        self.xddot = self.Jdot @ q_vel     # IMPORTANT: multiply manually

    # ------------------------------------------------------------------
    def _control_law(self, q, q_vel, pose_d, xdot_d, xddot_d):
        # Desired
        p_d = pose_d[:3]
        quat = pose_d[3:7].astype(np.float64) # convert 32 --> 64 bit, expcted by pinocchio
        quat_d = pin.Quaternion(quat[0], quat[1], quat[2], quat[3])
        R_d = quat_d.toRotationMatrix()

        # Current
        oMf = self.data.oMf[self.frame_id]
        p, R = oMf.translation, oMf.rotation

        # Errors
        e_pos = p - p_d
        quat_e = pin.Quaternion(R @ R_d.T)
        e_rot = quat_e.vec()

        e = np.hstack([e_pos, e_rot])
        e_dot = self.xdot - xdot_d
        e_ddot = (xddot_d - self.xddot)

        # Gains (TCP frame transformation optional)
        if self.tcp_frame:
            T = np.zeros((6, 6))
            T[:3, :3] = R_d
            T[3:, 3:] = np.eye(3)
            K0 = T @ self.K0 @ T.T
            K1 = T @ self.K1 @ T.T
        else:
            K0, K1 = self.K0, self.K1

        # Task-space control law
        v_task = e_ddot - K1 @ e_dot - K0 @ e

        # Pseudoinverse (damped recommended for stability)
        lam = 1e-6
        J_pinv = self.J.T @ np.linalg.inv(self.J @ self.J.T + lam * np.eye(6))

        # Nullspace
        N = np.eye(self.model.nq) - J_pinv @ self.J
        v_null = -self.K1N @ q_vel - self.K0N @ (q - self.q_NS)

        # Joint accelerations
        v = J_pinv @ v_task + N @ v_null

        # Torque command
        tau = self.M @ v + self.nle
        return tau

    def get_tool_jacobian(self, q, q_vel=None):
        """
        Returns the 6×n Jacobian of the tool frame in LOCAL_WORLD_ALIGNED coordinates.
        If q_vel is given, it also computes Jdot, but only J is returned.

        This matches exactly the representation used during control.
        """
        if q_vel is None:
            q_vel = np.zeros_like(q)

        # Compute kinematics fresh
        pin.forwardKinematics(self.model, self.data, q, q_vel)
        pin.updateFramePlacements(self.model, self.data)

        J = np.array(
            pin.computeFrameJacobian(
                self.model, self.data, q, self.frame_id, pin.LOCAL_WORLD_ALIGNED
            )
        )
        return J  # shape (6, model.nq)
