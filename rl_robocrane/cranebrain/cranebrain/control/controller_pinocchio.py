import numpy as np
import pinocchio as pin

n_joints = 7  # or whatever size is appropriate

# K0_joint = np.diag(np.ones(n_joints) * 500)
# K1_joint = np.diag(np.ones(n_joints) * 100)
K0_joint = np.diag(np.ones(n_joints) * 1200)
K1_joint = np.diag(np.ones(n_joints) * 200)
KI_joint = np.zeros((n_joints, n_joints))  # diagonal is all zeros


class CTcontrol:
    def __init__(self, pin_model, Ts, K0, K1, KI):
        self.dof_ = pin_model.nq
        self.Ts_ = Ts
        self.K0_ = K0
        self.K1_ = K1
        self.KI_ = KI
        self.xq_I_ = np.zeros(self.dof_)
        self.pin_model = pin_model
        self.pin_data = pin_model.createData()

    def update(self, q, q_vel, q_d, q_vel_d, q_acc_d):
        self._compute_model(q, q_vel)
        tau_ = self._control_law(q, q_vel, q_d, q_vel_d, q_acc_d)
        return tau_

    def _reset(self):
        self.xq_I_ = np.zeros(self.dof_)

    def _compute_model(self, q, q_vel):
        # Mass matrix
        self.mass_matrix_ = pin.crba(self.pin_model, self.pin_data, q)

        # Corriolis and gravity
        self.nle_ = pin.nonLinearEffects(self.pin_model, self.pin_data, q, q_vel)

    def _control_law(self, q, q_vel, q_d, q_vel_d, q_acc_d):
        # control error
        e = q - q_d
        e_dot = q_vel - q_vel_d

        # new input v
        v = q_acc_d - self.K0_ @ e - self.K1_ @ e_dot - self.KI_ @ (self.xq_I_)

        # integrator
        self.xq_I_ = self.xq_I_ + self.Ts_ * e

        # computed torque controller
        tau = self.mass_matrix_ @ v + self.nle_

        return tau
