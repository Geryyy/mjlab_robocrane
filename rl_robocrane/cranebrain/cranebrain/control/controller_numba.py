import numpy as np
import numba
from numba import jit
import time

n_joints = 7  # or whatever size is appropriate

K0_joint = np.diag(np.ones(n_joints) * 550)
K1_joint = np.diag(np.ones(n_joints) * 330)
KI_joint = np.zeros((n_joints, n_joints))  # diagonal is all zeros

class CTcontrol:
    def __init__(self, pin_model_iiwa, Ts, K0, K1, KI):
        self.Ts = Ts
        self.K0 = np.array(K0, dtype=np.float64)
        self.K1 = np.array(K1, dtype=np.float64)
        self.KI = np.array(KI, dtype=np.float64)
        self.integral_error = np.zeros(7, dtype=np.float64)
        self._update_jit = jit(nopython=True)(self._control_law)
    
    @staticmethod
    def _control_law(q, q_vel, q_d, q_vel_d, q_acc_d, integral_error, K0, K1, KI, Ts):
        e_pos = q_d - q
        e_vel = q_vel_d - q_vel
        integral_error = integral_error + e_pos * Ts
        tau = K0 @ e_pos + K1 @ e_vel + KI @ integral_error + q_acc_d
        return tau, integral_error
    
    def update(self, q, q_vel, q_d, q_vel_d, q_acc_d):
        tau, self.integral_error = self._update_jit(
            q.flatten().astype(np.float64), q_vel.flatten().astype(np.float64),
            q_d.flatten().astype(np.float64), q_vel_d.flatten().astype(np.float64), 
            q_acc_d.flatten().astype(np.float64), self.integral_error,
            self.K0, self.K1, self.KI, self.Ts
        )
        return tau

# Speed comparison test
if __name__ == "__main__":
    from controller_pinocchio import CTcontrol as CTcontrol_pinocchio
    from controller_pinocchio import K0_joint, K1_joint, KI_joint
    from cranebrain.common.load_model import load_pinocchio_iiwa_model
    
    # Load model
    pin_model_iiwa, _ = load_pinocchio_iiwa_model("./robocrane/robocrane_simple.xml")
    Ts = 0.001
    
    # Create controllers
    controller_orig = CTcontrol_pinocchio(pin_model_iiwa, Ts, K0_joint, K1_joint, KI_joint)
    controller_numba = CTcontrol(pin_model_iiwa, Ts, K0_joint, K1_joint, KI_joint)
    
    # Test data
    q = np.random.rand(7)
    q_vel = np.random.rand(7)
    q_d = np.random.rand(7)
    q_vel_d = np.random.rand(7)
    q_acc_d = np.random.rand(7)
    
    # Warm up numba
    controller_numba.update(q, q_vel, q_d, q_vel_d, q_acc_d)
    
    # Speed test
    n_iter = 100000
    
    # Original controller
    start = time.time()
    for _ in range(n_iter):
        controller_orig.update(q, q_vel, q_d, q_vel_d, q_acc_d)
    time_orig = time.time() - start
    
    # Numba controller
    start = time.time()
    for _ in range(n_iter):
        controller_numba.update(q, q_vel, q_d, q_vel_d, q_acc_d)
    time_numba = time.time() - start
    
    print(f"Original: {time_orig:.4f}s ({n_iter/time_orig:.0f} Hz)")
    print(f"Numba:    {time_numba:.4f}s ({n_iter/time_numba:.0f} Hz)")
    print(f"Speedup:  {time_orig/time_numba:.1f}x faster")
