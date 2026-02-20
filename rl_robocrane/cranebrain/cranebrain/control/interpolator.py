import numpy as np

class Interpolator:
    def __init__(self, q_acc_list, t_start, t_end, q_start, Ts=0.001):
        self.t_start = t_start
        self.t_end = t_end
        self.Ts = Ts
        self.q_start = np.array(q_start).reshape(-1, 1)

        self.T_traj = t_end - t_start
        self.q_acc = [acc for acc in q_acc_list]
        self.time_pts = np.linspace(t_start, t_end, len(self.q_acc))

        self.q_ = np.zeros_like(self.q_start)
        self.v_ = np.zeros_like(self.q_start)
        self.a_ = np.zeros_like(self.q_start)

    def interpolate(self, t):
        if t <= self.t_start:
            self.a_ = np.zeros_like(self.q_start)
            self.v_ = np.zeros_like(self.q_start)
            self.q_ = self.q_start
        elif t >= self.t_end:
            self.a_ = np.zeros_like(self.q_start)
            self.v_ = np.zeros_like(self.q_start)
        else:
            idx = np.searchsorted(self.time_pts, t) - 1
            idx = np.clip(idx, 0, len(self.q_acc) - 2)

            t0, t1 = self.time_pts[idx], self.time_pts[idx + 1]
            a0, a1 = self.q_acc[idx], self.q_acc[idx + 1]
            alpha = (t - t0) / (t1 - t0)
            self.a_ = ((1 - alpha) * a0 + alpha * a1).reshape(-1, 1)
            self.v_ = self.v_ + self.Ts * self.a_
            self.q_ = self.q_ + self.Ts * self.v_ + self.Ts**2 * self.a_

        return self.q_, self.v_, self.a_


import numpy as np
import matplotlib.pyplot as plt
from cranebrain.common.SteadyState import p2p_trajectory_random

if __name__ == "__main__":
    # Generate synthetic trajectory
    q_start_ = np.ones(9)
    q, q_vel, q_acc = p2p_trajectory_random(q_start=q_start_, num_pts=10)

    # Set up interpolator
    t_start = 1
    t_end = 3
    Ts = 0.001
    T_traj = t_end - t_start
    q_vel = q_vel / T_traj
    q_acc = q_acc / (T_traj**2)
    
    interp_ = Interpolator(q_acc, t_start, t_end, q_start_, Ts=Ts)

    # Create evaluation time grid
    t_grid = np.arange(t_start, t_end + Ts, Ts)

    q_interp_list = []
    v_interp_list = []
    a_interp_list = []

    # Run interpolation over time
    for t in t_grid:
        q_i, v_i, a_i = interp_.interpolate(t)
        q_interp_list.append(q_i.flatten())
        v_interp_list.append(v_i.flatten())
        a_interp_list.append(a_i.flatten())

    q_interp_arr = np.array(q_interp_list)
    v_interp_arr = np.array(v_interp_list)
    a_interp_arr = np.array(a_interp_list)

    # Plot results (q, v, a for joint 0 as example)
    # scale acc and vel to T_Traj
    
    joint_idx = 0
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axs[0].plot(t_grid, q_interp_arr[:, joint_idx], label='q_interp')
    axs[0].plot(np.linspace(t_start, t_end, len(q)), q[:, joint_idx], 'o--', label='q_true')
    axs[0].set_ylabel("Position")
    axs[0].legend()

    axs[1].plot(t_grid, v_interp_arr[:, joint_idx], label='v_interp')
    axs[1].plot(np.linspace(t_start, t_end, len(q_vel)), q_vel[:, joint_idx], 'o--', label='v_true')
    axs[1].set_ylabel("Velocity")
    axs[1].legend()

    axs[2].plot(t_grid, a_interp_arr[:, joint_idx], label='a_interp')
    axs[2].plot(np.linspace(t_start, t_end, len(q_acc)), 
                np.array(q_acc)[:, joint_idx].flatten(), 'o--', label='a_true')
    axs[2].set_ylabel("Acceleration")
    axs[2].set_xlabel("Time [s]")
    axs[2].legend()

    plt.suptitle(f"Joint {joint_idx} Trajectory Interpolation")
    plt.tight_layout()
    plt.show()