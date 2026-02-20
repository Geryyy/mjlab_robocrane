import os
import sys
from scipy.spatial.transform import Rotation as R


import numpy as np 

def homtrans_to_pos_yaw(homtrans):
    """
    Extracts the position and yaw angle from a 4x4 homogeneous transformation matrix.
    
    Parameters:
        homtrans (np.array): 4x4 Homogeneous transformation matrix.
    
    Returns:
        tuple: Position (x, y, z) and yaw angle in radians.
    """
    pos = homtrans[0:3, 3]  # Extract translation (position)
    yaw = np.arctan2(homtrans[1, 0], homtrans[0, 0])  # Extract yaw from rotation matrix
    return np.concatenate([pos, [yaw]])


def pos_yaw_to_homtrans(pos, yaw):
    """
    Creates a 4x4 homogeneous transformation matrix from a position and yaw angle.
    
    Parameters:
        pos (np.array): Position (x, y, z).
        yaw (float): Yaw angle in radians.
    
    Returns:
        np.array: 4x4 Homogeneous transformation matrix.
    """
    homtrans = np.eye(4)
    homtrans[0:3, 3] = pos  # Set translation (position)
    homtrans[0, 0] = np.cos(yaw)  # Set rotation matrix
    homtrans[1, 0] = np.sin(yaw)
    homtrans[0, 1] = -np.sin(yaw)
    homtrans[1, 1] = np.cos(yaw)
    return homtrans



def pos_rpy_to_homtrans(pos, rpy):
    """
    Creates a 4x4 homogeneous transformation matrix from position and RPY angles.
    
    Parameters:
        pos (np.array): Position (x, y, z).
        rpy (np.array): Roll, pitch, yaw angles (in radians).
    
    Returns:
        np.array: 4x4 Homogeneous transformation matrix.
    """
    rot = R.from_euler('xyz', rpy)  # RPY order
    homtrans = np.eye(4)
    homtrans[0:3, 0:3] = rot.as_matrix()
    homtrans[0:3, 3] = pos
    return homtrans


def homtrans_to_pos_rpy(homtrans):
    """
    Extracts position and RPY angles from a 4x4 homogeneous transformation matrix.
    
    Parameters:
        homtrans (np.array): 4x4 Homogeneous transformation matrix.
    
    Returns:
        tuple:
            pos (np.array): Position (x, y, z).
            rpy (np.array): Roll, pitch, yaw angles (in radians).
    """
    pos = homtrans[0:3, 3]
    rot_matrix = homtrans[0:3, 0:3]
    rpy = R.from_matrix(rot_matrix).as_euler('xyz')
    return pos, rpy




import pinocchio as pin

def forward_kinematics(pin_model, pin_data, frame_id, q):
    pin.forwardKinematics(pin_model, pin_data, q.T)
    pin.updateFramePlacements(pin_model, pin_data)
    oMf_tool = pin_data.oMf[frame_id]
    return oMf_tool.homogeneous



def cubic_joint_interpolation(q_start: np.ndarray, q_goal: np.ndarray, N_pts: int) -> np.ndarray:
    """
    Interpolates between q_start and q_goal using cubic polynomial interpolation.

    Args:
        q_start (np.ndarray): Start joint configuration (9D).
        q_goal (np.ndarray): Goal joint configuration (9D).
        N_pts (int): Number of interpolation points.

    Returns:
        np.ndarray: Array of shape (N_pts, 9) with interpolated joint configurations.
    """
    q_start = np.asarray(q_start)
    q_goal = np.asarray(q_goal)

    # Time vector from 0 to 1
    t = np.linspace(0, 1, N_pts).reshape(-1, 1)

    # Compute coefficients for cubic with zero start/end velocity
    a0 = q_start
    a1 = np.zeros_like(q_start)
    a2 = 3 * (q_goal - q_start)
    a3 = -2 * (q_goal - q_start)

    # Position
    q = a0 + a1 * t + a2 * (t ** 2) + a3 * (t ** 3)

    # Velocity: derivative of q(t)
    # q'(t) = a1 + 2 a2 t + 3 a3 t^2
    q_vel = a1 + 2 * a2 * t + 3 * a3 * (t ** 2)

    # Acceleration: derivative of velocity
    # q''(t) = 2 a2 + 6 a3 t
    q_acc = 2 * a2 + 6 * a3 * t

    return q, q_vel, q_acc



import matplotlib.pyplot as plt
import numpy as np

def plot_trajectory(q, q_vel, q_acc):
    """
    Plot joint positions, velocities, and accelerations over time for 9 DOFs.

    Args:
        q (np.ndarray): Joint positions, shape (N_pts, 9)
        q_vel (np.ndarray): Joint velocities, shape (N_pts, 9)
        q_acc (np.ndarray): Joint accelerations, shape (N_pts, 9)
    """
    N_pts = q.shape[0]
    t = np.linspace(0, 1, N_pts)

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].set_title("Joint Positions")
    for i in range(9):
        axes[0].plot(t, q[:, i], label=f'Joint {i+1}')
    axes[0].set_ylabel('Position (rad)')
    axes[0].legend(loc='upper right', ncol=3, fontsize='small')

    axes[1].set_title("Joint Velocities")
    for i in range(9):
        axes[1].plot(t, q_vel[:, i], label=f'Joint {i+1}')
    axes[1].set_ylabel('Velocity (rad/s)')
    axes[1].legend(loc='upper right', ncol=3, fontsize='small')

    axes[2].set_title("Joint Accelerations")
    for i in range(9):
        axes[2].plot(t, q_acc[:, i], label=f'Joint {i+1}')
    axes[2].set_xlabel('Normalized Time')
    axes[2].set_ylabel('Acceleration (rad/s²)')
    axes[2].legend(loc='upper right', ncol=3, fontsize='small')

    plt.tight_layout()    
    plt.show()


