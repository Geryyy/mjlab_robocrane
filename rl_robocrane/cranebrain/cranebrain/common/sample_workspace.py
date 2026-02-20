import numpy as np


def sample_cylindrical_workspace(r_min, r_max, z_min, z_max, grid_spacing):
    """
    Samples a cylindrical workspace with inner and outer radius and height limits.

    Args:
        r_min (float): Inner radius.
        r_max (float): Outer radius.
        z_min (float): Minimum z height.
        z_max (float): Maximum z height.
        grid_spacing (float): Distance between sample points.

    Returns:
        np.ndarray: Array of shape (N, 4) where each row is [x, y, z, yaw]
    """
    x_range = np.arange(-r_max, r_max + grid_spacing, grid_spacing)
    y_range = np.arange(-r_max, r_max + grid_spacing, grid_spacing)
    z_range = np.arange(z_min, z_max + grid_spacing, grid_spacing)
    yaw_range = np.linspace(-np.pi, np.pi, num=int(2 * np.pi / grid_spacing))

    samples = []
    for x in x_range:
        for y in y_range:
            r = np.sqrt(x**2 + y**2)
            if r_min <= r <= r_max:
                for z in z_range:
                    # for yaw in yaw_range:
                    yaw = np.random.uniform(-np.pi, np.pi)
                    samples.append([x, y, z, yaw])
    return np.array(samples)


def sample_cylindrical_workspace_segment(
    r_min, r_max, z_min, z_max, grid_spacing, theta_min=0.0, theta_max=2 * np.pi
):
    """
    Samples a cylindrical *segment* of the workspace with inner and outer radius, height limits,
    and an angular segment defined by theta_min and theta_max (in radians).

    Args:
        r_min (float): Inner radius.
        r_max (float): Outer radius.
        z_min (float): Minimum z height.
        z_max (float): Maximum z height.
        grid_spacing (float): Distance between sample points.
        theta_min (float): Minimum angle (radians) of the segment.
        theta_max (float): Maximum angle (radians) of the segment.

    Returns:
        np.ndarray: Array of shape (N, 4) where each row is [x, y, z, yaw]
    """
    x_range = np.arange(-r_max, r_max + grid_spacing, grid_spacing)
    y_range = np.arange(-r_max, r_max + grid_spacing, grid_spacing)
    z_range = np.arange(z_min, z_max + grid_spacing, grid_spacing)

    samples = []
    for x in x_range:
        for y in y_range:
            r = np.sqrt(x**2 + y**2)
            if r_min <= r <= r_max:
                theta = np.arctan2(y, x)
                # Normalize theta to [0, 2π)
                # if theta < 0:
                #     theta += 2 * np.pi
                # Correctly handle wraparound
                if theta_min <= theta_max:
                    in_segment = theta_min <= theta <= theta_max
                else:
                    in_segment = theta >= theta_min or theta <= theta_max

                if in_segment:
                    for z in z_range:
                        yaw = np.random.uniform(-np.pi, np.pi)
                        samples.append([x, y, z, yaw])
    return np.array(samples)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_workspace(samples, num_points=5000):
    """
    Visualize sampled workspace points in 3D. Color indicates yaw angle.
    X and Y axes are symmetrically centered around 0 to show cylinder properly.

    Args:
        samples (np.ndarray): Array of shape (N, 4), each row is [x, y, z, yaw]
        num_points (int): Max number of points to plot (for performance).
    """
    if samples.shape[0] > num_points:
        indices = np.random.choice(samples.shape[0], num_points, replace=False)
        sampled_points = samples[indices]
    else:
        sampled_points = samples

    x, y, z, yaw = sampled_points.T

    max_radius = max(np.max(np.abs(x)), np.max(np.abs(y)))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(x, y, z, c=yaw, cmap="hsv", s=5)
    fig.colorbar(p, ax=ax, label="Yaw (rad)")

    ax.set_xlim(-max_radius, max_radius)
    ax.set_ylim(-max_radius, max_radius)
    ax.set_zlim(z.min(), z.max())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("KUKA 7-DOF Workspace Sampling (Centered Cylinder View)")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # samples = sample_cylindrical_workspace(
    #     r_min=0.3,
    #     r_max=1.0,
    #     z_min=0.0,
    #     z_max=1.2,
    #     grid_spacing=0.2
    # )

    samples = sample_cylindrical_workspace_segment(
        r_min=0.6,
        r_max=1.0,
        z_min=0.1,
        z_max=0.8,
        grid_spacing=0.1,
        theta_min=0.0,
        theta_max=np.pi/2,
    )
    print(f"nr of samples: {len(samples)}")
    visualize_workspace(samples)
