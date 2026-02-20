import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


# --------------------------------------------------------------
# 1) HEIGHTFIELD DATA GENERATION
# --------------------------------------------------------------
def generate_smooth_hills_data(
    res=64,
    num_hills=5,
    hill_height=0.05,
    hill_sigma=0.2,
    global_tilt=0.1,
    small_noise=0.0002,
    only_tilt=False,
):
    """
    Generate a smooth 'chain of hills' heightfield (raw float array).
    No visualization, no saving.
    Returns:
        H: np.ndarray of shape (res, res) with float height values.
    """

    # Grid
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    # --- base: hills ---
    H = np.zeros((res, res))
    centers_x = np.linspace(-0.8, 0.8, num_hills)
    centers_y = np.random.uniform(-0.4, 0.4, size=num_hills)

    if not only_tilt:
        for cx, cy in zip(centers_x, centers_y):
            r2 = (xx - cx) ** 2 + (yy - cy) ** 2
            H += hill_height * np.exp(-r2 / (2 * hill_sigma**2))

    # --- global tilt ---
    if only_tilt:
        global_tilt *= 10
    roll = np.random.uniform(-global_tilt, global_tilt)
    pitch = np.random.uniform(-global_tilt, global_tilt)
    H += np.tan(roll) * xx + np.tan(pitch) * yy

    # --- low noise ---
    if not only_tilt:
        H += small_noise * np.random.randn(res, res)

    return H


# --------------------------------------------------------------
# 2) SAVE HEIGHTFIELD AS PNG
# --------------------------------------------------------------
def save_heightfield_png(H, filename="terrain.png"):
    """
    Normalize heightfield H to [0,255] and save as PNG.
    """

    H = np.asarray(H, dtype=np.float32)
    hmin, hmax = float(H.min()), float(H.max())
    eps = 1e-6
    H_norm = (H - hmin) / max(hmax - hmin, eps)

    img = (H_norm * 255).astype(np.uint8)
    Image.fromarray(img).save(filename)
    print(f"Saved PNG: {filename}")


# --------------------------------------------------------------
# 3) VISUALIZE HEIGHTFIELD IN 3D
# --------------------------------------------------------------
def visualize_heightfield(H):
    """
    Display a 3D surface plot of the heightfield.
    """

    res = H.shape[0]
    x = np.linspace(-1, 1, res)
    y = np.linspace(-1, 1, res)
    xx, yy = np.meshgrid(x, y)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, H, cmap="terrain", linewidth=0)
    ax.set_title("Smooth Chain of Hills (Tilt + Low Noise)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Height")
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------
# MAIN (debug/test)
# --------------------------------------------------------------
if __name__ == "__main__":
    H = generate_smooth_hills_data()
    save_heightfield_png(H, "terrain_smooth_hills.png")
    visualize_heightfield(H)
