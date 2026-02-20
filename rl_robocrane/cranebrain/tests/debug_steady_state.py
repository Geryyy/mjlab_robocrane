import mujoco
import mujoco.viewer
import numpy as np
from time import sleep

XML_PATH = "robocrane/robocrane_simplified.xml"

def rpy_to_quat(rpy):
    from scipy.spatial.transform import Rotation
    return Rotation.from_euler("xyz", rpy).as_quat()

def compute_passive_torque(model, data, q):
    data.qpos[:] = q
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)
    return data.qfrc_bias[7:9]

def is_stable(passive_torque, tol=1e-3):
    return np.linalg.norm(passive_torque) < tol

def simple_ik(model, data, target_pos, rpy, q_init, viewer, max_iterations=100):
    q = q_init.copy()
    data.qpos[:] = q
    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,
                                "lab/iiwa/cardan_joint/ur_gripper/gripping_point")

    quat_target = rpy_to_quat(rpy)

    for i in range(max_iterations):
        mujoco.mj_forward(model, data)
        pos_err = target_pos - data.site_xpos[site_id]
        dist = np.linalg.norm(pos_err)

        print(f"[IK Debug] Iter {i+1}/{max_iterations} | Pos Error: {pos_err} | Norm: {dist:.6f}")

        # Draw target marker in viewer
        viewer.user_scn.ngeom = 0
        mujoco.mjv_initGeom(
            viewer.user_scn.geoms[viewer.user_scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.02, 0, 0]),
            pos=target_pos,
            mat=np.eye(3).flatten(),
            rgba=np.array([1, 0, 0, 0.7]),
        )
        viewer.user_scn.ngeom += 1

        viewer.sync()

        if dist < 1e-4:
            print("[IK Debug] Solution reached.")
            return q

        jacp = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacp, None, data.site_xpos[site_id], site_id)

        jacp_act = jacp[:, :7]
        if np.linalg.matrix_rank(jacp_act) >= 3:
            dq = np.linalg.pinv(jacp_act) @ pos_err

            if np.allclose(dq, 0):
                print("[IK Debug] dq is zero → Stuck.")
                return None

            q[:7] += 0.1 * dq
            q[:7] = np.clip(q[:7], model.jnt_range[:7, 0], model.jnt_range[:7, 1])
            data.qpos[:] = q
        else:
            print("[IK Debug] Jacobian rank deficient.")
            return None

        sleep(0.02)

    print("[IK Debug] Max iterations reached. Final error:", dist)
    return None

def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    target_pos = np.array([0.5, 0.0, 0.3])
    rpy = [0.0, 0.0, 0.0]

    guesses = [
        np.array([0, 0, 0, -np.pi/2, 0, np.pi/2, 0]),
        np.array([0, -0.5, 0, -1.2, 0, 1.0, 0.0]),
    ]

    N = 5  # number of iterations

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for iteration in range(N):
            print(f"\n=== Iteration {iteration+1}/{N} ===")
            for guess in guesses:
                for passive in [np.array([0.0, 0.0]), np.array([0.1, 0.0]), np.array([0.0, 0.1])]:
                    q_init = np.zeros(model.nq)
                    q_init[:7] = guess
                    q_init[7:] = passive

                    q_sol = simple_ik(model, data, target_pos, rpy, q_init, viewer)
                    if q_sol is not None:
                        passive_torque = compute_passive_torque(model, data, q_sol)
                        print(f"Passive torque: {passive_torque} | Norm: {np.linalg.norm(passive_torque):.4f}")

                        if is_stable(passive_torque):
                            print("✅ Stable configuration.")
                        else:
                            print("⚠️ Not stable.")
                    else:
                        print("❌ IK failed (stuck).")

    print("Done.")

if __name__ == "__main__":
    main()
