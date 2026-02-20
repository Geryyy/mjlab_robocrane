import numpy as np
import mujoco
from ..common import BSplines as bs

def addTriad(scene, H, radius, length, ngeom, update, alpha=0.5):
    pos = H[0:3, 3]
    R = H[0:3, 0:3]

    pos_x = pos + length * R[:, 0]
    pos_y = pos + length * R[:, 1]
    pos_z = pos + length * R[:, 2]

    if not update:
        ngeom = scene.ngeom + 1
        scene.ngeom += 3

    mujoco.mjv_initGeom(scene.geoms[ngeom-1],
                        mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.array([1.0, 0.0, 0.0, alpha]))
    mujoco.mjv_connector(scene.geoms[ngeom-1],
                         mujoco.mjtGeom.mjGEOM_ARROW, radius,
                         pos, pos_x)

    mujoco.mjv_initGeom(scene.geoms[ngeom],
                        mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.array([0.0, 1.0, 0.0, alpha]))
    mujoco.mjv_connector(scene.geoms[ngeom],
                         mujoco.mjtGeom.mjGEOM_ARROW, radius,
                         pos, pos_y)

    mujoco.mjv_initGeom(scene.geoms[ngeom+1],
                        mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3),
                        np.zeros(3), np.zeros(9), np.array([0.0, 0.0, 1.0, alpha]))
    mujoco.mjv_connector(scene.geoms[ngeom+1],
                         mujoco.mjtGeom.mjGEOM_ARROW, radius,
                         pos, pos_z)

    return ngeom

def visualizePath(scene, H_ls, ngeoms, update):
    for i, H in enumerate(H_ls):
        length = 0.1
        radius = 0.01  
        alpha = 0.1
        ngeoms[i] = addTriad(scene, H, radius, length, ngeoms[i], update, alpha)




from dataclasses import dataclass
import mujoco

@dataclass
class BodyJointInfo:
    body_id: int = -1
    jnt_id: int = -1
    qpos_adr: int = -1
    joint_type: int = -1

def get_free_body_joint_info(body_name: str, model: mujoco.MjModel) -> BodyJointInfo:
    info = BodyJointInfo()

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name.encode())
    if body_id == -1:
        print(f"Error: Body with name '{body_name}' not found.")
        return info

    jnt_id = model.body_jntadr[body_id]
    if jnt_id == -1:
        print(f"Error: Body '{body_name}' has no joint.")
        return info

    jnt_type = model.jnt_type[jnt_id]
    if jnt_type != mujoco.mjtJoint.mjJNT_FREE:
        print(f"Error: Joint of body '{body_name}' is not a free joint.")
        return info

    info.body_id = body_id
    info.jnt_id = jnt_id
    info.qpos_adr = model.jnt_qposadr[jnt_id]
    info.joint_type = jnt_type

    return info


def set_body_free_joint(body_name: str, model: mujoco.MjModel, data: mujoco.MjData, pos: np.ndarray, quat: np.ndarray) -> None:
    body_info = get_free_body_joint_info(body_name, model)

    if body_info.body_id == -1:
        print(f"Error: Body '{body_name}' not found.")
        return

    data.qpos[body_info.qpos_adr:body_info.qpos_adr + 3] = pos
    data.qpos[body_info.qpos_adr + 3:body_info.qpos_adr + 7] = quat

    


def _add_sphere_to_scene(scn, pos, r, rgba):
    if scn.ngeom >= scn.maxgeom: return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([r, r, r], dtype=np.float64),
        np.asarray(pos, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    scn.ngeom += 1

def draw_ts_path_markers(viewer, ctr_pts_ts, k=2, n=180, r=0.006):
    # dense path from TS ctrl points
    t = bs.knot_vector(len(ctr_pts_ts), k)
    thetas = np.linspace(0.0, 1.0, n)
    ts_path = np.array([bs.bspline(th, t, ctr_pts_ts, k) for th in thetas])

    scn = viewer.user_scn
    scn.ngeom = 0
    budget = max(8, scn.maxgeom - 4)
    step = max(1, len(ts_path) // budget)
    for p in ts_path[::step]:
        _add_sphere_to_scene(scn, p[:3], r, (0.10, 0.70, 1.00, 0.85))
    _add_sphere_to_scene(scn, ts_path[0, :3],  r * 1.8, (0.20, 0.95, 0.20, 0.95))
    _add_sphere_to_scene(scn, ts_path[-1, :3], r * 1.8, (0.95, 0.20, 0.20, 0.95))


def print_body_names(mj_model):
    print("Bodies in model:")
    for i in range(mj_model.nbody):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        print(f"[{i}] {name}")


def print_site_names(mj_model):
    print("Sites in model:")
    for i in range(mj_model.nsite):
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_SITE, i)
        print(f"[{i}] {name}")


