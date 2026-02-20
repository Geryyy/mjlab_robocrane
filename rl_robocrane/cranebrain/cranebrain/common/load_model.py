import os

import numpy as np
import pinocchio as pin
from mujoco import MjData, MjModel

from cranebrain.utils.pinutil import print_joint_names

model_path = os.path.abspath("./robocrane/robocrane.xml")


def load_mujoco_model(model_path):
    mj_model = MjModel.from_xml_path(model_path)
    mj_data = MjData(mj_model)
    return mj_model, mj_data


def load_pinocchio_model(model_path):
    pin_model = pin.buildModelFromMJCF(model_path)
    print("pinocchio: loaded model from ", model_path)
    print("pinocchio: build reduced model")
    print("original nq: ", pin_model.nq)

    joints_to_lock = [joint_name for joint_name in pin_model.names[10:]]

    joints_to_lock_ids = []
    for jn in joints_to_lock:
        if pin_model.existJointName(jn):
            joints_to_lock_ids.append(pin_model.getJointId(jn))
        else:
            print("Warning: joint " + str(jn) + " does not belong to the model!")

    initial_joint_config = np.zeros(pin_model.nq)

    pin_model = pin.buildReducedModel(
        pin_model, joints_to_lock_ids, initial_joint_config
    )
    print("reduced model nq: ", pin_model.nq)
    pin_data = pin_model.createData()

    return pin_model, pin_data


def load_pinocchio_iiwa_model(model_path):
    pin_model = pin.buildModelFromMJCF(model_path)

    print("pinocchio: build reduced model")
    print("original nq: ", pin_model.nq)

    joints_to_lock = [joint_name for joint_name in pin_model.names[8:]]

    joints_to_lock_ids = []
    for jn in joints_to_lock:
        if pin_model.existJointName(jn):
            joints_to_lock_ids.append(pin_model.getJointId(jn))
        else:
            print("Warning: joint " + str(jn) + " does not belong to the model!")

    initial_joint_config = np.zeros(pin_model.nq)

    pin_model = pin.buildReducedModel(
        pin_model, joints_to_lock_ids, initial_joint_config
    )
    print("reduced model nq: ", pin_model.nq)
    pin_data = pin_model.createData()

    return pin_model, pin_data


def get_gripper_point_frame_id(pin_model):
    frame_name = "lab/iiwa/cardan_joint/ur_gripper/gripping_point"
    frame_id = pin_model.getFrameId(frame_name, pin.FrameType.BODY)
    print("gripper_point frame id: ", frame_id)
    return frame_id


def get_endeffector_frame_id(pin_model):
    frame_name = "lab/iiwa/lab/iiwa/iiwa_link_7"
    frame_id = pin_model.getFrameId(frame_name, pin.FrameType.BODY)
    print("gripper_point frame id: ", frame_id)
    return frame_id


def get_tool_body_id(pin_model):
    """
    Gets the body ID of the last body in the Pinocchio model.
    This is often assumed to be the tool or end-effector.

    Args:
        pin_model: The Pinocchio Model object.

    Returns:
        The body ID of the last body, or None if the model has no bodies.
    """
    if pin_model.nbodies > 0:
        return pin_model.nbodies - 1
    else:
        print("Warning: Pinocchio model has no bodies.")
        return None


import mujoco


def get_gripper_mj_geom_id(mj_model):
    # TODO: fix this returns body ids!!!
    """
    Return the geom ID of the gripper point in the MuJoCo model.

    Assumes the gripping point is defined as a geom named:
    "lab/iiwa/cardan_joint/ur_gripper/gripping_point"
    """
    geom_name = "lab/iiwa/cardan_joint/ur_gripper/gripping_point"
    try:
        geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, geom_name)
        print("gripper_point geom ID:", geom_id)
        return geom_id
    except ValueError:
        raise RuntimeError(f"[MuJoCo] Geom '{geom_name}' not found in model.")


def get_gripper_base_mj_geom_id(mj_model):
    geom_name = "lab/iiwa/cardan_joint/ur_gripper/base"
    try:
        geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, geom_name)
        print("gripper_base geom ID:", geom_id)
        return geom_id
    except ValueError:
        raise RuntimeError(f"[MuJoCo] Geom '{geom_name}' not found in model.")


def get_gripper_mj_body_id(mj_model):
    """
    Return the body ID of the gripper point in the MuJoCo model.

    Assumes the gripping point is defined as a body named:
    "lab/iiwa/cardan_joint/ur_gripper/gripping_point"
    """
    body_name = "lab/iiwa/cardan_joint/ur_gripper/gripping_point"
    try:
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        print("gripper_point body ID:", body_id)
        return body_id
    except ValueError:
        raise RuntimeError(f"[MuJoCo] Body '{body_name}' not found in model.")


def get_endeffector_mj_body_id(mj_model):
    body_name = "lab/iiwa/cardan_joint/ur_gripper/gripping_point"
    try:
        body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        print("gripper_point body ID:", body_id)
        return body_id
    except ValueError:
        raise RuntimeError(f"[MuJoCo] Body '{body_name}' not found in model.")


if __name__ == "__main__":
    model_path = "./robocrane/robocrane.xml"  # Replace with your model path
    mj_model, mj_data = load_mujoco_model(model_path)
    pin_model, pin_data = load_pinocchio_model(model_path)
    pin_model_iiwa, pin_data_iiwa = load_pinocchio_iiwa_model(model_path)

    print("Mujoco model loaded with nq:", mj_model.nq)
    print("Pinocchio model loaded with nq:", pin_model.nq)
    print("Pinocchio model (only iiwa) loaded with nq:", pin_model_iiwa.nq)

    # print_joint_names(pin_model_iiwa)
