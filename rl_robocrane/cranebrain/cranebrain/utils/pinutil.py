import pinocchio as pin
from pinocchio import casadi as cpin
import numpy as np


''' pinocchio utils '''

def print_frame_names(pin_model):
    for i, frame in enumerate(pin_model.frames):
        print(i, frame.name)


def print_joint_names(pin_model):
    for i, joint in enumerate(pin_model.joints):
        print(i, joint.shortname)
        

def get_frame_id(pin_model, frame_name):
    return pin_model.getFrameId(frame_name)


# Convert RPY to rotation matrix and create SE3 object
def create_se3_from_rpy_and_trans(translation, rpy):
    rotation_matrix = pin.rpy.rpyToMatrix(rpy[::-1])  # Reversing the order to ZYX
    return pin.SE3(rotation_matrix, translation)


''' forward kinematics '''

def get_frameSE3(pin_model, pin_data, q, frame_id):
    # pin.framesForwardKinematics(pin_model, pin_data, q)
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)
    return pin_data.oMf[frame_id].copy()

def get_rel_frameSE3(pin_model, pin_data, q, frame_id1, frame_id2):
    pin.framesForwardKinematics(pin_model, pin_data, q)
    # pin.updateFramePlacements(pin_model, pin_data)
    return (pin_data.oMf[frame_id1].homogeneous.inverse() * pin_data.oMf[frame_id2]).copy()


def get_jointSE3(pin_model, pin_data, q, joint_id):
    pin.forwardKinematics(pin_model, pin_data, q)
    # pin.updateFramePlacements(pin_model, pin_data)
    return pin_data.oMi[joint_id].copy()

def get_joint_to_frameSE3(pin_model, pin_data, q, joint_id, frame_id):
    pin.forwardKinematics(pin_model, pin_data, q)
    pin.framesForwardKinematics(pin_model, pin_data, q)
    # pin.updateFramePlacements(pin_model, pin_data)
    return (pin_data.oMi[joint_id].inverse() * pin_data.oMf[frame_id]).copy()


''' inverse kinematics '''

def get_frame_jacobian(pin_model, pin_data, q, frame_id):
    pin.computeJointJacobians(pin_model, pin_data, q)
    return pin.getFrameJacobian(pin_model, pin_data, frame_id, pin.ReferenceFrame.WORLD).copy()


def get_joint_jacobian(pin_model, pin_data, q, joint_id):
    Ji_local = pin.computeJointJacobian(pin_model, pin_data, q, joint_id)
    return Ji_local.copy()


def inverse_kinematics_clik(pin_model, pin_data, q_init, joint_id, oMdes, max_iter=1000, eps=1e-4):
    q = q_init.copy()
    DT = 1e-1
    damp = 1e-12

    i = 0
    while True:
        pin.forwardKinematics(pin_model, pin_data, q)
        iMd = pin_data.oMi[joint_id].actInv(oMdes)
        err = pin.log(iMd).vector  # in joint frame
        if np.linalg.norm(err) < eps:
            success = True
            break
        if i >= max_iter:
            success = False
            break
        J = pin.computeJointJacobian(pin_model, pin_data, q, joint_id)  # in joint frame
        J = -np.dot(pin.Jlog6(iMd.inverse()), J)
        v = -J.T.dot(np.linalg.solve(J.dot(J.T) + damp * np.eye(6), err))
        q = pin.integrate(pin_model, q, v * DT)
        if not i % 10:
            print("%d: error = %s" % (i, err.T))
        i += 1

    if success:
        print("Convergence achieved! Iterations: ", i)
    else:
        print(
            "\nWarning: the iterative algorithm has not reached convergence to the desired precision"
        )

    return q, success
    


if __name__ == "__main__":
    pin_model = pin.buildSampleModelManipulator()
    pin_data = pin_model.createData()
    q = pin.randomConfiguration(pin_model)
    
    joint_id = pin_model.nq - 1
    frame_id = len(pin_model.frames) - 1

    oMi_desired = get_jointSE3(pin_model, pin_data, q, joint_id)
    print("q: ", q.T)
    print("oMi: ", oMi_desired)

    q_init = pin.randomConfiguration(pin_model)
    oMi_init = get_jointSE3(pin_model, pin_data, q_init, joint_id)
    print("q_init: ", q_init.T)
    print("oMi: ", oMi_init)

    import time
    start_time = time.time()
    for i in range(100):
        q_ik = inverse_kinematics_clik(pin_model, pin_data, q_init, joint_id, oMi_desired)
    print("Time: ", (time.time() - start_time)/100)
    # q_ik = inverse_kinematics_clik(pin_model, pin_data, q_init, joint_id, oMi_desired)
    print("q_ik (joint): ", q_ik[0].T)


    
    

