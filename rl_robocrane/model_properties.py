import mujoco
import numpy as np



import mujoco
import numpy as np


def combine_body_inertias(model, start_body_id, end_body_id):
    """
    Combine inertia parameters of bodies in [start_body_id, end_body_id].

    Returns:
        total_mass
        combined_com (in body/world-aligned coordinates)
        combined_inertia_3x3 (about combined CoM)
    """

    assert 0 <= start_body_id <= end_body_id < model.nbody

    total_mass = 0.0
    weighted_com_sum = np.zeros(3)

    # First pass: compute total mass and global CoM
    for body_id in range(start_body_id, end_body_id + 1):
        m = model.body_mass[body_id]
        com_local = model.body_ipos[body_id]

        total_mass += m
        weighted_com_sum += m * com_local

    if total_mass == 0:
        raise ValueError("Total mass is zero.")

    combined_com = weighted_com_sum / total_mass

    # Second pass: compute inertia about combined CoM
    combined_inertia = np.zeros((3, 3))

    for body_id in range(start_body_id, end_body_id + 1):
        m = model.body_mass[body_id]

        # Diagonal inertia in body inertial frame
        I_diag = model.body_inertia[body_id]
        I_body = np.diag(I_diag)

        com_local = model.body_ipos[body_id]

        # Parallel axis theorem
        r = com_local - combined_com
        r_sq = np.dot(r, r)
        I_parallel = m * (r_sq * np.eye(3) - np.outer(r, r))

        combined_inertia += I_body + I_parallel

    return total_mass, combined_com, combined_inertia





    # Load model
def display_model_properties(model_path):
    model = mujoco.MjModel.from_xml_path(model_path)

    print(f"\nLoaded model: {model_path}")
    print(f"Number of bodies: {model.nbody}\n")

    print("Body mass properties:\n")
    print("-" * 60)

    for body_id in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)

        if name is None:
            name = f"<unnamed_{body_id}>"

        mass = model.body_mass[body_id]
        inertia = model.body_inertia[body_id]  # (Ixx, Iyy, Izz)
        com = model.body_ipos[body_id]         # local CoM position

        print(f"Body ID: {body_id}")
        print(f"  Name      : {name}")
        print(f"  Mass [kg] : {mass:.6f}")
        print(f"  Inertia   : {inertia}")
        print(f"  COM (local): {com}")
        print("-" * 60)


def main():
    model_path = "./robocrane/robocrane_contact_pin.xml"

    display_model_properties(model_path)

    mj_model = mujoco.MjModel.from_xml_path(model_path)
    total_mass, combined_com, combined_inertia = combine_body_inertias(mj_model, start_body_id=12, end_body_id=32)
    print(f"Combined body inertias:")
    print(f"  Total mass: {total_mass}")
    print(f"  Combined CoM: {combined_com}")
    print(f"  Combined inertia: {combined_inertia}")

if __name__ == "__main__":
    main()
