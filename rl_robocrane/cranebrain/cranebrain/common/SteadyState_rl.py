from importlib.resources import files

import casadi as ca
import numpy as np
import pinocchio as pin
from casadi import *
from pinocchio import casadi as cpin
from scipy.spatial.transform import Rotation

from cranebrain.common.load_model import (
    get_gripper_point_frame_id,
    load_mujoco_model,
    load_pinocchio_model,
)
from cranebrain.common.sample_workspace import (
    sample_cylindrical_workspace_segment,
    visualize_workspace,
)


class SteadyState:
    def __init__(self, pin_model, pin_data, tool_frame_id, tol=1e-6, max_iter=100):
        self.tol = tol
        self.max_iter = max_iter
        self.pin_model = pin_model
        self.pin_data = pin_data
        self.cpin_model = cpin.Model(pin_model)
        self.cpin_data = self.cpin_model.createData()
        self.tool_frame_id = tool_frame_id

        # joint bounds
        self.q_ub = np.array([170, 120, 170, 120, 170, 120, 175, 35, 35]) * np.pi / 180
        self.q_lb = -self.q_ub
        self.lbg = np.zeros(2)
        self.ubg = np.zeros(2)

        # Adjusted cost weights - increase orientation weight
        self.cost_weight_parameters = np.array(
            [1, 10, 0.01]
        )  # pos, orientation, q regularization

        # IK solver initialization
        self.ik_solver = self._generate_ipopt_solver()

    def _generate_ipopt_solver(self):
        dof = self.pin_model.nq
        q = SX.sym("q", 1, dof)
        pd = SX.sym("pd", 1, 3)  # desired position
        yaw_d = SX.sym("yaw_d", 1, 1)  # desired yaw
        q_init = SX.sym("q_init", 1, dof)
        cost_weights = SX.sym("cost_weight_parameters", 1, 3)

        # Forward kinematics
        cpin.forwardKinematics(self.cpin_model, self.cpin_data, q.T)
        cpin.updateFramePlacements(self.cpin_model, self.cpin_data)
        oMact = self.cpin_data.oMf[self.tool_frame_id]

        # Position error
        epos = oMact.translation - pd.T
        fpos = cost_weights[0] * dot(epos, epos)

        # Yaw error
        yaw_act = ca.atan2(oMact.rotation[1, 0], oMact.rotation[0, 0])
        fyaw = cost_weights[1] * (yaw_d - yaw_act) ** 2

        # FIXED: Orientation error - Z-axis should point down
        R_act = oMact.rotation
        z_axis_actual = R_act[:, 2]  # Current Z-axis direction
        z_axis_desired = ca.vertcat(0, 0, -1)  # Desired: pointing down

        # Method 1: Minimize the angle between current and desired Z-axis
        # When Z-axis points down, dot product should be -1
        # Error is minimized when dot(z_actual, z_desired) = -1
        dot_product = dot(z_axis_actual, z_axis_desired)
        e_orient = (
            dot_product + 1
        ) ** 2  # This is 0 when dot_product = -1 (pointing down)

        # Alternative Method 2: Direct component constraint (often more robust)
        # Ensure Z-component is negative and X,Y components are small
        # e_orient = (z_axis_actual[2] + 1)**2 + z_axis_actual[0]**2 + z_axis_actual[1]**2

        forient = cost_weights[1] * e_orient

        # Joint regularization
        q_err = q - q_init
        fq = cost_weights[2] * dot(q_err, q_err)

        # Total cost
        f = fpos + forient + fq + fyaw

        # Gravity constraint for passive joints (only last 2 joints)
        g = cpin.computeGeneralizedGravity(self.cpin_model, self.cpin_data, q.T)[7:]

        # Optimization problem
        opt_vars = vertcat(q[:])
        params = horzcat(cost_weights, pd, yaw_d, q_init)

        nlp = {"x": opt_vars, "f": f, "g": g, "p": params}
        opts = {
            "ipopt": {
                "tol": 1e-3,
                "max_iter": 100,
                'linear_solver' : 'ma57',
                "print_level": 0,
                "acceptable_tol": 1e-2,  # Allow slightly less precise solutions
                "mu_strategy": "adaptive",  # Better convergence
            },
            "print_time": 0,
        }
        return nlpsol("F", "ipopt", nlp, opts)

    def inverse_kinematics(self, pos_yaw, q0=None):
        if q0 is None:
            # Better initial guess - tool pointing down configuration
            q0 = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0, 0.0, 0.0])

        pd = np.array(pos_yaw[:3])
        yaw_d = pos_yaw[3] if len(pos_yaw) > 3 else 0.0  # Optional yaw

        params = np.concatenate(
            [self.cost_weight_parameters, pd, [yaw_d], q0.flatten()]
        )

        res = self.ik_solver(
            x0=q0,
            p=params.T,
            lbx=self.q_lb.T,
            ubx=self.q_ub.T,
            lbg=self.lbg.T,
            ubg=self.ubg.T,
        )

        if self.ik_solver.stats()["return_status"] != "Solve_Succeeded":
            return q0, False

        q_res = res["x"].full().flatten()

        # Verify the solution has tool pointing down
        if self._verify_tool_orientation(q_res):
            return q_res, True
        else:
            # print("Warning: Solution found but tool not pointing down properly")
            return q_res, False

    def _verify_tool_orientation(self, q):
        """Verify that the tool Z-axis is pointing down"""
        pin.forwardKinematics(self.pin_model, self.pin_data, q)
        pin.updateFramePlacements(self.pin_model, self.pin_data)

        oMtool = self.pin_data.oMf[self.tool_frame_id]
        z_axis = oMtool.rotation[:, 2]  # Z-axis direction

        # Check if Z-axis is pointing down (negative Z direction)
        # z_axis[2] should be negative and close to -1
        pointing_down = z_axis[2] < -0.7  # Allow some tolerance

        if not pointing_down:
            print(
                f"Tool Z-axis: {z_axis}, not pointing down (z-component: {z_axis[2]})"
            )

        return pointing_down


import time

import mujoco
import mujoco.viewer


def visualize_solution(mj_model, mj_data, sol_list):
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        for iteration, sol in enumerate(sol_list):
            mj_data.qpos[: mj_model.nq] = sol
            mujoco.mj_forward(mj_model, mj_data)
            viewer.sync()

            # Print tool orientation for verification
            tool_site_id = mujoco.mj_name2id(
                mj_model,
                mujoco.mjtObj.mjOBJ_SITE,
                "lab/iiwa/cardan_joint/ur_gripper/gripping_point",
            )
            if tool_site_id >= 0:
                tool_rotation = mj_data.site_xmat[tool_site_id].reshape(3, 3)
                z_axis = tool_rotation[:, 2]
                print(
                    f"Iteration {iteration + 1}: Z-axis = {z_axis}, pointing down: {z_axis[2] < -0.5}"
                )
            else:
                print(f"Iteration {iteration + 1}: qpos = {sol}")

            time.sleep(0.5)

        print("Visualization complete. Press any key to exit.")


def main():
    # Load robot model
    model_path = "../../../robocrane/robocrane_simplified.xml"
    mj_model, mj_data = load_mujoco_model(model_path)
    pin_model, pin_data = load_pinocchio_model(model_path)
    tool_frame_id = get_gripper_point_frame_id(pin_model)

    ss_solver = SteadyState(pin_model, pin_data, tool_frame_id)

    # Generate workspace samples using your workspace sampling function
    print("Generating workspace samples...")
    workspace_samples = sample_cylindrical_workspace_segment(
        r_min=0.5,
        r_max=0.6,
        z_min=0.1,
        z_max=0.4,
        grid_spacing=0.02,  # Denser sampling for better coverage
        theta_min=-np.pi / 8,
        theta_max=np.pi / 8,  # Half-circle (front hemisphere)
    )

    print(f"Generated {len(workspace_samples)} workspace samples")

    # Visualize the sampled workspace before IK solving
    print("Visualizing workspace samples...")
    visualize_workspace(workspace_samples, num_points=2000)

    # Use the workspace samples for IK solving
    num_samples = min(len(workspace_samples), 50)  # Limit for reasonable compute time
    sample_indices = np.random.choice(
        len(workspace_samples), num_samples, replace=False
    )
    selected_samples = workspace_samples[sample_indices]

    successful_samples = []
    successful_configs = []
    pointing_down_count = 0

    print(f"Solving IK for {num_samples} workspace samples with tool pointing down...")

    for i, sample in enumerate(selected_samples):
        if i % 50 == 0:
            print(f"Progress: {i}/{num_samples}")

        x, y, z, yaw = sample
        yaw = np.pi / 2
        q_sol, success = ss_solver.inverse_kinematics([x, y, z, yaw])

        if success:
            successful_samples.append([x, y, z, yaw])
            successful_configs.append(q_sol)

            # Check if tool is actually pointing down
            if ss_solver._verify_tool_orientation(q_sol):
                pointing_down_count += 1

    successful_samples = np.array(successful_samples)
    successful_configs = np.array(successful_configs)

    print(f"Successful IK solutions: {len(successful_samples)} / {num_samples}")
    print(
        f"Solutions with tool pointing down: {pointing_down_count} / {len(successful_samples)}"
    )

    # Calculate success rate
    workspace_coverage = len(successful_samples) / num_samples * 100
    pointing_down_rate = (
        pointing_down_count / len(successful_samples) * 100
        if len(successful_samples) > 0
        else 0
    )

    print(f"Workspace coverage: {workspace_coverage:.1f}%")
    print(f"Tool pointing down rate: {pointing_down_rate:.1f}%")

    # Save results
    np.savez(
        "successful_samples_down.npz",
        samples=successful_samples,
        configs=successful_configs,
        workspace_samples=workspace_samples,  # Save original workspace for reference
        stats={
            "coverage": workspace_coverage,
            "pointing_down_rate": pointing_down_rate,
        },
    )
    print("Results saved to successful_samples_down.npz")

    # Visualize successful samples in workspace
    if len(successful_samples) > 0:
        print("Visualizing successful IK solutions in workspace...")
        visualize_workspace(successful_samples, num_points=len(successful_samples))

        print(
            "Visualizing robot configurations - check that tool Z-axis points down..."
        )
        # Show a selection of successful configurations
        num_to_show = min(10, len(successful_configs))
        indices_to_show = np.linspace(
            0, len(successful_configs) - 1, num_to_show, dtype=int
        )
        configs_to_show = successful_configs[indices_to_show]
        visualize_solution(mj_model, mj_data, configs_to_show)


if __name__ == "__main__":
    main()
