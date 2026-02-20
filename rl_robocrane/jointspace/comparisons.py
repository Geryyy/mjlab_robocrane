"""
JointspaceViewer — uses shared play_common backend.
Telemetry remains environment-specific.
"""

import argparse
import glob
import os
import sys
import time

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import tikzplotlib
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from cranebrain.control.mpc_expert import MPCExpertPolicy

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from trainer.env_factory import make_single_env

DISTURBANCE_TIME = 101
DISTURBANCE_MAG = 1.75

gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])  # Upper subplot is 3x taller

plt.style.use("paper")
plt.rcParams.update(
    {
        "text.usetex": True,
    }
)
plt.rcParams["text.latex.preamble"] = r"\usepackage{siunitx}\usepackage{amsmath}"


def find_model(algo: str, env_name: str):
    algo = algo.lower()
    print("pwd: ", os.getcwd())
    print("env: ", env_name)
    search_paths = [
        "./comparison_models/*.zip",
    ]
    for pattern in search_paths:
        files = sorted(glob.glob(pattern))
        if files:
            latest = files[-1]
            print(f"📂 Using model: {latest}")
            return latest
    raise FileNotFoundError(f"❌ No {algo.upper()} model found.")


def run_viewer(
    base_env,
    env,
    env_name="jointspace",
    send_telemetry_fn=None,
    algo="PPO",
    model_path=None,
    steps=300,
    visu=False,
    keyboard=False,
    show_obs=False,
    test=False,
    expert=False,
    custom_goal=None,
    use_disturbance=False,
):
    """
    Play viewer for any Robocrane environment.
    Telemetry must be implemented per environment.

    Args:
        create_env_fn: function(max_steps) -> Gym env
        algo: "PPO" or "SAC"
        model_path: optional custom model path
        visu: enable goal/normal visualization
        keyboard: manual keyboard control
        show_obs: print observation
    """

    # --------------------------
    # Resolve model path
    # --------------------------
    algo = algo.upper()

    if test:
        model = None
    elif keyboard:
        model = None
        from cranebrain.utils.keyboard_control import KeyboardController

        keyboard_ctrl = KeyboardController()
    elif expert:
        expert_policy = MPCExpertPolicy(
            model_path="./../robocrane/robocrane_contact_pin.xml",
            env=base_env,
            Ts=base_env.dt,
            N_horizon=100,
            regenerate=False,
            N_path_pts=0,
        )
    else:
        if model_path is None:
            model_path = find_model(algo, env_name)

        # --------------------------
        # Load model

        if algo == "PPO":
            model = PPO.load(model_path, env=env, device="cuda:0")
        else:
            model = SAC.load(model_path, env=env, device="cuda:0")

    # --------------------------
    # Viewer setup
    # --------------------------
    print("🎥 Starting MuJoCo viewer...")
    mj_model = base_env.mj_model
    mj_data = base_env.mj_data

    # Logging arrays for observations and actions
    # If we do not reach steps we have a problem, use lists instead
    # and convert to numpy arrays at the end
    log_data = {
        "t": [],
        "obs": [],
        "act": [],
        "tau": [],
        "q": [],
        "dq": [],
        "ddq": [],
        "p": [],
        "p_goal": [],
        "fails": [],
        "t_comp": [],
    }

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        obs = env.reset()
        base_env.set_goal(custom_goal)
        if expert:
            expert_policy.reset(base_env.mj_data.qpos[: base_env.dof], np.zeros(9))
            expert_policy.update_task(base_env.goal_pos, base_env.goal_yaw)
        viewer.update_hfield(0)
        episode_reward = 0.0
        step_counter = 0

        cam_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, "ortho_camera")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id
        viewer.cam.orthographic = True
        viewer.sync()

        # Visualization geoms (goal + arrows)
        if visu:
            goal_pos_geom_id = viewer.user_scn.ngeom
            viewer.user_scn.ngeom += 1

            goal_normal_arrow_geom_id = viewer.user_scn.ngeom
            viewer.user_scn.ngeom += 1

            gripper_arrow_geom_id = viewer.user_scn.ngeom
            viewer.user_scn.ngeom += 1

            # Initialize them with mjv_initGeom
            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[goal_pos_geom_id],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([0.02, 0, 0]),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.array([0, 1, 1, 0.7]),
            )

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[goal_normal_arrow_geom_id],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.007, 0, 0]),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.2, 0.8, 0.2, 0.9]),
            )

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[gripper_arrow_geom_id],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.007, 0, 0]),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.2, 0.2, 0.8, 0.9]),
            )

        while viewer.is_running():
            # --------------------------
            # Select action
            # --------------------------
            if keyboard:
                action = keyboard_ctrl.get_action().reshape(1, -1)
            elif expert:
                u1, info_mpc, _ = expert_policy.predict(obs)
                q_dot = u1
                # action = q_dot[:7] / base_env.qdot_max_user
                action = base_env.normalize(
                    q_dot[:7], base_env.qvel_min[:7], base_env.qvel_max[:7]
                )
                action = action.reshape((1, 7))
            elif model is None:
                act_dim = env.action_space.shape[0]
                action = np.zeros((1, act_dim))
                # action = -1 * base_env._obs.qdot_hint_norm.reshape((1,7))
            else:
                t_comp_start = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                t_comp = time.perf_counter() - t_comp_start

            # --------------------------
            # Step environment
            # --------------------------
            if step_counter == DISTURBANCE_TIME and use_disturbance:
                add_disturbance(base_env, mag=DISTURBANCE_MAG)
            obs, reward, done, info = env.step(action)
            # if done[0]:
            #     pass
            info = info[0]
            episode_reward += reward[0]

            # Log obs and actions
            if not done[0]:
                log_data["t"].append(step_counter * base_env.dt)
                log_data["obs"].append(obs[0])
                log_data["act"].append(action[0])
                log_data["tau"].append(base_env.mj_data.ctrl[: base_env.dof_act].copy())
                log_data["q"].append(base_env.mj_data.qpos[: base_env.dof].copy())
                log_data["dq"].append(base_env.mj_data.qvel[: base_env.dof].copy())
                log_data["ddq"].append(base_env.mj_data.qacc[: base_env.dof].copy())
                log_data["p"].append(base_env.get_gripper_pose_yaw())
                log_data["p_goal"].append(
                    np.concatenate((base_env.goal_pos, [base_env.goal_yaw]))
                )
                if expert and info_mpc["status"] > 0:
                    log_data["fails"].append(True)
                else:
                    log_data["fails"].append(False)
                if expert:
                    log_data["t_comp"].append(expert_policy.controller.get_opt_time())
                else:
                    log_data["t_comp"].append(t_comp)
            step_counter += 1

            # --------------------------
            # Visualization
            # --------------------------
            if visu:
                # Goal point
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[goal_pos_geom_id],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.02, 0, 0]),
                    base_env.goal_pos,
                    np.eye(3).flatten(),
                    np.array([0, 1, 1, 0.6]),
                )

                # Normal arrow
                p1 = base_env.goal_pos
                p2 = p1 + 0.1 * base_env.surface_normal
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[goal_normal_arrow_geom_id],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.01,
                    p1,
                    p2,
                )

                # Gripper Z-axis
                g = base_env.get_gripper_position()
                g2 = g + 0.1 * base_env.get_gripper_z_axis()
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[gripper_arrow_geom_id],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.01,
                    g,
                    g2,
                )

            viewer.sync()
            # time.sleep(0.7 * base_env.dt)

            # --------------------------
            # Episode termination
            # --------------------------
            # Check success
            dist_p = np.linalg.norm(log_data["p"][-1][:3] - base_env.goal_pos)
            dist_yaw = np.linalg.norm(log_data["p"][-1][3] - base_env.goal_yaw)
            if dist_p < 0.03 and dist_yaw < 10 * np.pi / 180:
                success = True
                log_data["steps"] = step_counter
            else:
                success = False
                log_data["steps"] = step_counter
            if done[0]:
                print(f"Success: {success}")
                print(f"Steps taken: {step_counter}")
                print(
                    f"Episode done — reward={episode_reward:.2f} steps={step_counter}"
                )
                log_data["success"] = success
                log_data["p_accuracy"] = dist_p
                log_data["r_accuracy"] = dist_yaw
                # Convert to numpy arrays
                for key in log_data:
                    if key not in ["success", "steps", "p_accuracy", "r_accuracy"]:
                        log_data[key] = np.array(log_data[key])
                return log_data


def plot_log_data(log_data, label="", color=0, linestyle="-", save=False):
    t = log_data["t"]

    # plt.figure(1, figsize=(12, 8))
    #
    # # Plot joint positions
    # plt.subplot(3, 1, 1)
    # for i in range(9):
    #     plt.plot(time, log_data["q"][:, i], label=f"Joint {i + 1} {label}")
    # plt.title("Joint Positions")
    # plt.xlabel("Time Step")
    # plt.ylabel("Position (rad)")
    # plt.legend()
    #
    # # Plot joint velocities
    # plt.subplot(3, 1, 2)
    # for i in range(9):
    #     plt.plot(time, log_data["dq"][:, i], label=f"Joint {i + 1} {label}")
    # plt.title("Joint Velocities")
    # plt.xlabel("Time Step")
    # plt.ylabel("Velocity (rad/s)")
    # plt.legend()
    #
    # # Plot gripper position and goal position
    # plt.subplot(3, 1, 3)
    # for i in range(4):
    #     plt.plot(
    #         time, log_data["p"][:, i], f"C{i}", label=f"Gripper Pose {i + 1} {label}"
    #     )
    #     plt.plot(
    #         time,
    #         log_data["p_goal"][:, i],
    #         f"C{i}--",
    #         label=f"Goal Pose {i + 1} {label}",
    #     )
    # plt.title("Gripper and Goal Positions")
    # plt.xlabel("Time Step")
    # plt.ylabel("Position (m)")
    # plt.legend()

    time_label = r"Time / $\si{\second}$"

    plt.figure(1)
    plt.plot(
        t,
        np.linalg.norm(log_data["dq"][:, 7:9], axis=1),
        f"C{color}",
        label=f"{label}",
        linestyle=linestyle,
    )
    plt.xlabel(time_label)
    plt.ylabel(
        r"$\lVert \dot{\mathbf{q}}_\mathrm{p} \rVert_2$ / $\si{\radian\per\second}$"
    )
    plt.xlim([t[0], t[-1]])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=4)
    plt.vlines(
        x=t[DISTURBANCE_TIME - 1],
        ymin=0.0,
        ymax=5.0,
        color="black",
        linestyle="--",
    )
    plt.ylim([0, 5])
    tikzplotlib.save("plots/passive_joints.tex")

    plt.figure(2)
    plt.subplot(gs[0])
    p_norm = np.linalg.norm(log_data["p"][:, :3] - log_data["p_goal"][:, :3], axis=1)
    plt.plot(t, p_norm, f"C{color}", label=f"{label}", linestyle=linestyle)
    # plt.xlabel(time_label)
    plt.ylabel(
        r"$\lVert \mathbf{p} - \mathbf{p}_{\text{goal}} \rVert_2$ / $\si{\meter}$"
    )
    ymax = 0.25
    plt.vlines(
        x=t[DISTURBANCE_TIME - 1], ymin=0, ymax=ymax, color="black", linestyle="--"
    )
    # mark time where fails occur by shading the background
    plt.ylim([0, ymax])
    plt.xlim([t[0], t[-1]])

    # plt.figure(6)
    plt.subplot(gs[1])
    # Make the height of the lower plot less
    plt.subplots_adjust(hspace=0.3)
    plt.plot(
        t,
        log_data["fails"],
        f"C{color}",
        label=f"{label}",
        linestyle=linestyle,
    )
    plt.xlabel(time_label)
    plt.ylabel("Failures / -")
    plt.xlim([t[0], t[-1]])
    plt.yticks([0, 1], ["False", "True"])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=4)
    plt.vlines(x=t[DISTURBANCE_TIME - 1], ymin=0, ymax=1, color="black", linestyle="--")
    tikzplotlib.save("plots/position_error.tex")

    # p_norm = np.abs(log_data["p"][:, 3] - log_data["p_goal"][:, 3])
    # plt.plot(t, p_norm, f"C{color}--", label=f"Yaw Error {label}")
    # plt.xlabel(time_label)
    # plt.ylabel(r"Yaw error / $\si{\radian}$")
    # plt.legend()

    plt.figure(3)
    act_mag = np.linalg.norm(log_data["act"], axis=1)
    plt.plot(t, act_mag, f"C{color}", label=f"{label}", linestyle=linestyle)
    plt.xlabel(time_label)
    plt.ylabel(r"$\lVert \dot{\mathbf{q}} \rVert_2$ / $\si{\radian\per\second}$")
    plt.xlim([t[0], t[-1]])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=4)
    plt.vlines(
        x=t[DISTURBANCE_TIME - 1],
        ymin=0,
        ymax=1.5,
        color="black",
        linestyle="--",
    )
    plt.ylim([0, 1.5])
    tikzplotlib.save("plots/action_magnitude.tex")

    plt.figure(4)
    tau_mag = np.linalg.norm(log_data["tau"], axis=1)
    plt.plot(t, tau_mag, f"C{color}", label=f"{label}", linestyle=linestyle)
    plt.xlabel(time_label)
    plt.ylabel(r"$\lVert \boldsymbol{\tau}_\mathrm{a} \rVert_2$ / $\si{\newton\meter}$")
    plt.xlim([t[0], t[-1]])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=4)
    ymax = 58
    plt.vlines(
        x=t[DISTURBANCE_TIME - 1],
        ymin=45.5,
        ymax=ymax,
        color="black",
        linestyle="--",
    )
    plt.ylim([45.5, 58])
    tikzplotlib.save("plots/torque_magnitude.tex")

    plt.figure(5)
    plt.plot(
        t, 1000 * log_data["t_comp"], f"C{color}", label=f"{label}", linestyle=linestyle
    )
    plt.xlabel(time_label)
    plt.ylabel(r"$t_{\text{comp}}$ / $\si{\milli\second}$")
    plt.xlim([t[0], t[-1]])
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.55), ncol=4)
    plt.vlines(
        x=t[DISTURBANCE_TIME - 1],
        ymin=0,
        ymax=20,
        color="black",
        linestyle="--",
    )
    plt.ylim([0, 20])
    tikzplotlib.save("plots/t_comp.tex")


def add_disturbance(env, mag=0.1):
    env.mj_data.qvel[7:9] += mag * (2.0 * np.random.normal(2) - 1.0)


# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["PPO", "SAC"], default="PPO")
    parser.add_argument("--model", type=str)
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--visu", action="store_true")
    parser.add_argument("--obs", action="store_true")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    np.random.seed(0)

    # --------------------------
    # Create environment (single)
    # --------------------------
    expert = True
    create_env_fn_mpc = lambda: make_single_env(
        mode="jointspace",
        max_episode_steps=args.steps,
        randomize_hfield=False,
        randomize_body_com=False,
        expert=expert,
    )
    dummy_env = DummyVecEnv([lambda: create_env_fn_mpc()])
    env_mpc = dummy_env
    # env = VecFrameStack(env, n_stack=10)
    base_env_mpc = env_mpc.envs[0]

    expert = False
    create_env_fn_rl = lambda: make_single_env(
        mode="jointspace",
        max_episode_steps=args.steps,
        randomize_hfield=False,
        randomize_body_com=False,
        expert=expert,
    )
    dummy_env = DummyVecEnv([lambda: create_env_fn_rl()])
    env_rl = dummy_env
    # env = VecFrameStack(env, n_stack=10)
    base_env_rl = env_rl.envs[0]

    use_disturbance = False
    goal = np.array([0.42, -0.1, 0.23, -0.417])
    log_data_mpc = run_viewer(
        base_env=base_env_mpc,
        env=env_mpc,
        env_name="jointspace",
        algo=args.algo,
        model_path=args.model,
        steps=args.steps,
        visu=args.visu,
        keyboard=args.keyboard,
        show_obs=args.obs,
        test=args.test,
        expert=True,
        custom_goal=goal,
        use_disturbance=use_disturbance,
    )

    log_data_rl = run_viewer(
        base_env=base_env_rl,
        env=env_rl,
        env_name="jointspace",
        algo=args.algo,
        model_path=args.model,
        steps=args.steps,
        visu=args.visu,
        keyboard=args.keyboard,
        show_obs=args.obs,
        test=args.test,
        expert=False,
        custom_goal=goal,
        use_disturbance=use_disturbance,
    )

    plot_log_data(log_data_mpc, label="MPC", color=0, save=False)
    plot_log_data(log_data_rl, label="RL", color=1, save=True)

    # Save position trajectories as csv
    np.savetxt("plots/p_p2p_mpc.csv", log_data_mpc["p"])
    np.savetxt("plots/p_p2p_rl.csv", log_data_rl["p"])

    use_disturbance = True
    log_data_mpc_dist = run_viewer(
        base_env=base_env_mpc,
        env=env_mpc,
        env_name="jointspace",
        algo=args.algo,
        model_path=args.model,
        steps=args.steps,
        visu=args.visu,
        keyboard=args.keyboard,
        show_obs=args.obs,
        test=args.test,
        expert=True,
        custom_goal=goal,
        use_disturbance=use_disturbance,
    )

    log_data_rl_dist = run_viewer(
        base_env=base_env_rl,
        env=env_rl,
        env_name="jointspace",
        algo=args.algo,
        model_path=args.model,
        steps=args.steps,
        visu=args.visu,
        keyboard=args.keyboard,
        show_obs=args.obs,
        test=args.test,
        expert=False,
        custom_goal=goal,
        use_disturbance=use_disturbance,
    )

    plot_log_data(
        log_data_mpc_dist, label="MPC (dist)", linestyle="--", color=0, save=False
    )
    plot_log_data(
        log_data_rl_dist, label="RL (dist)", linestyle="--", color=1, save=True
    )
    plt.show()

    successes = []
    steps = []
    p_accuracy = []
    r_accuracy = []
    t_comp_mean = []
    t_comp_std = []
    fails = []
    for i in range(10):
        # Sample a goal
        np.random.seed(i)
        goal_pos = base_env_rl.sample_goal_pos()
        goal_yaw = np.random.uniform(-np.pi, np.pi)
        goal = np.concatenate([goal_pos, [goal_yaw]])

        log_data_mpc = run_viewer(
            base_env=base_env_mpc,
            env=env_mpc,
            env_name="jointspace",
            algo=args.algo,
            model_path=args.model,
            steps=args.steps,
            visu=args.visu,
            keyboard=args.keyboard,
            show_obs=args.obs,
            test=args.test,
            expert=True,
            custom_goal=goal,
            use_disturbance=use_disturbance,
        )

        log_data_rl = run_viewer(
            base_env=base_env_rl,
            env=env_rl,
            env_name="jointspace",
            algo=args.algo,
            model_path=args.model,
            steps=args.steps,
            visu=args.visu,
            keyboard=args.keyboard,
            show_obs=args.obs,
            test=args.test,
            expert=False,
            custom_goal=goal,
            use_disturbance=use_disturbance,
        )

        successes.append([log_data_mpc["success"], log_data_rl["success"]])
        steps.append([log_data_mpc["steps"], log_data_rl["steps"]])
        p_accuracy.append([log_data_mpc["p_accuracy"], log_data_rl["p_accuracy"]])
        r_accuracy.append([log_data_mpc["r_accuracy"], log_data_rl["r_accuracy"]])
        t_comp_mean.append(
            [np.mean(log_data_mpc["t_comp"]), np.mean(log_data_rl["t_comp"])]
        )
        t_comp_std.append(
            [np.std(log_data_mpc["t_comp"]), np.std(log_data_rl["t_comp"])]
        )
        fails.append(
            [
                np.sum(log_data_mpc["fails"]) / steps[-1][0],
                np.sum(log_data_rl["fails"]) / steps[-1][1],
            ]
        )
    print("-----------")
    success_mpc = np.sum(np.array(successes)[:, 0]) / len(successes)
    success_rl = np.sum(np.array(successes)[:, 1]) / len(successes)
    print(f"MPC success rate: {success_mpc}")
    print(f"RL success rate: {success_rl}")
    print("-----------")
    print(f"MPC steps: {np.mean(np.array(steps)[:, 0])}")
    print(f"RL steps: {np.mean(np.array(steps)[:, 1])}")
    print("-----------")
    print(f"MPC p accuracy: {np.mean(np.array(p_accuracy)[:, 0])}")
    print(f"RL p accuracy: {np.mean(np.array(p_accuracy)[:, 1])}")
    print("-----------")
    print(f"MPC r accuracy: {np.mean(np.array(r_accuracy)[:, 0])}")
    print(f"RL r accuracy: {np.mean(np.array(r_accuracy)[:, 1])}")
    print("-----------")
    print(f"MPC t comp mean: {np.mean(np.array(t_comp_mean)[:, 0])}")
    print(f"RL t comp mean: {np.mean(np.array(t_comp_mean)[:, 1])}")
    print("-----------")
    print(f"MPC t comp std: {np.mean(np.array(t_comp_std)[:, 0])}")
    print(f"RL t comp std: {np.mean(np.array(t_comp_std)[:, 1])}")
    print("-----------")
    print(f"MPC fails: {np.mean(np.array(fails)[:, 0])}")
    print(f"RL fails: {np.mean(np.array(fails)[:, 1])}")
    # Save to file
    with open(
        f"plots/results{'with_dist' if use_disturbance else 'without_dist'}.txt", "w"
    ) as f:
        f.write(f"MPC success rate: {success_mpc}\n")
        f.write(f"RL success rate: {success_rl}\n")
        f.write("-----------\n")
        f.write(f"MPC steps: {np.mean(np.array(steps)[:, 0])}\n")
        f.write(f"RL steps: {np.mean(np.array(steps)[:, 1])}\n")
        f.write("-----------\n")
        f.write(f"MPC p accuracy: {np.mean(np.array(p_accuracy)[:, 0])}\n")
        f.write(f"RL p accuracy: {np.mean(np.array(p_accuracy)[:, 1])}\n")
        f.write("-----------\n")
        f.write(f"MPC r accuracy: {np.mean(np.array(r_accuracy)[:, 0])}\n")
        f.write(f"RL r accuracy: {np.mean(np.array(r_accuracy)[:, 1])}\n")
        f.write("-----------\n")
        f.write(f"MPC t comp mean: {np.mean(np.array(t_comp_mean)[:, 0])}\n")
        f.write(f"RL t comp mean: {np.mean(np.array(t_comp_mean)[:, 1])}\n")
        f.write("-----------\n")
        f.write(f"MPC t comp std: {np.mean(np.array(t_comp_std)[:, 0])}\n")
        f.write(f"RL t comp std: {np.mean(np.array(t_comp_std)[:, 1])}\n")
        f.write("-----------\n")
        f.write(f"MPC fails: {np.mean(np.array(fails)[:, 0])}\n")
        f.write(f"RL fails: {np.mean(np.array(fails)[:, 1])}\n")


if __name__ == "__main__":
    main()
