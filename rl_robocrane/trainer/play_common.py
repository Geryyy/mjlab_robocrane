"""
play_common.py — Shared MuJoCo viewer backend for all Robocrane environments.

Each environment folder must provide:
    - create_env(max_steps)
    - telemetry sender (inside play.py)
"""

import glob
import os
import time

import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
from cranebrain.control.mpc_expert import MPCExpertPolicy
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


# ============================================================
# Model finder
# ============================================================
def find_latest_model(algo: str, env_name: str):
    algo = algo.lower()
    print("pwd: ", os.getcwd())
    print("env: ", env_name)
    search_paths = [
        # f"./../trainer/trained_models/checkpoints/{algo}/{env_name}/{algo}_step_*.zip",
        f"./../trainer/trained_models/best_{algo}/{env_name}/*.zip",
        # f"./../trainer/trained_models/{algo}_robocrane_final.zip",
    ]
    for pattern in search_paths:
        files = sorted(glob.glob(pattern))
        if files:
            latest = files[-1]
            print(f"📂 Using model: {latest}")
            return latest
    raise FileNotFoundError(f"❌ No {algo.upper()} model found.")


# ============================================================
# Viewer runner (shared)
# ============================================================
def run_viewer(
    create_env_fn,
    env_name="jointspace",
    send_telemetry_fn=None,
    algo="PPO",
    model_path=None,
    steps=1000,
    visu=False,
    keyboard=False,
    show_obs=False,
    test=False,
    expert=False,
    difficulty=0.0,
):
    """
    Play viewer for any Robocrane environment.
    Telemetry must be implemented per environment.

    Args:
        create_env_fn: function(max_steps) -> Gym env
        send_telemetry_fn: optional function(sock, env, obs, action, reward, info)
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

    # --------------------------
    # Create environment (single)
    # --------------------------
    dummy_env = DummyVecEnv([lambda: create_env_fn(steps)])
    env = dummy_env
    # env = VecFrameStack(env, n_stack=5)
    base_env = env.envs[0]
    base_env.set_difficulty(difficulty)

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
            N_horizon=60,
            regenerate=True,
        )
    else:
        if model_path is None:
            model_path = find_latest_model(algo, env_name)

        # --------------------------
        # Load model

        if algo == "PPO":
            # model = PPO.load(model_path, env=env, device="cuda:0")
            # Check if model_path is npz
            if model_path.endswith(".npz"):
                model = RecurrentPPO(
                    "MlpLstmPolicy",
                    policy_kwargs=dict(
                        net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                        activation_fn=nn.Tanh,
                        # log_std_init=cfg["log_std"],
                    ),
                    env=env,
                    verbose=0,
                    device="cuda:0",
                )
                # 2. Load parameters from the .npz file and set them.
                params = np.load(model_path)
                state_dict = {k: torch.tensor(v) for k, v in params.items()}
                model.policy.load_state_dict(state_dict)
            else:
                model = RecurrentPPO.load(model_path, env=env, device="cuda:0")
        else:
            model = SAC.load(model_path, env=env, device="cuda:0")

    # --------------------------
    # Telemetry socket (optional)
    # --------------------------
    sock = None
    if send_telemetry_fn is not None:
        import socket

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print("📡 Telemetry enabled → 127.0.0.1:9870")

    # --------------------------
    # Viewer setup
    # --------------------------
    print("🎥 Starting MuJoCo viewer...")
    mj_model = base_env.mj_model
    mj_data = base_env.mj_data

    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        obs = env.reset()
        if expert:
            expert_policy.reset(base_env.mj_data.qpos[: base_env.dof], np.zeros(9))
            expert_policy.update_task(base_env.goal_pos, base_env.goal_yaw)
        viewer.update_hfield(0)
        episode_reward = 0.0
        step_counter = 0
        frame_time = 1.0 / 30.0

        # Visualization geoms (goal + arrows)
        if visu:
            goal_pos_geom_id = viewer.user_scn.ngeom
            viewer.user_scn.ngeom += 1

            goal_normal_arrow_geom_id = viewer.user_scn.ngeom
            viewer.user_scn.ngeom += 1

            force_arrow_geom_id = viewer.user_scn.ngeom
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
                viewer.user_scn.geoms[force_arrow_geom_id],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.007, 0, 0]),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.5, 0.5, 0.5, 0.9]),
            )

            mujoco.mjv_initGeom(
                viewer.user_scn.geoms[gripper_arrow_geom_id],
                type=mujoco.mjtGeom.mjGEOM_ARROW,
                size=np.array([0.007, 0, 0]),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.array([0.2, 0.2, 0.8, 0.9]),
            )

        lstm_states = None
        while viewer.is_running():
            t_loop_start = time.perf_counter()
            # --------------------------
            # Select action
            # --------------------------
            if keyboard:
                action = keyboard_ctrl.get_action().reshape(1, -1)
            elif expert:
                u1, info, x1 = expert_policy.predict(obs)
                q_dot = x1[9:]
                action = q_dot[:7] / base_env.qdot_max_user
                action = action.reshape((1, 7))
            elif model is None:
                act_dim = env.action_space.shape[0]
                action = np.zeros((1, act_dim))
                # action = -1 * base_env._obs.qdot_hint_norm.reshape((1,7))
            else:
                # action, _ = model.predict(obs, deterministic=True)
                action, lstm_states = model.predict(
                    obs, state=lstm_states, deterministic=True
                )

            # --------------------------
            # Step environment
            # --------------------------
            obs, reward, done, info = env.step(action)
            info = info[0]
            episode_reward += reward[0]
            step_counter += 1

            # Debug obs
            if show_obs:
                base_env.show_obs(obs[0])

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

                # Force arrow
                f1 = base_env.get_gripper_position()
                force_ext = base_env.contact_metrics()["force_ext"]
                f2 = f1 + 0.1 * force_ext
                mujoco.mjv_connector(
                    viewer.user_scn.geoms[force_arrow_geom_id],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.01,
                    f1,
                    f2,
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
            t_loop_end = time.perf_counter()
            t_loop = t_loop_end - t_loop_start
            if t_loop < base_env.dt:
                time.sleep(base_env.dt - t_loop)

            # --------------------------
            # Episode termination
            # --------------------------
            if done[0]:
                print(
                    f"Episode done — reward={episode_reward:.2f} steps={step_counter}"
                )
                obs = env.reset()
                if expert:
                    expert_policy.reset(
                        base_env.mj_data.qpos[: base_env.dof], np.zeros(9)
                    )
                    expert_policy.update_task(base_env.goal_pos, base_env.goal_yaw)
                viewer.update_hfield(0)
                episode_reward = 0.0
                step_counter = 0

            # --------------------------
            # Telemetry
            # --------------------------
            if sock and send_telemetry_fn:
                send_telemetry_fn(sock, env, obs, action, reward, info)
