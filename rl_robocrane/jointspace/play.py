"""
JointspaceViewer — uses shared play_common backend.
Telemetry remains environment-specific.
"""

import argparse
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

# Proto for this env
import robocrane_telemetry_pb2

from trainer.env_factory import make_single_env
from trainer.play_common import run_viewer


# ============================================
# Telemetry for Jointspace env
# ============================================
def send_telemetry(sock, env, obs, action, reward, info):
    import time

    base_env = env.envs[0]
    o = base_env._obs  # ObsStruct instance

    msg = robocrane_telemetry_pb2.RobocraneTelemetry()
    msg.timestamp_ms = int(time.time() * 1000)

    # --------------------------------------------------
    # Helper Functions
    # --------------------------------------------------
    def safe_extend(target, field):
        """Extend repeated float protobuf field if attribute exists."""
        if hasattr(o, field):
            val = getattr(o, field)
            if val is None:
                return
            if hasattr(val, "tolist"):
                target.extend(val.tolist())
            elif isinstance(val, (list, tuple)):
                target.extend(val)

    def safe_set(target_name, field):
        """Set scalar protobuf float field if attribute exists."""
        if hasattr(o, field):
            val = getattr(o, field)
            if isinstance(val, (int, float)):
                setattr(msg, target_name, float(val))

    # --------------------------------------------------
    # Raw joint state (MuJoCo)
    # --------------------------------------------------
    q = base_env.mj_data.qpos[: base_env.dof]
    qd = base_env.mj_data.qvel[: base_env.dof]
    qdd = base_env.mj_data.qacc[: base_env.dof]
    msg.q.extend(q.tolist())
    msg.q_dot.extend(qd.tolist())
    msg.q_ddot.extend(qdd.tolist())

    # --------------------------------------------------
    # Normalized joint states (ObsStruct)
    # --------------------------------------------------
    safe_extend(msg.q_norm, "q_norm")
    safe_extend(msg.qdot_norm, "qdot_norm")

    # --------------------------------------------------
    # Desired joint states
    # --------------------------------------------------
    msg.q_d.extend(base_env.q_d.tolist())
    msg.qdot_d.extend(base_env.qdot_d.tolist())
    msg.qddot_d.extend(base_env.qddot_d.tolist())

    # --------------------------------------------------
    # Task-space errors
    # --------------------------------------------------
    safe_extend(msg.e_pos, "e_pos")
    msg.e_yaw = o.e_yaw

    # --------------------------------------------------
    # End-effector pose
    # --------------------------------------------------
    safe_extend(msg.pos_ef, "pos_ef")
    msg.yaw_ef = o.yaw_ef

    # --------------------------------------------------
    # Goal pose
    # --------------------------------------------------
    safe_extend(msg.goal_pos, "goal_pos")
    safe_set("goal_yaw", "goal_yaw")

    # --------------------------------------------------
    # Gripper / object pose (optional)
    # --------------------------------------------------
    safe_extend(msg.pos, "pos")
    safe_set("yaw", "yaw")

    # --------------------------------------------------
    # Previous action (7 DoF)
    # --------------------------------------------------
    safe_extend(msg.a_prev, "a_prev")

    # --------------------------------------------------
    # Terrain / contact
    # --------------------------------------------------
    safe_extend(msg.surface_normal, "surface_normal")
    safe_extend(msg.z_axis, "z_axis")

    # --------------------------------------------------
    # External forces / wrench
    # --------------------------------------------------
    msg.force_ext.extend((base_env._obs.force_ext * base_env.force_normalizer).tolist())
    msg.wrench_ext.extend(
        (base_env._obs.wrench_ext * base_env.wrench_normalizer).tolist()
    )

    # --------------------------------------------------
    # Action (from SB3 → shape (1, act_dim))
    # --------------------------------------------------
    msg.action_raw.extend(action[0].tolist())

    msg.tau_raw.extend(base_env.tau.tolist())

    # --------------------------------------------------
    # Total reward (from argument list)
    # --------------------------------------------------
    msg.reward = float(reward[0])

    # --------------------------------------------------
    # Reward terms from info dict
    # --------------------------------------------------
    for k, v in info.items():
        # Only include numeric values
        if isinstance(v, (int, float)):
            msg.reward_terms[k] = float(v)

    # --------------------------------------------------
    # UDP send
    # --------------------------------------------------
    sock.sendto(msg.SerializeToString(), ("127.0.0.1", 9870))


# ============================================
# Main
# ============================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["PPO", "SAC"], required=True)
    parser.add_argument("--model", type=str)
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--visu", action="store_true")
    parser.add_argument("--obs", action="store_true")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--expert", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--difficulty", type=int, default=1)
    args = parser.parse_args()

    run_viewer(
        create_env_fn=lambda max_steps: make_single_env(
            mode="jointspace",
            max_episode_steps=max_steps,
            randomize_hfield=args.force,
            randomize_body_com=False,
            use_force=args.force,
        ),
        env_name="jointspace",
        send_telemetry_fn=send_telemetry if args.telemetry else None,
        algo=args.algo,
        model_path=args.model,
        steps=args.steps,
        visu=args.visu,
        keyboard=args.keyboard,
        show_obs=args.obs,
        test=args.test,
        expert=args.expert,
        difficulty=args.difficulty,
    )


if __name__ == "__main__":
    main()
