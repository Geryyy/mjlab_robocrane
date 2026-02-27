"""Play/evaluate a trained RoboCrane policy."""

from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "mjlab" / "src"))
sys.path.insert(0, str(_ROOT / "rsl_rl"))
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_ROOT / "rl_robocrane" / "jointspace"))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from mjlab_robocrane.task import TASK_ID

try:
    import robocrane_telemetry_pb2
except Exception:
    robocrane_telemetry_pb2 = None


def _resolve_policy_from_checkpoint(checkpoint_file: Path) -> str:
    loaded = torch.load(str(checkpoint_file), map_location="cpu", weights_only=False)
    actor_state = loaded.get("actor_state_dict", {})
    return "lstm" if any(k.startswith("rnn.") for k in actor_state) else "mlp"


class TelemetryStreamer:
    def __init__(self, host: str, port: int, every_n_steps: int = 1):
        if robocrane_telemetry_pb2 is None:
            raise RuntimeError(
                "Telemetry requested but robocrane_telemetry_pb2 is unavailable. "
                "Ensure protobuf runtime is installed and rl_robocrane/jointspace is present."
            )
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._addr = (host, int(port))
        self._every_n_steps = max(1, int(every_n_steps))
        self._step = 0

    @staticmethod
    def _to_list(x) -> list[float]:
        if x is None:
            return []
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().tolist()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (list, tuple)):
            return list(x)
        return [float(x)]

    def send_step(
        self,
        env,
        action: torch.Tensor,
        reward: torch.Tensor,
    ) -> None:
        self._step += 1
        if self._step % self._every_n_steps != 0:
            return

        msg = robocrane_telemetry_pb2.RobocraneTelemetry()
        msg.timestamp_ms = int(time.time() * 1000)

        base_env = env.unwrapped
        robot = base_env.scene["robot"]
        cmd = base_env.command_manager.get_term("goal_pose")
        ctc = base_env.action_manager.get_term("joint_acc_ctc")

        q = robot.data.joint_pos[0]
        qdot = robot.data.joint_vel[0]
        qddot = robot.data.joint_acc[0]
        msg.q.extend(self._to_list(q))
        msg.q_dot.extend(self._to_list(qdot))
        msg.q_ddot.extend(self._to_list(qddot))

        # Approximate normalized joint states.
        q_limits = robot.data.joint_pos_limits[0]
        q_den = (q_limits[:, 1] - q_limits[:, 0]).clamp_min(1e-6)
        q_norm = 2.0 * (q - q_limits[:, 0]) / q_den - 1.0
        msg.q_norm.extend(self._to_list(q_norm))
        qdot_norm = torch.tanh(qdot)
        msg.qdot_norm.extend(self._to_list(qdot_norm))

        msg.q_d.extend(self._to_list(ctc.q_d[0]))
        msg.qdot_d.extend(self._to_list(ctc.qdot_d[0]))
        msg.qddot_d.extend(self._to_list(ctc.qddot_d[0]))

        ee_pos = cmd.ee_pos_w[0]
        ee_yaw = cmd.ee_yaw[0]
        goal = cmd.command[0]
        goal_pos = goal[:3]
        goal_yaw = goal[3]
        e_pos = ee_pos - goal_pos
        e_yaw = ee_yaw - goal_yaw

        msg.pos_ef.extend(self._to_list(ee_pos))
        msg.yaw_ef = float(ee_yaw.item())
        msg.goal_pos.extend(self._to_list(goal_pos))
        msg.goal_yaw = float(goal_yaw.item())
        msg.e_pos.extend(self._to_list(e_pos))
        msg.e_yaw = float(e_yaw.item())

        msg.a_prev.extend(self._to_list(base_env.action_manager.prev_action[0]))
        msg.action_raw.extend(self._to_list(action[0]))
        msg.tau_raw.extend(self._to_list(ctc.tau_cmd[0]))

        msg.force_ext.extend(self._to_list(ctc.force_ext[0]))
        msg.wrench_ext.extend(self._to_list(ctc.wrench_ext[0, 3:]))
        msg.force_norm = float(ctc.force_norm[0].item())
        msg.tau_res_norm = float(ctc.tau_res_norm[0].item())

        msg.reward = float(reward[0].item())
        for name, value in base_env.reward_manager.get_active_iterable_terms(0):
            msg.reward_terms[name] = float(value[0])

        self._sock.sendto(msg.SerializeToString(), self._addr)


class TelemetryVecEnvWrapper:
    """Step-intercept wrapper that streams telemetry without modifying mjlab."""

    def __init__(self, env, streamer: TelemetryStreamer):
        self._env = env
        self._streamer = streamer

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, actions: torch.Tensor):
        obs, rew, dones, extras = self._env.step(actions)
        self._streamer.send_step(self._env, actions, rew)
        return obs, rew, dones, extras


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--agent",
        type=str,
        choices=("trained", "zero", "random"),
        default="trained",
    )
    parser.add_argument("--checkpoint-file", type=str, default=None)
    parser.add_argument("--wandb-run-path", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--viewer",
        type=str,
        choices=("auto", "native", "viser"),
        default="auto",
    )
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-length", type=int, default=600)
    parser.add_argument("--no-terminations", action="store_true")
    parser.add_argument("--telemetry", action="store_true")
    parser.add_argument("--telemetry-host", type=str, default="127.0.0.1")
    parser.add_argument("--telemetry-port", type=int, default=9870)
    parser.add_argument("--telemetry-every", type=int, default=1)
    parser.add_argument(
        "--policy",
        type=str,
        choices=("auto", "mlp", "lstm"),
        default="auto",
        help="Policy architecture for checkpoint loading.",
    )
    args = parser.parse_args()

    configure_torch_backends()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(TASK_ID, play=True)
    agent_cfg = load_rl_cfg(TASK_ID)

    dummy_mode = args.agent in {"zero", "random"}
    trained_mode = not dummy_mode

    if args.no_terminations:
        env_cfg.terminations = {}
        print("[INFO]: Terminations disabled")

    log_dir: Path | None = None
    resume_path: Path | None = None
    if trained_mode:
        log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
        if args.checkpoint_file is not None:
            resume_path = Path(args.checkpoint_file)
            if not resume_path.exists():
                raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
            print(f"[INFO]: Loading checkpoint: {resume_path.name}")
        else:
            if args.wandb_run_path is None:
                raise ValueError(
                    "`wandb_run_path` is required when `checkpoint_file` is not provided."
                )
            resume_path, was_cached = get_wandb_checkpoint_path(
                log_root_path, Path(args.wandb_run_path)
            )
            run_id = resume_path.parent.name
            checkpoint_name = resume_path.name
            cached_str = "cached" if was_cached else "downloaded"
            print(
                f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
            )
        log_dir = resume_path.parent

        policy = args.policy
        if policy == "auto":
            policy = _resolve_policy_from_checkpoint(resume_path)
            print(f"[INFO]: Resolved policy from checkpoint: {policy}")
        if policy == "lstm":
            agent_cfg = replace(
                agent_cfg,
                actor=replace(agent_cfg.actor, class_name="RNNModel"),
                critic=replace(agent_cfg.critic, class_name="RNNModel"),
            )
        elif policy == "mlp":
            agent_cfg = replace(
                agent_cfg,
                actor=replace(agent_cfg.actor, class_name="MLPModel"),
                critic=replace(agent_cfg.critic, class_name="MLPModel"),
            )
        else:
            raise ValueError(f"Unsupported policy: {policy}")

    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs

    render_mode = "rgb_array" if (trained_mode and args.video) else None
    if args.video and dummy_mode:
        print(
            "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
        )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if trained_mode and args.video:
        assert log_dir is not None
        print("[INFO] Recording videos during play")
        env = VideoRecorder(
            env,
            video_folder=log_dir / "videos" / "play",
            step_trigger=lambda step: step == 0,
            video_length=args.video_length,
            disable_logger=True,
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    if args.telemetry:
        streamer = TelemetryStreamer(
            host=args.telemetry_host,
            port=args.telemetry_port,
            every_n_steps=args.telemetry_every,
        )
        env = TelemetryVecEnvWrapper(env, streamer)

    if dummy_mode:
        action_shape: tuple[int, ...] = env.unwrapped.action_space.shape
        if args.agent == "zero":

            class PolicyZero:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return torch.zeros(action_shape, device=env.unwrapped.device)

            action_policy = PolicyZero()
        else:

            class PolicyRandom:
                def __call__(self, obs) -> torch.Tensor:
                    del obs
                    return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

            action_policy = PolicyRandom()
    else:
        runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        runner.load(
            str(resume_path), load_cfg={"actor": True}, strict=True, map_location=device
        )
        action_policy = runner.get_inference_policy(device=device)

    if args.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = args.viewer

    if resolved_viewer == "native":
        NativeMujocoViewer(env, action_policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, action_policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


if __name__ == "__main__":
    main()
