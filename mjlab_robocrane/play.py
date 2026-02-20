"""Play/evaluate a trained RoboCrane policy."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict, replace
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "mjlab" / "src"))
sys.path.insert(0, str(_ROOT / "rsl_rl"))
sys.path.insert(0, str(_ROOT))

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from mjlab_robocrane.task import TASK_ID


def _resolve_policy_from_checkpoint(checkpoint_file: Path) -> str:
  loaded = torch.load(str(checkpoint_file), map_location="cpu", weights_only=False)
  actor_state = loaded.get("actor_state_dict", {})
  return "lstm" if any(k.startswith("rnn.") for k in actor_state) else "mlp"


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
    print("[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir).")
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
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
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
