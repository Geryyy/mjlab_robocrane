"""Train RoboCrane in mjlab with rsl_rl PPO."""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "mjlab" / "src"))
sys.path.insert(0, str(_ROOT / "rsl_rl"))
sys.path.insert(0, str(_ROOT))

from mjlab.scripts.train import TrainConfig, launch_training

from mjlab_robocrane.task import TASK_ID


def _parse_gpu_ids(raw: str) -> list[int] | str | None:
  if raw.lower() == "cpu":
    return None
  if raw.lower() == "all":
    return "all"
  return [int(v.strip()) for v in raw.split(",") if v.strip()]


def _parse_tags(raw: str) -> tuple[str, ...]:
  if not raw.strip():
    return ()
  return tuple(v.strip() for v in raw.split(",") if v.strip())


def _pick_num_mini_batches(
  total_batch_size: int, target_minibatch_size: int, max_batches: int = 64
) -> int:
  if total_batch_size <= 0:
    return 1
  if target_minibatch_size <= 0:
    return 1

  # Prefer factors for exact splits; fallback to ceiling.
  desired_batches = max(1, math.ceil(total_batch_size / target_minibatch_size))
  candidates = [
    b for b in range(1, min(max_batches, total_batch_size) + 1) if total_batch_size % b == 0
  ]
  if not candidates:
    return min(max_batches, desired_batches)
  return min(candidates, key=lambda b: abs(b - desired_batches))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--n-envs", type=int, default=1024)
  parser.add_argument("--max-iterations", type=int, default=10_000)
  parser.add_argument("--num-steps-per-env", type=int, default=24)
  parser.add_argument(
    "--policy",
    type=str,
    choices=("mlp", "lstm"),
    default="lstm",
    help="Policy architecture for actor/critic.",
  )
  parser.add_argument("--target-minibatch-size", type=int, default=2048)
  parser.add_argument("--disable-auto-mini-batches", action="store_true")
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--experiment-name", type=str, default="robocrane_jointspace")
  parser.add_argument("--run-name", type=str, default="")
  parser.add_argument(
    "--logger",
    type=str,
    choices=("wandb", "tensorboard"),
    default=None,
    help="Training logger backend. Default: value from rl_cfg.py.",
  )
  parser.add_argument(
    "--wandb-project",
    type=str,
    default=None,
    help="W&B project name. Default: value from rl_cfg.py.",
  )
  parser.add_argument(
    "--wandb-tags",
    type=str,
    default="",
    help='Comma-separated W&B tags, e.g. "robocrane,lstm,ctc".',
  )
  parser.add_argument(
    "--gpu-ids",
    type=str,
    default="0",
    help='Examples: "0", "0,1", "all", "cpu"',
  )
  args = parser.parse_args()

  cfg = TrainConfig.from_task(TASK_ID)
  cfg.env.scene.num_envs = args.n_envs
  cfg.env.seed = args.seed

  if args.policy == "lstm":
    actor_cfg = replace(cfg.agent.actor, class_name="RNNModel")
    critic_cfg = replace(cfg.agent.critic, class_name="RNNModel")
    cfg = replace(cfg, agent=replace(cfg.agent, actor=actor_cfg, critic=critic_cfg))
    print("[INFO] Using recurrent policy: RNNModel (LSTM)")
  else:
    print("[INFO] Using feed-forward policy: MLPModel")

  algorithm_cfg = cfg.agent.algorithm
  if args.disable_auto_mini_batches:
    num_mini_batches = algorithm_cfg.num_mini_batches
  else:
    rollout_batch = args.n_envs * args.num_steps_per_env
    num_mini_batches = _pick_num_mini_batches(
      total_batch_size=rollout_batch,
      target_minibatch_size=args.target_minibatch_size,
    )
    print(
      "[INFO] Auto mini-batches: "
      f"rollout_batch={rollout_batch}, "
      f"target_minibatch_size={args.target_minibatch_size}, "
      f"num_mini_batches={num_mini_batches}, "
      f"actual_minibatch_size={rollout_batch // num_mini_batches}"
    )
  algorithm_cfg = replace(algorithm_cfg, num_mini_batches=num_mini_batches)

  agent_cfg = replace(
    cfg.agent,
    seed=args.seed,
    max_iterations=args.max_iterations,
    num_steps_per_env=args.num_steps_per_env,
    algorithm=algorithm_cfg,
    experiment_name=args.experiment_name,
    run_name=args.run_name,
    logger=(args.logger or cfg.agent.logger),
    wandb_project=(args.wandb_project or cfg.agent.wandb_project),
    wandb_tags=_parse_tags(args.wandb_tags) if args.wandb_tags else cfg.agent.wandb_tags,
  )
  cfg = replace(cfg, agent=agent_cfg, gpu_ids=_parse_gpu_ids(args.gpu_ids))

  launch_training(TASK_ID, cfg)


if __name__ == "__main__":
  main()
