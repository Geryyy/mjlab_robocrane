"""Task registration for RoboCrane."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfg import robocrane_jointspace_env_cfg
from .rl_cfg import robocrane_ppo_runner_cfg

TASK_ID = "Mjlab-Robocrane-Jointspace-v0"

register_mjlab_task(
  task_id=TASK_ID,
  env_cfg=robocrane_jointspace_env_cfg(play=False),
  play_env_cfg=robocrane_jointspace_env_cfg(play=True),
  rl_cfg=robocrane_ppo_runner_cfg(),
)

