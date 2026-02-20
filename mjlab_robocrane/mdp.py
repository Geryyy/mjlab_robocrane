"""Custom commands, observations, rewards, and terminations for RoboCrane."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import euler_xyz_from_quat, sample_uniform, wrap_to_pi

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


class GoalPoseCommand(CommandTerm):
  cfg: GoalPoseCommandCfg

  def __init__(self, cfg: GoalPoseCommandCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    self.robot: Entity = env.scene[cfg.entity_name]

    site_ids, site_names = self.robot.find_sites(cfg.ee_site_name, preserve_order=True)
    if not site_ids:
      raise ValueError(
        f"Could not find end-effector site pattern '{cfg.ee_site_name}'. "
        f"Available sites: {self.robot.site_names}"
      )
    self._ee_site_id = site_ids[0]
    self._ee_site_name = site_names[0]

    self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.target_yaw = torch.zeros(self.num_envs, 1, device=self.device)

    self.metrics["goal_distance"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["goal_yaw_error"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.target_pos_w, self.target_yaw], dim=-1)

  @property
  def ee_site_name(self) -> str:
    return self._ee_site_name

  @property
  def ee_pos_w(self) -> torch.Tensor:
    return self.robot.data.site_pos_w[:, self._ee_site_id]

  @property
  def ee_yaw(self) -> torch.Tensor:
    ee_quat = self.robot.data.site_pose_w[:, self._ee_site_id, 3:7]
    yaw = euler_xyz_from_quat(ee_quat)[2]
    return yaw.unsqueeze(-1)

  def _update_metrics(self) -> None:
    pos_err = torch.linalg.norm(self.ee_pos_w - self.target_pos_w, dim=-1)
    yaw_err = torch.abs(wrap_to_pi((self.ee_yaw - self.target_yaw).squeeze(-1)))
    self.metrics["goal_distance"] = pos_err
    self.metrics["goal_yaw_error"] = yaw_err

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    n = len(env_ids)
    if n == 0:
      return

    pos_ranges = self.cfg.position_ranges
    lower = torch.tensor(
      [pos_ranges.x[0], pos_ranges.y[0], pos_ranges.z[0]], device=self.device
    )
    upper = torch.tensor(
      [pos_ranges.x[1], pos_ranges.y[1], pos_ranges.z[1]], device=self.device
    )
    pos = sample_uniform(lower, upper, (n, 3), device=self.device)
    pos = pos + self._env.scene.env_origins[env_ids]

    yaw = sample_uniform(
      self.cfg.yaw_range[0], self.cfg.yaw_range[1], (n, 1), device=self.device
    )

    self.target_pos_w[env_ids] = pos
    self.target_yaw[env_ids] = yaw

  def _update_command(self) -> None:
    pass


@dataclass(kw_only=True)
class GoalPoseCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  ee_site_name: str = "gripping_point"
  yaw_range: tuple[float, float] = (-3.14159, 3.14159)

  @dataclass
  class PositionRanges:
    x: tuple[float, float] = (0.20, 0.55)
    y: tuple[float, float] = (-0.25, 0.25)
    z: tuple[float, float] = (0.15, 0.55)

  position_ranges: PositionRanges = field(default_factory=PositionRanges)

  def build(self, env: "ManagerBasedRlEnv") -> GoalPoseCommand:
    return GoalPoseCommand(self, env)


def goal_position_error(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  cmd = cast(GoalPoseCommand, env.command_manager.get_term(command_name))
  return cmd.ee_pos_w - cmd.target_pos_w


def goal_yaw_error(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  cmd = cast(GoalPoseCommand, env.command_manager.get_term(command_name))
  return wrap_to_pi(cmd.ee_yaw - cmd.target_yaw)


def goal_pose(
  env: "ManagerBasedRlEnv",
  command_name: str,
) -> torch.Tensor:
  cmd = cast(GoalPoseCommand, env.command_manager.get_term(command_name))
  return cmd.command


def ee_pose(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
  cmd = cast(GoalPoseCommand, env.command_manager.get_term(command_name))
  return torch.cat([cmd.ee_pos_w, cmd.ee_yaw], dim=-1)


def position_tracking_exp(
  env: "ManagerBasedRlEnv", command_name: str, std: float
) -> torch.Tensor:
  pos_error = goal_position_error(env, command_name)
  return torch.exp(-torch.sum(torch.square(pos_error), dim=-1) / (std * std))


def yaw_tracking_exp(
  env: "ManagerBasedRlEnv", command_name: str, std: float
) -> torch.Tensor:
  yaw_error = goal_yaw_error(env, command_name).squeeze(-1)
  return torch.exp(-(yaw_error * yaw_error) / (std * std))


def success_bonus(
  env: "ManagerBasedRlEnv",
  command_name: str,
  pos_threshold: float,
  yaw_threshold: float,
) -> torch.Tensor:
  pos_error = torch.linalg.norm(goal_position_error(env, command_name), dim=-1)
  yaw_error = torch.abs(goal_yaw_error(env, command_name).squeeze(-1))
  return ((pos_error < pos_threshold) & (yaw_error < yaw_threshold)).float()


def passive_joint_velocity_l2(
  env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
  robot: Entity = env.scene[asset_cfg.name]
  return torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=-1)


def illegal_contact(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1) > 0

