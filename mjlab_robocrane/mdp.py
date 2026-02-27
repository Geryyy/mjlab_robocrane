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

from .ctc_action import JointAccelerationCtcAction

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")
qdot_limits = torch.Tensor(
    [
        1.48353,
        1.48353,
        1.74533,
        1.30900,
        2.26893,
        2.35619,
        2.35619,
    ]
)


class GoalPoseCommand(CommandTerm):
    cfg: GoalPoseCommandCfg

    def __init__(self, cfg: GoalPoseCommandCfg, env: "ManagerBasedRlEnv"):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        site_ids, site_names = self.robot.find_sites(
            cfg.ee_site_name, preserve_order=True
        )
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
        self.metrics["phase_contact"] = torch.zeros(self.num_envs, device=self.device)

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
        self.metrics["phase_contact"] = torch.full(
            (self.num_envs,), float(self._is_contact_phase()), device=self.device
        )

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return

        if self._is_contact_phase():
            pos_ranges = self.cfg.contact_position_ranges
            yaw_low, yaw_high = self.cfg.contact_yaw_range
        else:
            pos_ranges = self.cfg.free_position_ranges
            yaw_low, yaw_high = self.cfg.free_yaw_range

        lower = torch.tensor(
            [pos_ranges.x[0], pos_ranges.y[0], pos_ranges.z[0]], device=self.device
        )
        upper = torch.tensor(
            [pos_ranges.x[1], pos_ranges.y[1], pos_ranges.z[1]], device=self.device
        )
        pos = sample_uniform(lower, upper, (n, 3), device=self.device)
        pos = pos + self._env.scene.env_origins[env_ids]

        yaw = sample_uniform(yaw_low, yaw_high, (n, 1), device=self.device)

        self.target_pos_w[env_ids] = pos
        self.target_yaw[env_ids] = yaw

    def _update_command(self) -> None:
        pass

    def _is_contact_phase(self) -> bool:
        return self._env.common_step_counter >= self.cfg.curriculum_switch_steps


@dataclass(kw_only=True)
class GoalPoseCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    ee_site_name: str = "gripping_point"
    free_yaw_range: tuple[float, float] = (-3.14159, 3.14159)
    contact_yaw_range: tuple[float, float] = (-0.6, 0.6)
    curriculum_switch_steps: int = 200_000

    @dataclass
    class PositionRanges:
        x: tuple[float, float] = (0.20, 0.55)
        y: tuple[float, float] = (-0.25, 0.25)
        z: tuple[float, float] = (0.15, 0.55)

    free_position_ranges: PositionRanges = field(default_factory=PositionRanges)

    @dataclass
    class ContactPositionRanges:
        x: tuple[float, float] = (0.42, 0.48)
        y: tuple[float, float] = (-0.05, 0.05)
        z: tuple[float, float] = (0.235, 0.255)

    contact_position_ranges: ContactPositionRanges = field(
        default_factory=ContactPositionRanges
    )

    def build(self, env: "ManagerBasedRlEnv") -> GoalPoseCommand:
        return GoalPoseCommand(self, env)


def in_contact_phase(
    env: "ManagerBasedRlEnv", command_name: str = "goal_pose"
) -> torch.Tensor:
    cmd = cast(GoalPoseCommand, env.command_manager.get_term(command_name))
    return torch.full(
        (env.num_envs,), float(cmd._is_contact_phase()), device=env.device
    )


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


def pose_tracking_exp(env: "ManagerBasedRlEnv", command_name: str) -> torch.Tensor:
    pos_error = goal_position_error(env, command_name)
    pos_error_norm = torch.linalg.norm(pos_error, dim=-1)
    r_pos1 = torch.exp(-20 * pos_error_norm)
    r_pos2 = torch.exp(-10 * pos_error_norm)
    r_pos3 = torch.exp(-5 * pos_error_norm)
    yaw_error = goal_yaw_error(env, command_name).squeeze(-1)
    yaw_error_norm = torch.abs(yaw_error)
    r_yaw1 = torch.exp(-16 * yaw_error_norm)
    r_yaw2 = torch.exp(-8 * yaw_error_norm)
    r_yaw3 = torch.exp(-2 * yaw_error_norm)
    r_pose = r_pos1 * r_yaw1 + r_pos2 * r_yaw2 + r_pos3 * r_yaw3
    return r_pose


def success_bonus(
    env: "ManagerBasedRlEnv",
    command_name: str,
    pos_threshold: float,
    yaw_threshold: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    pos_error = torch.linalg.vector_norm(goal_position_error(env, command_name), dim=-1)
    # yaw_error = torch.abs(goal_yaw_error(env, command_name).squeeze(-1))
    dq = robot.data.joint_vel[:, asset_cfg.joint_ids]
    sway_err = torch.linalg.vector_norm(dq, dim=-1)
    r_success = torch.where(
        pos_error < pos_threshold, torch.exp(-sway_err), torch.zeros_like(pos_error)
    )
    return r_success


def success_bonus_contact(
    env: "ManagerBasedRlEnv",
    command_name: str,
    pos_threshold: float,
    yaw_threshold: float,
) -> torch.Tensor:
    phase = in_contact_phase(env, command_name)
    return phase * success_bonus(env, command_name, pos_threshold, yaw_threshold)


def passive_joint_velocity_l2(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    dq = robot.data.joint_vel[:, asset_cfg.joint_ids]
    dq /= 50
    return torch.sum(torch.square(dq), dim=-1)


def joint_limit_violation(
    env: "ManagerBasedRlEnv", asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    q_limit = robot.data.joint_pos_limits[:, asset_cfg.joint_ids]
    return torch.any(q >= q_limit[:, :, 1]) or torch.any(q <= q_limit[:, :, 0])


def joint_vel_l2(
    env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG
) -> torch.Tensor:
    """Penalize joint velocities on the articulation using L2 squared kernel."""
    robot: Entity = env.scene[asset_cfg.name]
    dq = robot.data.joint_vel[:, asset_cfg.joint_ids]
    dq = torch.clip(
        dq, -qdot_limits[asset_cfg.joint_ids], qdot_limits[asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(robot.data.joint_vel[:, asset_cfg.joint_ids]), dim=1)


def action_l2(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Penalize the actions using L2 squared kernel."""
    return torch.sum(torch.square(env.action_manager.action), dim=1)


def passive_joint_pos_shaping_exp(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    q_norm_sq = torch.sum(torch.square(q), dim=-1)
    r_pas_jnt = torch.exp(-10 * q_norm_sq)
    r_pas_jnt += torch.exp(-5 * q_norm_sq)
    r_pas_jnt += torch.exp(-1 * q_norm_sq)
    return r_pas_jnt


def redundancy_joint_shaping_exp(
    env: "ManagerBasedRlEnv",
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    robot: Entity = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.exp(-torch.sum(torch.square(q), dim=-1))


def illegal_contact(env: "ManagerBasedRlEnv", sensor_name: str) -> torch.Tensor:
    sensor: ContactSensor = env.scene[sensor_name]
    assert sensor.data.found is not None
    return sensor.data.found.squeeze(-1) > 0


def _ctc_action_term(
    env: "ManagerBasedRlEnv", action_term_name: str = "joint_acc_ctc"
) -> JointAccelerationCtcAction:
    return cast(
        JointAccelerationCtcAction, env.action_manager.get_term(action_term_name)
    )


def tcp_force_norm(
    env: "ManagerBasedRlEnv", action_term_name: str = "joint_acc_ctc"
) -> torch.Tensor:
    action_term = _ctc_action_term(env, action_term_name)
    return action_term.force_norm


def tcp_tau_residual_norm(
    env: "ManagerBasedRlEnv", action_term_name: str = "joint_acc_ctc"
) -> torch.Tensor:
    action_term = _ctc_action_term(env, action_term_name)
    return action_term.tau_res_norm


def tcp_force_tracking_exp(
    env: "ManagerBasedRlEnv",
    desired_force: float,
    std: float,
    command_name: str = "goal_pose",
    action_term_name: str = "joint_acc_ctc",
) -> torch.Tensor:
    action_term = _ctc_action_term(env, action_term_name)
    force_norm = action_term.force_norm
    force_reward = torch.exp(-torch.square(force_norm - desired_force) / (std * std))
    phase = in_contact_phase(env, command_name)
    return phase * force_reward
