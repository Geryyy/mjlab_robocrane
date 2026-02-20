"""Environment configuration for RoboCrane joint-space control."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import (
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from .asset import get_robocrane_robot_cfg
from .ctc_action import JointAccelerationCtcActionCfg
from .mdp import GoalPoseCommandCfg
from . import mdp


def robocrane_jointspace_env_cfg(
  *,
  play: bool = False,
  num_envs: int = 1024,
) -> ManagerBasedRlEnvCfg:
  actor_terms = {
    "goal_pos_error": ObservationTermCfg(
      func=mdp.goal_position_error,
      params={"command_name": "goal_pose"},
      noise=Unoise(n_min=-0.002, n_max=0.002),
    ),
    "goal_yaw_error": ObservationTermCfg(
      func=mdp.goal_yaw_error,
      params={"command_name": "goal_pose"},
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "ee_pose": ObservationTermCfg(
      func=mdp.ee_pose,
      params={"command_name": "goal_pose"},
      noise=Unoise(n_min=-0.001, n_max=0.001),
    ),
    "goal_pose": ObservationTermCfg(
      func=mdp.goal_pose,
      params={"command_name": "goal_pose"},
    ),
    "joint_pos": ObservationTermCfg(
      func=envs_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.005, n_max=0.005),
    ),
    "joint_vel": ObservationTermCfg(
      func=envs_mdp.joint_vel_rel,
      noise=Unoise(n_min=-0.02, n_max=0.02),
    ),
    "actions": ObservationTermCfg(func=envs_mdp.last_action),
  }

  critic_terms = {**actor_terms}

  commands: dict[str, CommandTermCfg] = {
    "goal_pose": GoalPoseCommandCfg(
      resampling_time_range=(8.0, 12.0),
      entity_name="robot",
      ee_site_name="gripping_point",
    )
  }

  events = {
    "reset_scene_to_default": EventTermCfg(func=envs_mdp.reset_scene_to_default, mode="reset"),
    "reset_robot_joints": EventTermCfg(
      func=envs_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.03, 0.03),
        "velocity_range": (-0.01, 0.01),
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=("joint_[1-7]", "joint_cj1", "joint_cj2")
        ),
      },
    ),
  }

  rewards = {
    "position_tracking": RewardTermCfg(
      func=mdp.position_tracking_exp,
      weight=6.0,
      params={"command_name": "goal_pose", "std": 0.08},
    ),
    "yaw_tracking": RewardTermCfg(
      func=mdp.yaw_tracking_exp,
      weight=2.0,
      params={"command_name": "goal_pose", "std": 0.30},
    ),
    "success_bonus": RewardTermCfg(
      func=mdp.success_bonus,
      weight=8.0,
      params={
        "command_name": "goal_pose",
        "pos_threshold": 0.02,
        "yaw_threshold": 0.12,
      },
    ),
    "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.03),
    "joint_torque": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1.0e-4),
    "joint_pos_limits": RewardTermCfg(
      func=envs_mdp.joint_pos_limits,
      weight=-2.0,
      params={
        "asset_cfg": SceneEntityCfg(
          "robot", joint_names=("joint_[1-7]", "joint_cj1", "joint_cj2")
        )
      },
    ),
    "passive_joint_vel": RewardTermCfg(
      func=mdp.passive_joint_velocity_l2,
      weight=-0.05,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=("joint_cj1", "joint_cj2"))},
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
    "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
  }

  cfg = ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=num_envs,
      env_spacing=1.0,
      entities={"robot": get_robocrane_robot_cfg()},
    ),
    observations={
      "actor": ObservationGroupCfg(
        terms=actor_terms,
        concatenate_terms=True,
        enable_corruption=not play,
      ),
      "critic": ObservationGroupCfg(
        terms=critic_terms,
        concatenate_terms=True,
        enable_corruption=False,
      ),
    },
    actions={
      "joint_acc_ctc": JointAccelerationCtcActionCfg(
        entity_name="robot",
        joint_names=(
          "joint_1",
          "joint_2",
          "joint_3",
          "joint_4",
          "joint_5",
          "joint_6",
          "joint_7",
        ),
        # Match original acceleration action bounds.
        qddot_limits=(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0),
        # Match legacy velocity bounds (~deg/s from old env).
        qdot_limits=(
          1.48353,
          1.48353,
          1.74533,
          1.30900,
          2.26893,
          2.35619,
          2.35619,
        ),
        # XML actuator torque limits.
        torque_limits=(176.0, 176.0, 110.0, 110.0, 110.0, 40.0, 40.0),
        # Default computed-torque tracking gains.
        kp=(120.0, 120.0, 100.0, 90.0, 70.0, 60.0, 50.0),
        kd=(12.0, 12.0, 10.0, 9.0, 7.0, 6.0, 5.0),
        # Diagonal inertia approximation (tunable).
        inertia_diag=(8.0, 8.0, 5.0, 3.0, 2.0, 1.2, 0.8),
      )
    },
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="base",
      distance=1.2,
      elevation=-20.0,
      azimuth=120.0,
    ),
    sim=SimulationCfg(
      nconmax=256,
      njmax=2048,
      mujoco=MujocoCfg(
        timestep=0.001,
        iterations=20,
        ls_iterations=20,
        impratio=10,
        cone="elliptic",
      ),
    ),
    decimation=30,
    episode_length_s=20.0,
  )

  if play:
    cfg.episode_length_s = 1.0e9

  return cfg
