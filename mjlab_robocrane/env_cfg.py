"""Environment configuration for RoboCrane joint-space control."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import (
    MetricsTermCfg,
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

from . import mdp
from .asset import get_robocrane_robot_cfg
from .ctc_action import JointAccelerationCtcActionCfg
from .mdp import GoalPoseCommandCfg


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
            curriculum_switch_steps=200_000,
        )
    }

    events = {
        "reset_scene_to_default": EventTermCfg(
            func=envs_mdp.reset_scene_to_default, mode="reset"
        ),
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
        "stay_alive": RewardTermCfg(
            func=mdp.stay_alive,
            weight=1.0,
            params={},
        ),
        "pose_tracking": RewardTermCfg(
            func=mdp.pose_tracking_exp,
            weight=1.0,
            params={"command_name": "goal_pose"},
        ),
        "success_bonus_free": RewardTermCfg(
            func=mdp.success_bonus,
            weight=20.0,
            params={
                "command_name": "goal_pose",
                "pos_threshold": 0.02,
                "yaw_threshold": 0.12,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("joint_3", "joint_5")
                ),
            },
        ),
        # "success_bonus_contact": RewardTermCfg(
        #     func=mdp.success_bonus_contact,
        #     weight=10.0,
        #     params={
        #         "command_name": "goal_pose",
        #         "pos_threshold": 0.03,
        #         "yaw_threshold": 0.16,
        #     },
        # ),
        # "tcp_force_tracking": RewardTermCfg(
        #   func=mdp.tcp_force_tracking_exp,
        #   weight=5.0,
        #   params={"desired_force": 22.0, "std": 6.0, "command_name": "goal_pose"},
        # ),
        "joint_velocity": RewardTermCfg(
            func=envs_mdp.joint_vel_l2,
            weight=-0.2,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=("joint_[1-7]")),
            },
        ),
        "action": RewardTermCfg(func=envs_mdp.action_l2, weight=-0.5),
        "action_rate": RewardTermCfg(func=envs_mdp.action_rate_l2, weight=-0.03),
        # "action_acc": RewardTermCfg(func=envs_mdp.action_acc_l2, weight=-0.02),
        "redundancy_joint_shape": RewardTermCfg(
            func=mdp.redundancy_joint_shaping_exp,
            weight=0.3,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("joint_3", "joint_5")
                ),
                "std": 0.45,
            },
        ),
        "passive_joint_pos_shape": RewardTermCfg(
            func=mdp.passive_joint_pos_shaping_exp,
            weight=0.35,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("joint_cj1", "joint_cj2")
                ),
            },
        ),
        # "joint_torque": RewardTermCfg(func=envs_mdp.joint_torques_l2, weight=-1.0e-4),
        "joint_pos_limits": RewardTermCfg(
            func=envs_mdp.joint_pos_limits,
            weight=-5.0,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("joint_[1-7]", "joint_cj1", "joint_cj2")
                )
            },
        ),
        "passive_joint_vel": RewardTermCfg(
            func=mdp.passive_joint_velocity_l2,
            weight=-0.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=("joint_cj1", "joint_cj2")
                )
            },
        ),
    }

    terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "nan_detection": TerminationTermCfg(func=envs_mdp.nan_detection),
    }

    metrics = {
        "tcp_force_norm": MetricsTermCfg(func=mdp.tcp_force_norm),
        "tcp_tau_res_norm": MetricsTermCfg(func=mdp.tcp_tau_residual_norm),
        "contact_phase": MetricsTermCfg(func=mdp.in_contact_phase),
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
                # Acceleration action bounds.
                qddot_limits=(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0),
                # Velocity bounds
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
                kp=(1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0, 1200.0),
                kd=(200.0, 200.0, 200.0, 200.0, 200.0, 200.0, 200.0),
                # Low-pass filtering on policy actions for smoother acceleration commands.
                action_lpf_alpha=0.0,
            )
        },
        commands=commands,
        events=events,
        rewards=rewards,
        terminations=terminations,
        metrics=metrics,
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
