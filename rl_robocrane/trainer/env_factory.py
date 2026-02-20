# trainer/env_factory.py

import os

import gymnasium as gym
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv

# Dynamically add root path for env imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
import sys

if ROOT not in sys.path:
    sys.path.append(ROOT)


def make_env(
    env_name: str,
    rank: int,
    seed: int,
    control_dt=0.02,
    max_episode_steps=512,
    randomize_com=False,
    randomize_hfield=False,
    use_force=False,
):
    def _init():
        if env_name == "cartesian":
            from cartesian.RobocraneCartesianEnv import RobocraneCartesianEnv

            env = RobocraneCartesianEnv(
                mj_model_path="./../robocrane/robocrane_contact.xml",
                pin_model_path="./../robocrane/robocrane_contact_pin.xml",
                max_episode_steps=max_episode_steps,
                control_dt=control_dt,
                randomize_body_com=randomize_com,
                randomize_hfield=randomize_hfield,
            )
        elif env_name == "jointspace":
            from jointspace.JointRobocraneEnv import JointRobocraneEnv

            env = JointRobocraneEnv(
                mj_model_path="./../robocrane/robocrane_contact.xml",
                pin_model_path="./../robocrane/robocrane_contact_pin.xml",
                max_episode_steps=max_episode_steps,
                control_dt=control_dt,
                randomize_body_com=randomize_com,
                randomize_hfield=randomize_hfield,
                use_force=use_force,
            )
        else:
            raise ValueError(f"Unknown environment type: {env_name}")

        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


def make_vec_env(env_name: str, n_envs: int, seed: int, **env_kwargs):
    env_fns = [make_env(env_name, i, seed, **env_kwargs) for i in range(n_envs)]
    return SubprocVecEnv(env_fns, start_method="forkserver")


# --------------------------------------------------------
# Make a single environment instance (non-vectorized)
# --------------------------------------------------------
def make_single_env(
    mode="cartesian",
    max_episode_steps=1000,
    randomize_body_com=False,
    randomize_hfield=False,
    expert=False,
    use_force=False,
    seed=None,
):
    """
    Returns ONE Robocrane environment instance.
    This is used by play.py for interactive visualization.
    """
    if mode == "cartesian":
        from cartesian.RobocraneCartesianEnv import RobocraneCartesianEnv

        env = RobocraneCartesianEnv(
            mj_model_path="./../robocrane/robocrane_contact.xml",
            pin_model_path="./../robocrane/robocrane_contact_pin.xml",
            max_episode_steps=max_episode_steps,
            control_dt=0.02,
            randomize_body_com=randomize_body_com,
            randomize_hfield=randomize_hfield,
        )
    elif mode == "jointspace":
        from jointspace.JointRobocraneEnv import JointRobocraneEnv

        env = JointRobocraneEnv(
            mj_model_path="./../robocrane/robocrane_contact.xml",
            pin_model_path="./../robocrane/robocrane_contact_pin.xml",
            max_episode_steps=max_episode_steps,
            control_dt=0.02,
            randomize_body_com=randomize_body_com,
            randomize_hfield=randomize_hfield,
            expert=expert,
            use_force=use_force,
        )
    else:
        raise ValueError(
            f"Unknown mode '{mode}', expected 'cartesian' or 'jointspace'."
        )

    if seed is not None:
        env.reset(seed=seed)

    return env
