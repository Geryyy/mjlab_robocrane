# trainer/configs.py

import numpy as np


def get_algo_config(algo: str, n_envs: int):
    algo = algo.upper()

    if algo == "PPO":
        return dict(
            total_timesteps=int(1e7),
            learning_rate=4e-4,
            n_steps=512,
            batch_size=512 * n_envs,
            n_epochs=10,
            ent_coef=3e-4,
            clip_range=0.3,
            log_std=-1.0,
        )

    elif algo == "SAC":
        return dict(
            total_timesteps=int(1e7),
            learning_rate=3e-4,
            buffer_size=2_000_000,
            learning_starts=10_000,
            batch_size=512,
            train_freq=(1, "step"),
            gradient_steps=1,
            gamma=0.99,
            tau=0.005,
        )

    else:
        raise ValueError(f"Unknown algo {algo}")
