"""PPO configuration for RoboCrane task."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


def robocrane_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="tanh",
            obs_normalization=True,
            stochastic=True,
            init_noise_std=0.1,
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="tanh",
            obs_normalization=True,
            stochastic=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=3.0e-4,
            num_learning_epochs=8,
            num_mini_batches=8,
            learning_rate=3.0e-4,
            schedule="fixed",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="robocrane_jointspace",
        logger="wandb",
        wandb_project="mjlab_robocrane",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=10_000,
        clip_actions=1.0,
    )
