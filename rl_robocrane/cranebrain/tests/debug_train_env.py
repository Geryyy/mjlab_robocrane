# Debug training script with visualization
import os

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from robocrane_env import RobocraneEnv


class DebugCallback(BaseCallback):
    """Debug callback with detailed logging"""

    def __init__(self, log_freq=100):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self) -> bool:
        # Log metrics more frequently for debugging
        if self.num_timesteps % self.log_freq == 0:
            info = self.locals.get("infos", [{}])[0]  # Get first env info

            # Log all environment metrics
            metrics = {
                k: v
                for k, v in info.items()
                if k != "episode" and isinstance(v, (int, float, np.number))
            }
            if metrics:
                metrics["timesteps"] = self.num_timesteps
                wandb.log(metrics)

                # Print key metrics for console debugging
                if "success_rate" in metrics:
                    print(
                        f"Step {self.num_timesteps}: Success={metrics['success_rate']:.3f}, "
                        f"Action={metrics.get('action_norm', 0):.3f}, "
                        f"PassiveVel={metrics.get('passive_vel_norm', 0):.3f}"
                    )

        return True


if __name__ == "__main__":
    # Initialize WandB
    config = {
        "algorithm": "PPO",
        "environment": "RobocraneDamping-Debug",
        "n_envs": 1,  # Single environment for debugging
        "total_timesteps": 100_000,  # Shorter for debugging
        "learning_rate": 3e-3,
        "n_steps": 256,
        "batch_size": 256,
        "architecture": [256, 256],
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "ent_coef": 0.01,
    }

    wandb.init(
        project="robocrane-rl-damping-debug",
        config=config,
        name=f"debug-{wandb.util.generate_id()}",
        tags=["robocrane", "debug", "visualization"],
        notes="Debug training with video logging",
    )
    config = wandb.config

    # Setup device
    device = "cpu"  # Use CPU for better MLP performance
    print(f"Using device: {device}")

    # Create single environment
    env = RobocraneEnv()

    # Test environment
    print("Testing environment...")
    obs, info = env.reset()
    print(f"✓ Observation shape: {obs.shape}")
    print(f"✓ Info keys: {list(info.keys())}")

    # Test a few steps to see reward components
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"✓ Step successful. Reward: {reward:.3f}")
    print(f"✓ Info: {info}")

    # Create PPO model (no vectorization for debugging)
    model = PPO(
        "MlpPolicy",
        env,
        device=device,
        policy_kwargs=dict(net_arch=config.architecture, activation_fn=torch.nn.ReLU),
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=10,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        verbose=1,
    )

    # Setup debug callback
    debug_callback = DebugCallback(log_freq=100)

    print("Starting debug training...")

    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=debug_callback,
            progress_bar=True,
        )

        print("Debug training completed!")

    finally:
        wandb.finish()

    print("🎉 Debug session complete! Check WandB for videos and metrics.")

