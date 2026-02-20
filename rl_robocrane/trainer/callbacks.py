# trainer/callbacks.py

import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from tqdm import tqdm

from wandb.integration.sb3 import WandbCallback


class CurriculumCallback(BaseCallback):
    def __init__(self, total_timesteps, update_freq=100_000, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.update_freq = update_freq

    def _on_step(self):
        # Only update every update_freq steps
        if self.num_timesteps % self.update_freq != 0:
            return True

        # Simple linear schedule
        progress = self.num_timesteps / self.total_timesteps
        difficulty = min(1.0, progress)

        # Update environments in vectorized env
        self.training_env.env_method("set_difficulty", difficulty)

        if self.verbose:
            print(
                f"[Curriculum] Step {self.num_timesteps}, difficulty = {difficulty:.3f}"
            )
        return True


class ProgressBarCallback(BaseCallback):
    def __init__(self, total_timesteps: int, algo_name: str):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.algo_name = algo_name

    def _on_training_start(self):
        self.start = time.time()
        self.pbar = tqdm(
            total=self.total_timesteps, desc=f"Training {self.algo_name}", smoothing=0.1
        )

    def _on_step(self):
        self.pbar.n = self.model.num_timesteps
        self.pbar.refresh()
        return True

    def _on_training_end(self):
        self.pbar.close()
        print(f"⏳ Training finished in {(time.time() - self.start) / 60:.2f} min")


class SaveDuringTrainingCallback(BaseCallback):
    def __init__(self, save_path: str, save_freq: int = 200_000, algo: str = "algo"):
        super().__init__()
        os.makedirs(save_path, exist_ok=True)
        self.save_path = save_path
        self.save_freq = save_freq
        self.algo = algo.lower()

    def _on_step(self):
        if self.num_timesteps % self.save_freq == 0:
            model_path = os.path.join(
                self.save_path, f"{self.algo}_step_{self.num_timesteps:010d}"
            )
            self.model.save(model_path)
            if hasattr(self.model, "replay_buffer"):
                self.model.save_replay_buffer(model_path + "_replay")
            print(f"💾 Saved model -> {model_path}.zip")
        return True


class InfoLoggerCallback(BaseCallback):
    def __init__(self, log_freq=1000):
        super().__init__()
        self.log_freq = log_freq

    def _on_step(self):
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        log_data = {}
        for key in infos[0].keys():
            vals = [
                info[key]
                for info in infos
                if key in info and isinstance(info[key], (float, int, np.number))
            ]
            if vals:
                log_data[f"info/{key}"] = float(np.mean(vals))

        if log_data and (self.num_timesteps % self.log_freq == 0):
            import wandb

            wandb.log(log_data, step=self.num_timesteps)

        return True
