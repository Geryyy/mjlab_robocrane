# trainer/train_robocrane.py

import argparse
import glob
import os
import time

import torch.nn as nn
import wandb
from callbacks import (
    CurriculumCallback,
    EvalCallback,
    InfoLoggerCallback,
    ProgressBarCallback,
    SaveDuringTrainingCallback,
)
from configs import get_algo_config
from env_factory import make_vec_env
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.vec_env import VecFrameStack, VecMonitor
from utils import clear_dir, cosine_schedule, linear_schedule
from wandb.integration.sb3 import WandbCallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env", type=str, choices=["cartesian", "jointspace"], default="cartesian"
    )
    parser.add_argument("--algo", type=str, choices=["PPO", "SAC"], default="PPO")
    parser.add_argument("--n_envs", type=int, default=32)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    env_name = args.env
    algo = args.algo.upper()
    n_envs = args.n_envs
    use_force = args.force

    cfg = get_algo_config(algo, n_envs)

    seed = int(time.time()) % 10000

    # Paths
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{algo}_{env_name}_{timestamp}_{'force' if use_force else ''}"
    base_dir = "./trained_models"
    best_dir = f"{base_dir}/best_{algo.lower()}/{env_name}"
    ckpt_dir = f"{base_dir}/checkpoints/{algo.lower()}/{env_name}"
    final_path = f"{base_dir}/{algo.lower()}_{env_name}_final"

    clear_dir(ckpt_dir)
    os.makedirs(best_dir, exist_ok=True)

    # WandB
    run = wandb.init(
        project="robocrane_rl",
        name=run_name,
        config=dict(algo=algo, env=env_name, **cfg),
        sync_tensorboard=True,
        save_code=True,
        settings=wandb.Settings(code_dir=".."),
    )

    # run.log_code(
    #     root=".",   # repository root relative to train.py
    #     include_fn=lambda path: path.endswith((".py", ".xml", ".proto")) or True
    # )

    # Build environments
    vec_env = make_vec_env(
        env_name, n_envs, seed, use_force=use_force, randomize_hfield=use_force
    )
    vec_env = VecMonitor(vec_env)
    # vec_env = VecFrameStack(vec_env, n_stack=5)

    # Load or create model
    if algo == "PPO":
        kwargs = dict(
            policy="MlpLstmPolicy",
            env=vec_env,
            learning_rate=cosine_schedule(
                cfg["learning_rate"], cfg["learning_rate"] * 0.01
            ),
            n_steps=cfg["n_steps"],
            batch_size=cfg["batch_size"],
            n_epochs=cfg["n_epochs"],
            ent_coef=cfg["ent_coef"],
            clip_range=cfg["clip_range"],
            policy_kwargs=dict(
                net_arch=dict(pi=[512, 256, 128], vf=[512, 256, 128]),
                activation_fn=nn.Tanh,
                # log_std_init=cfg["log_std"],
            ),
            tensorboard_log=f"./wandb/robocrane/runs/{run_name}",
            device="cuda",
            seed=seed,
            verbose=1,
        )
        model_class = RecurrentPPO

    else:  # SAC
        kwargs = dict(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=linear_schedule(cfg["learning_rate"]),
            buffer_size=cfg["buffer_size"],
            learning_starts=cfg["learning_starts"],
            batch_size=cfg["batch_size"],
            tau=cfg["tau"],
            gamma=cfg["gamma"],
            train_freq=cfg["train_freq"],
            gradient_steps=cfg["gradient_steps"],
            ent_coef="auto",
            policy_kwargs=dict(net_arch=[512, 256, 128], activation_fn=nn.ReLU),
            tensorboard_log=f"./wandb/robocrane/runs/{run_name}",
            device="cuda",
            seed=seed,
            verbose=1,
        )
        model_class = SAC

    # Finetune or new
    if args.finetune:
        ckpts = sorted(glob.glob(os.path.join(best_dir, "*.zip")))
        if not ckpts:
            raise FileNotFoundError(f"No best model found in {best_dir}")
        model_path = ckpts[-1]
        model = model_class.load(model_path, env=vec_env, device="cuda")
        print(f"Loaded model {model_path} for finetuning.")
    elif args.force:
        print("Using force environment!")
        ckpts = sorted(glob.glob(os.path.join(best_dir, "../best_model_lstm_acc.zip")))
        if not ckpts:
            raise FileNotFoundError(f"No best model found in {best_dir}")
        model_path = ckpts[-1]
        print(f"Loading model {model_path} for force finetuning.")
        pretrained = model_class.load(model_path, device="cuda")

        print("Creating new model and initializing with pretrained weights...")
        model = model_class(**kwargs)

        old_sd = pretrained.policy.state_dict()
        new_sd = model.policy.state_dict()

        for name, new_param in new_sd.items():
            if name not in old_sd:
                continue

            old_param = old_sd[name]

            # Case 1: shapes match → direct load
            if new_param.shape == old_param.shape:
                new_param.copy_(old_param)
                continue

            # Case 2: input layer expanded → partial reuse
            if new_param.ndim == 2 and old_param.ndim == 2:
                old_out, old_in = old_param.shape
                new_out, new_in = new_param.shape

                if old_out == new_out and new_in > old_in:
                    print(f"🔄 Partially loading layer {name} (expanded input).")

                    # copy old part
                    new_param[:, :old_in].copy_(old_param)

                    # initialize the extra columns
                    nn.init.orthogonal_(new_param[:, old_in:], gain=1.0)
                    continue

            # Case 3: incompatible layer (e.g., final layer) → keep new init
            print(f"⚠️ Reinitializing incompatible layer {name}.")
    else:
        model = model_class(**kwargs)

        # Reset the value function
        for n, p in model.policy.named_parameters():
            if ("policy_net" in n or "value_net" in n) and "weight" in n:
                nn.init.orthogonal_(p, gain=1.0)

        # Set std higher again because we want to learn something new
        for n, p in model.policy.named_parameters():
            if "log_std" in n:
                p.data.fill_(-0.5)

    # Callbacks
    callbacks = [
        WandbCallback(
            model_save_freq=200_000,
            model_save_path=f"./wandb/robocrane/models/{run_name}",
            verbose=2,
        ),
        SaveDuringTrainingCallback(ckpt_dir, save_freq=200_000, algo=algo),
        InfoLoggerCallback(log_freq=1000),
        CurriculumCallback(cfg["total_timesteps"], update_freq=50_000, verbose=1),
        ProgressBarCallback(cfg["total_timesteps"], algo),
        EvalCallback(
            vec_env,
            n_eval_episodes=5,
            best_model_save_path=best_dir,
            log_path=os.path.join(best_dir, "eval_logs"),
            eval_freq=10_000,
            deterministic=True,
            render=False,
        ),
    ]

    # Train
    model.learn(
        total_timesteps=cfg["total_timesteps"],
        callback=callbacks,
        progress_bar=False,
    )

    model.save(final_path)
    print(f"🎉 Saved final model to {final_path}.zip")

    wandb.finish()


if __name__ == "__main__":
    main()
