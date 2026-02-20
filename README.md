# RoboCrane RL Repository

This repository contains two RoboCrane reinforcement learning stacks:

- `mjlab_robocrane/`: current GPU-oriented setup using `mjlab` (MuJoCo Warp) + `rsl_rl` PPO
- `rl_robocrane/`: legacy setup based on Stable-Baselines and classic MuJoCo (mostly CPU-bound)

## Repository Layout

- `mjlab_robocrane/`
  - New task implementation (env, rewards, action/controller, training, play)
  - Start here for current development
- `mjlab/`
  - Third-party framework (git submodule)
- `rsl_rl/`
  - Third-party RL algorithms (git submodule)
- `rl_robocrane/`
  - Original project code (jointspace env + old training pipeline)
- `environment.yml`
  - Conda environment for the new `mjlab_robocrane` workflow
- `logs/`
  - Training outputs/checkpoints

## Quick Start (New Stack)

From repo root:

```bash
git submodule update --init --recursive
conda env create -f environment.yml
conda activate mjlab_robocrane
```

Optional (for Weights & Biases logging):

```bash
wandb login
```

## Train (New Stack)

```bash
python3 mjlab_robocrane/train.py --n-envs 1024 --gpu-ids 0 --max-iterations 10000
```

Useful variants:

```bash
# LSTM policy
python3 mjlab_robocrane/train.py --policy lstm

# TensorBoard instead of W&B
python3 mjlab_robocrane/train.py --logger tensorboard
```

## Play / Evaluate (New Stack)

```bash
python3 mjlab_robocrane/play.py \
  --checkpoint-file logs/rsl_rl/robocrane_jointspace/<run_dir>/model_<iter>.pt \
  --num-envs 1 \
  --viewer native
```

If needed, force policy type while loading checkpoint:

```bash
python3 mjlab_robocrane/play.py --checkpoint-file <ckpt> --policy lstm
```

## Legacy Stack (Stable-Baselines)

Legacy environment and scripts are in `rl_robocrane/`:

- Joint-space env: `rl_robocrane/jointspace/`
- Training scripts: `rl_robocrane/trainer/`
- Legacy conda file: `rl_robocrane/environment.yml`

Use this only if you need parity with the old implementation.

## Where To Look First

- Task registration: `mjlab_robocrane/task.py`
- Environment config (observations/actions/rewards/terminations): `mjlab_robocrane/env_cfg.py`
- Custom rewards/commands/MDP terms: `mjlab_robocrane/mdp.py`
- Computed-torque acceleration action: `mjlab_robocrane/ctc_action.py`
- Robot asset config: `mjlab_robocrane/asset.py`
- New stack details: `mjlab_robocrane/README.md`
