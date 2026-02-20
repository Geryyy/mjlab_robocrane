# mjlab_robocrane

RoboCrane joint-space RL task implemented on:
- `mjlab` (MuJoCo Warp vectorized GPU simulation)
- `rsl_rl` PPO (through mjlab runner)

## Train

From repo root:

```bash
python3 mjlab_robocrane/train.py --n-envs 1024 --gpu-ids 0 --max-iterations 10000
```

Useful options:

```bash
python3 mjlab_robocrane/train.py --help
```

Examples:

```bash
# Multi-GPU
python3 mjlab_robocrane/train.py --gpu-ids 0,1 --n-envs 4096

# CPU only
python3 mjlab_robocrane/train.py --gpu-ids cpu --n-envs 64

# Auto-scale PPO mini-batches for large env counts
python3 mjlab_robocrane/train.py --n-envs 2048 --target-minibatch-size 2048

# Recurrent PPO policy (LSTM)
python3 mjlab_robocrane/train.py --policy lstm

# W&B run with tags
python3 mjlab_robocrane/train.py --logger wandb --wandb-project mjlab_robocrane --wandb-tags robocrane,lstm,ctc
```

## Play

```bash
python3 mjlab_robocrane/play.py \
  --checkpoint-file logs/rsl_rl/robocrane_jointspace/<run_dir>/model_<iter>.pt \
  --num-envs 1 \
  --viewer native

# For recurrent checkpoints, force LSTM if needed
python3 mjlab_robocrane/play.py --checkpoint-file <ckpt> --policy lstm
```

Dummy agents for debugging:

```bash
python3 mjlab_robocrane/play.py --agent random --num-envs 1
python3 mjlab_robocrane/play.py --agent zero --num-envs 1
```

## Conda Setup

From repo root:

```bash
conda env create -f mjlab_robocrane/environment.yml
conda activate mjlab_robocrane
```

Logs/checkpoints are written under:

```text
logs/rsl_rl/robocrane_jointspace/
```

## Notes

- Task id: `Mjlab-Robocrane-Jointspace-v0`
- Uses cleaned XML model at `rl_robocrane/robocrane/robocrane_mjlab_clean.xml`
- Controls `joint_1`..`joint_7` with computed-torque control driven by acceleration actions
- Includes automatic PPO `num_mini_batches` scaling from rollout batch size unless `--disable-auto-mini-batches` is set
- `--policy lstm` switches actor/critic to `rsl_rl` `RNNModel` (LSTM)
- Default logger is W&B; switch with `--logger tensorboard` if needed

W&B setup:

```bash
conda run -n mjlab_robocrane wandb login
```
