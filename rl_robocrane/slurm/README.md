# Slurm jobs for `rl_robocrane` on TU Wien GPU cluster

This folder contains Slurm job scripts to run the `rl_robocrane` experiments on the TU Wien GPU cluster (DGX).

It assumes:

- Your repo lives at: `~/repos/rl_robocrane`
- You use a Conda env called: `rl_robocrane`
- You have GPU access on the cluster


---

## 0. First time only: login and clone the repo

On your **local machine** (with VPN if needed):

```bash
ssh uXXXXXXX@dgx.tuwien.ac.at
```
Replace uXXXXXXX with your TU account.

On the cluster login node:

```bash
mkdir -p ~/repos
cd ~/repos

git clone <YOUR_GIT_REMOTE_URL> rl_robocrane
cd rl_robocrane
```

(If the repo is already there, just cd ~/repos/rl_robocrane and git pull to update.)

## 1. One-time setup: Conda environment on the cluster

You only need to create the environment once. All jobs (and all compute nodes) will reuse it.

From ~/repos/rl_robocrane:
```bash
# 1. Load modules so conda is available
module purge
module load anaconda
module load cuda/12.1
module load gcc/11

# 2. Create the environment (first time only)
conda env create -f environment.yml

# 3. Activate it
conda activate rl_robocrane
```

(If the env already exists, just do steps 1 + 3.)

## 1.1. Quick sanity check

Still in the repo root:
```bash
python -c "import torch; import mujoco; print('Torch CUDA:', torch.cuda.is_available())"
```


You should see Torch CUDA: True.
If that works, your environment is ready for Slurm.

# 2. Job scripts overview

This folder is expected to contain (for example):

- train_cartesian_gpu.job → trains PPO in 02_cartesian/
- train_jointspace_gpu.job → trains PPO in 01_jointspace/
- train_bc_gpu.job → imitation learning in 03_imitation_learning/
- logs/ → Slurm output/error logs

Each .job script:
- loads anaconda, cuda, gcc
- activates rl_robocrane
- sets MUJOCO_GL=osmesa for headless MuJoCo
- cds into the right subfolder
- runs the corresponding train.py script
- You don’t run these directly; you submit them with sbatch.

# 3. Submitting jobs (GPU training, 3 days runtime)

From the repo root:
```bash
cd ~/repos/rl_robocrane
```

## 3.1. Cartesian PPO (02_cartesian)
```bash
sbatch slurm/train_cartesian_gpu.job
```

# 4. Monitoring jobs and logs
## 4.1. Check running / pending jobs

`squeue -u $USER`

## 4.2. Cancel a job

## 4.3. Inspect logs
Slurm writes logs to slurm/logs/ (as configured in the .job files), e.g.:
```bash
cd ~/repos/rl_robocrane/slurm

ls logs
tail -f logs/cartesian_1234567.out
tail -f logs/cartesian_1234567.err
```
Use Ctrl+C to stop tail -f.