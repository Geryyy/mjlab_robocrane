# rl_robocrane

Reinforcement learning setup for the 9-DoF **Robocrane** system (7 actuated + 2 passive joints).  
The agent learns torque-based control via a computed-torque (CT) controller in MuJoCo using **Stable-Baselines3** (PPO/SAC).  
The environment includes temporal observation history for better swing damping, collision avoidance, and smooth control.

---

## 🧩 Environment overview

**`RobocraneEnv`**
- 9-DoF full joint state (q, q̇) + gripper pose (x, y, z, yaw) + goal pose  
- 7-DoF normalized acceleration action → converted to torques via CT control (Pinocchio)  
- Augmented observation includes a short **history window** of past states and actions  
- Reward combines progress, stability, jerk, and passive damping terms (with detailed info for WandB logging)

---

## ⚙️ Setup


To recreate the exact same environment on another machine:

```bash
conda env create -f environment.yml
conda activate rl_robocrane
cd rl_robocrane
pip install -e ./cranebrain
```

```bash
# 1️⃣ Create environment
conda create -n rl_robocrane python=3.10 -c conda-forge
conda activate rl_robocrane

# 2️⃣ Install essentials
conda install jupyterlab pytorch::pytorch-cuda -c pytorch -c nvidia
conda install -c conda-forge mujoco gymnasium pinocchio protobuf tqdm wandb

# 3️⃣ Install cranebrain package
pip install -e ./cranebrain
```


## Training 
```bash
# PPO training
python train.py --algo PPO

# SAC training
python train.py --algo SAC
```


## Policy replay
```bash
# Auto-load latest PPO model
python play.py --algo PPO

# Auto-load latest SAC model
python play.py --algo SAC

# Load specific downloaded model
python play.py --algo SAC --model ./trained_models/best_sac/best_model.zip

# Stream live telemetry to PlotJuggler
python play.py --algo SAC --telemetry
```
