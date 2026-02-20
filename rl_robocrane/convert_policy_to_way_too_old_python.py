import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO

model = RecurrentPPO.load("best_model")
model2 = RecurrentPPO.load("best_model")

# Extract PyTorch state dict
state_dict = model.policy.state_dict()

# Convert each tensor to numpy
np_state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}

np.savez("ppo_policy_weights.npz", **np_state_dict)

print("Saved ppo_policy_weights.npz")

np_state_dict = np.load("ppo_policy_weights.npz")
state_dict = {k: torch.tensor(v) for k, v in np_state_dict.items()}
print(state_dict)

model2.policy.load_state_dict(state_dict)

# Compare models
state_dict = model.policy.state_dict()
state_dict2 = model2.policy.state_dict()
# Convert each tensor to numpy
np_state_dict = {k: v.cpu().numpy() for k, v in state_dict.items()}
np_state_dict2 = {k: v.cpu().numpy() for k, v in state_dict2.items()}

for (k, v), (k2, v2) in zip(np_state_dict.items(), np_state_dict2.items()):
    print(k == k2)
    print(np.array_equal(v, v2))
