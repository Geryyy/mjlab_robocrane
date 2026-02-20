# trainer/utils.py

import os
import shutil

import numpy as np


def clear_dir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def linear_schedule(initial_value):
    def func(progress_remaining: float):
        return progress_remaining * initial_value

    return func


def cosine_schedule(initial_value: float, final_value: float):
    def func(progress_remaining: float) -> float:
        progress = 1 - progress_remaining
        cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
        return final_value + (initial_value - final_value) * cosine_decay

    return func
