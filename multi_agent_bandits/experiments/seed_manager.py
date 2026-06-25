"""
seed_manager.py

Centralized seed management for reproducible experiments.
"""

import os
import random

import numpy as np
import torch


def generate_seeds(base_seed: int, n_seeds: int) -> list:
    """
    Generate a deterministic list of pseudo-random seeds from a base seed.

    Args:
        base_seed (int): The master seed.
        n_seeds (int): Number of unique seeds to generate.

    Returns:
        list: List of n_seeds unique random seeds.
    """
    rng = random.Random(base_seed)
    return [rng.randint(0, 2**31 - 1) for _ in range(n_seeds)]


def set_seed(seed: int, deterministic_cudnn: bool = True):
    """
    Set all random seeds for reproducibility.

    Args:
        seed (int): The seed value to use.
        deterministic_cudnn (bool): If True, configures CuDNN for deterministic
                                    behavior (may slow down training).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic_cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True

    os.environ["PYTHONHASHSEED"] = str(seed)


def reset_seed():
    """Reset all seeds to a new random value."""
    new_seed = random.randint(0, 2**32 - 1)
    set_seed(new_seed)
    return new_seed
