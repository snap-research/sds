import os
import math
import warnings
import logging
import datetime
from contextlib import contextmanager
from typing import Any

from loguru import logger
from beartype import beartype
import torch

#----------------------------------------------------------------------------
# Shuffling utils.

@beartype
def pseudo_shuffle(num_samples: int, seed: int, num_splits: int, split_idx: int) -> list[int]:
    """
    This function shuffles a dataset and returns the indices of a given rank.
    Somewhat similar to:
        > np.random.RandomState(seed).permutation(list(range(num_samples)))[split_idx * (num_samples // num_splits):(split_idx + 1) * (num_samples // num_splits)],
    but much faster and more memory efficient. It does not create the entire list of indices, so works in O(split_size) memory.
    TODO: switch to better PRPs (e.g., with a Feistel network).
    """
    assert 0 <= split_idx < num_splits
    per_split = num_samples // num_splits
    start = split_idx * per_split
    end = start + per_split if split_idx < num_splits - 1 else num_samples

    a = coprime_seed(seed, num_samples)
    b = (seed ^ 0xdeadbeef) % num_samples  # some fast mixing

    return [(a * i + b) % num_samples for i in range(start, end)]

@beartype
def coprime_seed(seed: int, N: int) -> int:
    """Finds an odd integer a ≥ 1 seeded from `seed` such that gcd(a, N) = 1. Runs in O(log^2(N)) time."""
    a = (seed << 1) | 1  # always odd
    while math.gcd(a, N) != 1:
        a += 2
    return a % N

#----------------------------------------------------------------------------
