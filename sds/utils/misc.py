import math

import numpy as np
from beartype import beartype

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
# Misc utils.

@beartype
def probabilities_to_counts(probabilities: list[float] | np.ndarray, min_count: int = 1) -> list[int]:
    """
    Converts a list of probabilities to counts, ensuring that each count is at least `min_count`.
    The sum of the counts will be equal to the sum of the probabilities multiplied by the total number of samples.
    """
    assert all(p >= 0 for p in probabilities), "Probabilities must be non-negative."
    total_prob = sum(probabilities)

    if total_prob == 0:
        return [min_count] * len(probabilities)
    if all(p == 1 / len(probabilities) for p in probabilities):
        # A convenient shortcut: if all probabilities are equal, return the same count for each.
        return [min_count] * len(probabilities)

    probabilities = np.array(probabilities)
    counts = probabilities / min([p for p in probabilities if p > 0])  # Normalize to avoid division by zero.
    counts = np.round(counts).astype(int)
    counts[counts < min_count] = min_count
    counts[probabilities == 0] = 0

    return counts.tolist()

@beartype
def normalize_ratios(ratios: list[float | int | None]) -> np.ndarray:
    if any(r is None for r in ratios):
        assert all(r is None for r in ratios), f"All ratios must be None or all must be specified. Got: {ratios}"
        ratios = [1.0] * len(ratios)

    ratios = np.array(ratios, dtype=float)
    assert ratios.min() >= 0, f"Ratios must be non-negative. Got: {ratios}"
    assert ratios.max() > 0, f"Ratios must not be all zeros. Got: {ratios}"
    ratios = ratios / np.sum(ratios)

    return ratios

#----------------------------------------------------------------------------
