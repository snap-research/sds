import numpy as np
from collections import Counter

from sds.utils.misc import pseudo_shuffle, coprime_seed


def test_shuffle_no_collisions():
    N = 100_000
    seed = 123
    a = coprime_seed(seed, N)
    b = 42
    perm = [(a * i + b) % N for i in range(N)]
    assert len(set(perm)) == N, "Permutation has collisions"


def test_pseudo_shuffle_full_coverage():
    N = 10_000
    splits = 10
    seed = 1337
    all_indices = []
    for i in range(splits):
        chunk = pseudo_shuffle(N, seed, splits, i)
        assert len(chunk) == N // splits or i == splits - 1
        all_indices.extend(chunk)
    assert sorted(all_indices) == list(range(N)), "Magic shuffle misses or duplicates indices"


def test_entropy_of_positions():
    N = 1000
    splits = 10
    num_trials = 1_000
    sample_positions = 100
    tracked = np.random.default_rng(0).choice(N, size=sample_positions, replace=False)
    counters = {p: Counter() for p in tracked}

    for seed in range(num_trials):
        for i in range(splits):
            indices = pseudo_shuffle(N, seed, splits, i)
            base = i * (N // splits)
            for offset, val in enumerate(indices):
                global_pos = base + offset
                if global_pos in counters:
                    counters[global_pos][val] += 1

    entropies = []
    for p in tracked:
        freqs = np.array(list(counters[p].values()))
        probs = freqs / freqs.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))  # Avoid log(0)
        entropies.append(entropy)

    avg_entropy = np.mean(entropies)
    ideal_entropy = np.log2(N)
    print(f"Avg entropy: {avg_entropy:.2f} / {ideal_entropy:.2f} ({100 * avg_entropy / ideal_entropy:.2f}%)")
    assert avg_entropy > 0.95 * ideal_entropy, "Insufficient entropy in shuffled positions"

