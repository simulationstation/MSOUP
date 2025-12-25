import itertools

import numpy as np

from msoup.gf2 import gf2_is_consistent, gf2_rank


def brute_force_count(matrix: np.ndarray, rhs: np.ndarray) -> int:
    n_vars = matrix.shape[1]
    count = 0
    for bits in itertools.product([0, 1], repeat=n_vars):
        x = np.array(bits, dtype=np.uint8)
        if np.all((matrix @ x) % 2 == rhs):
            count += 1
    return count


def test_small_bruteforce_counts_match_rank() -> None:
    rng = np.random.default_rng(123)
    for _ in range(5):
        n_vars = 8
        n_constraints = 6
        matrix = rng.integers(0, 2, size=(n_constraints, n_vars), dtype=np.uint8)
        rhs = rng.integers(0, 2, size=n_constraints, dtype=np.uint8)
        brute_count = brute_force_count(matrix, rhs)
        if gf2_is_consistent(matrix, rhs):
            expected = 2 ** (n_vars - gf2_rank(matrix))
        else:
            expected = 0
        assert brute_count == expected
