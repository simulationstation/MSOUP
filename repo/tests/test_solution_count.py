import numpy as np

from msoup.constraints import ConstraintSystem
from msoup.entropy import solution_count
from msoup.gf2 import gf2_matmul


def brute_force_count(system: ConstraintSystem) -> int:
    n = system.n_vars
    count = 0
    for value in range(2 ** n):
        x = ((value >> np.arange(n)) & 1).astype(np.uint8)
        if np.all((gf2_matmul(system.matrix, x) ^ system.rhs) == 0):
            count += 1
    return count


def test_solution_count_bruteforce():
    rng = np.random.default_rng(321)
    n = 10
    matrix = rng.integers(0, 2, size=(6, n), dtype=np.uint8)
    rhs = rng.integers(0, 2, size=6, dtype=np.uint8)
    system = ConstraintSystem(matrix=matrix, rhs=rhs)
    assert solution_count(system) == brute_force_count(system)
