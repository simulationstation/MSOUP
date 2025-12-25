"""Exact entropy calculations for XOR-SAT systems."""
from __future__ import annotations

import numpy as np

from .constraints import ConstraintSystem
from .gf2 import gf2_is_consistent, gf2_rank


def solution_count(system: ConstraintSystem) -> int:
    if not gf2_is_consistent(system.matrix, system.rhs):
        return 0
    rank = gf2_rank(system.matrix)
    return 2 ** (system.n_vars - rank)


def entropy_nats(system: ConstraintSystem) -> float:
    count = solution_count(system)
    if count == 0:
        return float("-inf")
    return np.log(count)


def delta_entropy(peeled: ConstraintSystem, pinned: ConstraintSystem) -> float:
    return entropy_nats(pinned) - entropy_nats(peeled)
