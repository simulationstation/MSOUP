"""Constraint construction utilities for XOR-SAT systems."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .gf2 import gf2_rank, gf2_row_reduce, to_gf2


@dataclass(frozen=True)
class ConstraintSystem:
    matrix: np.ndarray
    rhs: np.ndarray

    @property
    def n_vars(self) -> int:
        return self.matrix.shape[1]

    @property
    def n_constraints(self) -> int:
        return self.matrix.shape[0]

    @property
    def rank(self) -> int:
        return gf2_rank(self.matrix)


@dataclass(frozen=True)
class ConstraintPair:
    peeled: ConstraintSystem
    pinned: ConstraintSystem

    @property
    def delta_m(self) -> int:
        return self.pinned.rank - self.peeled.rank


def _random_xor_matrix(rng: np.random.Generator, m: int, n: int, density: float) -> np.ndarray:
    mat = rng.random((m, n)) < density
    return mat.astype(np.uint8)


def build_random_constraints(
    rng: np.random.Generator,
    n: int,
    m: int,
    density: float,
    ensure_rank: int | None = None,
    assignment: np.ndarray | None = None,
) -> ConstraintSystem:
    attempts = 0
    while True:
        attempts += 1
        matrix = _random_xor_matrix(rng, m, n, density)
        if ensure_rank is not None and gf2_rank(matrix) < ensure_rank:
            if attempts < 100:
                continue
        x = assignment if assignment is not None else rng.integers(0, 2, size=n, dtype=np.uint8)
        rhs = (matrix @ x) % 2
        return ConstraintSystem(matrix=matrix, rhs=rhs.astype(np.uint8))


def add_independent_constraints(
    rng: np.random.Generator,
    base: ConstraintSystem,
    extra: int,
    density: float,
    assignment: np.ndarray | None = None,
) -> ConstraintSystem:
    matrix = base.matrix.copy()
    rhs = base.rhs.copy()
    reduced = gf2_row_reduce(matrix)
    basis = reduced.matrix[: reduced.rank].copy()
    pivots = list(reduced.pivots)
    added = 0
    attempts = 0
    max_attempts = 1000
    while added < extra:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Unable to add independent constraints within limit.")
        new_row = _random_xor_matrix(rng, 1, base.n_vars, density)[0]
        candidate = new_row.copy()
        for pivot_col, basis_row in zip(pivots, basis):
            if candidate[pivot_col] == 1:
                candidate ^= basis_row
        if candidate.any():
            pivot_col = int(np.argmax(candidate))
            for idx, basis_row in enumerate(basis):
                if basis_row[pivot_col] == 1:
                    basis[idx] ^= candidate
            basis = np.vstack([basis, candidate])
            pivots.append(pivot_col)
            matrix = np.concatenate([matrix, new_row.reshape(1, -1)], axis=0)
            if assignment is None:
                new_rhs = rng.integers(0, 2, size=1, dtype=np.uint8)
            else:
                new_rhs = np.array([(new_row @ assignment) % 2], dtype=np.uint8)
            rhs = np.concatenate([rhs, new_rhs.astype(np.uint8)])
            added += 1
    return ConstraintSystem(matrix=matrix, rhs=rhs)


def build_constraint_pair(
    rng: np.random.Generator,
    n: int,
    m_base: int,
    delta_m: int,
    density: float,
) -> ConstraintPair:
    effective_m = min(m_base, n - delta_m)
    assignment = rng.integers(0, 2, size=n, dtype=np.uint8)
    peeled = build_random_constraints(
        rng,
        n=n,
        m=effective_m,
        density=density,
        assignment=assignment,
        ensure_rank=min(effective_m, n - delta_m),
    )
    pinned = add_independent_constraints(
        rng,
        peeled,
        extra=delta_m,
        density=density,
        assignment=assignment,
    )
    return ConstraintPair(peeled=peeled, pinned=pinned)


def combine_constraints(systems: Tuple[ConstraintSystem, ...]) -> ConstraintSystem:
    matrices = [to_gf2(sys.matrix) for sys in systems]
    rhs = [to_gf2(sys.rhs) for sys in systems]
    return ConstraintSystem(matrix=np.concatenate(matrices, axis=0), rhs=np.concatenate(rhs, axis=0))
