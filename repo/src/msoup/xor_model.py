"""XOR-SAT micro-model utilities for the decision harness."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gf2 import gf2_rank, gf2_matmul, gf2_row_reduce


@dataclass(frozen=True)
class XorConstraintPair:
    a_minus: np.ndarray
    b_minus: np.ndarray
    a_plus: np.ndarray
    b_plus: np.ndarray

    @property
    def delta_m(self) -> int:
        return gf2_rank(self.a_plus) - gf2_rank(self.a_minus)


def _sample_full_rank_matrix(
    rng: np.random.Generator,
    n_rows: int,
    n_vars: int,
    max_attempts: int = 200,
) -> np.ndarray:
    for _ in range(max_attempts):
        mat = rng.integers(0, 2, size=(n_rows, n_vars), dtype=np.uint8)
        if gf2_rank(mat) == n_rows:
            return mat
    raise RuntimeError("Unable to generate full-rank GF(2) matrix")


def _sample_dependent_rows(
    rng: np.random.Generator,
    basis: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    if n_rows <= 0:
        return np.zeros((0, basis.shape[1]), dtype=np.uint8)
    rows = []
    while len(rows) < n_rows:
        coeffs = rng.integers(0, 2, size=basis.shape[0], dtype=np.uint8)
        if not coeffs.any():
            continue
        row = (coeffs @ basis) % 2
        if row.any():
            rows.append(row.astype(np.uint8))
    return np.vstack(rows)


def _sample_independent_rows_against(
    rng: np.random.Generator,
    existing: np.ndarray,
    extra: int,
) -> np.ndarray:
    reduced = gf2_row_reduce(existing)
    basis = reduced.matrix[: reduced.rank].copy()
    pivots = list(reduced.pivots)
    rows = []
    attempts = 0
    max_attempts = 2000
    while len(rows) < extra:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("Unable to add independent constraints")
        candidate = rng.integers(0, 2, size=existing.shape[1], dtype=np.uint8)
        reduced_candidate = candidate.copy()
        for pivot_col, basis_row in zip(pivots, basis):
            if reduced_candidate[pivot_col] == 1:
                reduced_candidate ^= basis_row
        if reduced_candidate.any():
            pivot_col = int(np.argmax(reduced_candidate))
            for idx, basis_row in enumerate(basis):
                if basis_row[pivot_col] == 1:
                    basis[idx] ^= reduced_candidate
            basis = np.vstack([basis, reduced_candidate])
            pivots.append(pivot_col)
            rows.append(candidate)
    return np.vstack(rows)


def generate_constraint_pair(
    rng: np.random.Generator,
    n: int,
    m_base: int,
    delta_m: int,
) -> XorConstraintPair:
    """Generate consistent XOR constraints with controlled rank increase."""
    target_rank = min(m_base, n - delta_m)
    if target_rank < 1:
        raise ValueError("target_rank must be at least 1")
    base_basis = _sample_full_rank_matrix(rng, target_rank, n)
    dependent = _sample_dependent_rows(rng, base_basis, m_base - target_rank)
    a_minus = np.vstack([base_basis, dependent]).astype(np.uint8)

    extra_basis = _sample_independent_rows_against(rng, a_minus, delta_m)
    a_plus = np.vstack([a_minus, extra_basis]).astype(np.uint8)
    if gf2_rank(a_plus) - gf2_rank(a_minus) != delta_m:
        raise RuntimeError("Failed to generate required independent constraints")

    assignment = rng.integers(0, 2, size=n, dtype=np.uint8)
    b_minus = gf2_matmul(a_minus, assignment)
    b_plus = gf2_matmul(a_plus, assignment)
    return XorConstraintPair(a_minus=a_minus, b_minus=b_minus, a_plus=a_plus, b_plus=b_plus)
