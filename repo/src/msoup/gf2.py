"""GF(2) linear algebra utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


def to_gf2(matrix: np.ndarray) -> np.ndarray:
    return np.array(matrix, dtype=np.uint8) % 2


def gf2_rank(matrix: np.ndarray) -> int:
    """Compute rank over GF(2) using row reduction."""
    mat = to_gf2(matrix).copy()
    m, n = mat.shape
    rank = 0
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if mat[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            mat[[row, pivot]] = mat[[pivot, row]]
        for r in range(m):
            if r != row and mat[r, col] == 1:
                mat[r, :] ^= mat[row, :]
        rank += 1
        row += 1
        if row == m:
            break
    return rank


def gf2_rank_augmented(matrix: np.ndarray, vector: np.ndarray) -> int:
    vec = to_gf2(vector).reshape(-1, 1)
    aug = np.concatenate([to_gf2(matrix), vec], axis=1)
    return gf2_rank(aug)


def gf2_matmul(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    mat = to_gf2(matrix)
    vec = to_gf2(vector)
    return (mat @ vec) % 2


def gf2_matmul_batch(matrix: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    mat = to_gf2(matrix)
    vecs = to_gf2(vectors)
    return (vecs @ mat.T) % 2


@dataclass(frozen=True)
class RowReduceResult:
    matrix: np.ndarray
    rank: int
    pivots: Tuple[int, ...]


def gf2_row_reduce(matrix: np.ndarray) -> RowReduceResult:
    mat = to_gf2(matrix).copy()
    m, n = mat.shape
    pivots = []
    row = 0
    for col in range(n):
        pivot = None
        for r in range(row, m):
            if mat[r, col] == 1:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            mat[[row, pivot]] = mat[[pivot, row]]
        for r in range(m):
            if r != row and mat[r, col] == 1:
                mat[r, :] ^= mat[row, :]
        pivots.append(col)
        row += 1
        if row == m:
            break
    return RowReduceResult(matrix=mat, rank=len(pivots), pivots=tuple(pivots))


def gf2_nullspace_basis(matrix: np.ndarray) -> np.ndarray:
    """Return a basis for the nullspace of matrix over GF(2)."""
    reduced = gf2_row_reduce(matrix)
    mat = reduced.matrix
    m, n = mat.shape
    pivots = set(reduced.pivots)
    free_cols = [c for c in range(n) if c not in pivots]
    basis = []
    for free in free_cols:
        vec = np.zeros(n, dtype=np.uint8)
        vec[free] = 1
        for row, col in enumerate(reduced.pivots):
            if mat[row, free] == 1:
                vec[col] = 1
        basis.append(vec)
    if not basis:
        return np.zeros((0, n), dtype=np.uint8)
    return np.vstack(basis)


def gf2_is_consistent(matrix: np.ndarray, vector: np.ndarray) -> bool:
    return gf2_rank(matrix) == gf2_rank_augmented(matrix, vector)
