"""Coarse-graining utilities for constraint matrices."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constraints import ConstraintPair, ConstraintSystem
from .gf2 import gf2_rank


@dataclass(frozen=True)
class DeltaMCoarsegrainResult:
    block_size: int
    map_name: str
    delta_m: int


@dataclass(frozen=True)
class CoarseGrainResult:
    block_size: int
    n_vars: int
    delta_m: int


def block_projection(n: int, block_size: int) -> np.ndarray:
    n_blocks = int(np.ceil(n / block_size))
    proj = np.zeros((n, n_blocks), dtype=np.uint8)
    for idx in range(n):
        proj[idx, idx // block_size] = 1
    return proj


def randomized_projection(
    rng: np.random.Generator,
    n: int,
    n_blocks: int,
    density: float = 0.5,
) -> np.ndarray:
    proj = (rng.random((n, n_blocks)) < density).astype(np.uint8)
    for idx in range(n_blocks):
        proj[idx, idx] = 1
    return proj


def coarsegrain_system(system: ConstraintSystem, projection: np.ndarray) -> ConstraintSystem:
    matrix = (system.matrix @ projection) % 2
    return ConstraintSystem(matrix=matrix.astype(np.uint8), rhs=system.rhs.copy())


def coarsegrain_pair(pair: ConstraintPair, projection: np.ndarray) -> ConstraintPair:
    return ConstraintPair(
        peeled=coarsegrain_system(pair.peeled, projection),
        pinned=coarsegrain_system(pair.pinned, projection),
    )


def coarsegrain_blocks(pair: ConstraintPair, block_sizes: Iterable[int]) -> list[CoarseGrainResult]:
    results = []
    for block_size in block_sizes:
        projection = block_projection(pair.peeled.n_vars, block_size)
        coarse = coarsegrain_pair(pair, projection)
        results.append(
            CoarseGrainResult(
                block_size=block_size,
                n_vars=coarse.peeled.n_vars,
                delta_m=coarse.delta_m,
            )
        )
    return results


def coarsegrain_random(
    pair: ConstraintPair,
    block_sizes: Iterable[int],
    rng: np.random.Generator,
    density: float = 0.5,
) -> list[CoarseGrainResult]:
    results = []
    for block_size in block_sizes:
        n_blocks = int(np.ceil(pair.peeled.n_vars / block_size))
        projection = randomized_projection(rng, pair.peeled.n_vars, n_blocks, density=density)
        coarse = coarsegrain_pair(pair, projection)
        results.append(
            CoarseGrainResult(
                block_size=block_size,
                n_vars=coarse.peeled.n_vars,
                delta_m=coarse.delta_m,
            )
        )
    return results


def block_sum_projection(n: int, block_size: int) -> np.ndarray:
    return block_projection(n, block_size)


def seeded_random_projection(
    rng: np.random.Generator,
    n: int,
    n_blocks: int,
    density: float = 0.5,
) -> np.ndarray:
    return randomized_projection(rng, n, n_blocks, density=density)


def coarsegrain_delta_m(
    a_minus: np.ndarray,
    a_plus: np.ndarray,
    block_sizes: Iterable[int],
    rng: np.random.Generator,
    density: float = 0.5,
) -> list[DeltaMCoarsegrainResult]:
    results: list[DeltaMCoarsegrainResult] = []
    n_vars = a_minus.shape[1]
    for block_size in block_sizes:
        n_blocks = int(np.ceil(n_vars / block_size))
        proj_block = block_sum_projection(n_vars, block_size)
        proj_random = seeded_random_projection(rng, n_vars, n_blocks, density=density)
        for name, projection in ("block", proj_block), ("random", proj_random):
            minus_cg = (a_minus @ projection) % 2
            plus_cg = (a_plus @ projection) % 2
            delta_m = gf2_rank(plus_cg) - gf2_rank(minus_cg)
            results.append(
                DeltaMCoarsegrainResult(
                    block_size=block_size,
                    map_name=name,
                    delta_m=delta_m,
                )
            )
    return results
