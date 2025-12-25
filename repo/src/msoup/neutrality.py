"""Neutrality checks for MSoup/Mverse constraint models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constraints import ConstraintPair
from .dual import DualEstimate, estimate_mu_from_ensembles


@dataclass(frozen=True)
class NeutralityResult:
    j_over_t: float
    mu_over_t: float
    delta_m: int
    delta_e_over_t: float
    neutrality_error: float


def estimate_mu_over_t(
    pair: ConstraintPair,
    j_over_t: float,
    rng: np.random.Generator,
    n_samples: int,
) -> DualEstimate:
    return estimate_mu_from_ensembles(
        peeled=pair.peeled,
        pinned=pair.pinned,
        j_over_t=j_over_t,
        rng=rng,
        n_samples=n_samples,
    )


def neutrality_checks(
    pairs: Iterable[ConstraintPair],
    j_over_t: float,
    rng: np.random.Generator,
    n_samples: int,
) -> list[NeutralityResult]:
    results = []
    for pair in pairs:
        estimate = estimate_mu_over_t(pair, j_over_t, rng, n_samples)
        delta_m = pair.delta_m
        delta_e_over_t = -estimate.mu_over_t * delta_m
        neutrality_error = delta_e_over_t + delta_m * np.log(2)
        results.append(
            NeutralityResult(
                j_over_t=j_over_t,
                mu_over_t=estimate.mu_over_t,
                delta_m=delta_m,
                delta_e_over_t=delta_e_over_t,
                neutrality_error=neutrality_error,
            )
        )
    return results
