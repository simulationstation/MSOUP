"""Neutrality checks for MSoup/Mverse constraint models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .constraints import ConstraintPair
from .dual import DualEstimate, estimate_mu_from_ensembles
from .gf2 import gf2_matmul_batch, gf2_rank


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


@dataclass(frozen=True)
class MuOverTResult:
    j_over_t: float
    mu_over_t: float
    log_z_minus: float
    log_z_plus: float


def _log_mean_exp(values: np.ndarray) -> float:
    max_val = np.max(values)
    return float(max_val + np.log(np.mean(np.exp(values - max_val))))


def estimate_mu_over_t_mc(
    a_minus: np.ndarray,
    b_minus: np.ndarray,
    a_plus: np.ndarray,
    b_plus: np.ndarray,
    j_over_t: float,
    samples: np.ndarray,
    delta_m: int | None = None,
) -> MuOverTResult:
    """Estimate μ/T using Monte Carlo samples of assignments."""
    n_vars = samples.shape[1]
    if delta_m is None:
        delta_m = gf2_rank(a_plus) - gf2_rank(a_minus)
    if delta_m <= 0:
        raise ValueError("delta_m must be positive")

    def _log_z(a: np.ndarray, b: np.ndarray) -> float:
        parity = gf2_matmul_batch(a, samples)
        violations = np.sum(parity != b, axis=1)
        log_weights = -j_over_t * violations.astype(np.float64)
        return n_vars * np.log(2.0) + _log_mean_exp(log_weights)

    log_z_minus = _log_z(a_minus, b_minus)
    log_z_plus = _log_z(a_plus, b_plus)
    mu_over_t = (log_z_plus - log_z_minus) / delta_m
    return MuOverTResult(
        j_over_t=j_over_t,
        mu_over_t=mu_over_t,
        log_z_minus=log_z_minus,
        log_z_plus=log_z_plus,
    )
