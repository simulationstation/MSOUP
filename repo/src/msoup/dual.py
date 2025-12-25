"""Dual-variable / maximum entropy utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .constraints import ConstraintSystem
from .gf2 import gf2_matmul_batch


@dataclass(frozen=True)
class DualEstimate:
    free_energy_over_t: float
    mean_violations: float
    mu_over_t: float


def _enumerate_assignments(n: int) -> np.ndarray:
    values = np.arange(2 ** n, dtype=np.uint64)
    bits = ((values[:, None] >> np.arange(n)) & 1).astype(np.uint8)
    return bits


def _log_mean_exp(values: np.ndarray) -> float:
    vmax = np.max(values)
    return vmax + np.log(np.mean(np.exp(values - vmax)))


def dual_estimate_exact(system: ConstraintSystem, j_over_t: float) -> DualEstimate:
    assignments = _enumerate_assignments(system.n_vars)
    violations = gf2_matmul_batch(system.matrix, assignments) ^ system.rhs
    v_count = violations.sum(axis=1)
    log_weights = -j_over_t * v_count
    z_log = _log_mean_exp(log_weights) + np.log(len(v_count))
    f_over_t = -z_log
    weights = np.exp(log_weights - np.max(log_weights))
    mean_v = (weights * v_count).sum() / weights.sum()
    mu_over_t = f_over_t / system.rank if system.rank != 0 else 0.0
    return DualEstimate(free_energy_over_t=f_over_t, mean_violations=mean_v, mu_over_t=mu_over_t)


def estimate_mu_from_ensembles(
    peeled: ConstraintSystem,
    pinned: ConstraintSystem,
    j_over_t: float,
    rng: np.random.Generator,
    n_samples: int = 2048,
) -> DualEstimate:
    n = peeled.n_vars
    samples = rng.integers(0, 2, size=(n_samples, n), dtype=np.uint8)
    v_peel = gf2_matmul_batch(peeled.matrix, samples) ^ peeled.rhs
    v_pin = gf2_matmul_batch(pinned.matrix, samples) ^ pinned.rhs
    e_peel = v_peel.sum(axis=1)
    e_pin = v_pin.sum(axis=1)

    log_weights_peel = -j_over_t * e_peel
    log_weights_pin = -j_over_t * e_pin
    log_mean_peel = _log_mean_exp(log_weights_peel)
    log_mean_pin = _log_mean_exp(log_weights_pin)
    f_peel = -(log_mean_peel + n * np.log(2))
    f_pin = -(log_mean_pin + n * np.log(2))

    weights_pin = np.exp(log_weights_pin - np.max(log_weights_pin))
    mean_v = (weights_pin * e_pin).sum() / weights_pin.sum()

    delta_m = pinned.rank - peeled.rank
    mu_over_t = (f_pin - f_peel) / delta_m if delta_m != 0 else 0.0
    return DualEstimate(free_energy_over_t=f_pin, mean_violations=mean_v, mu_over_t=mu_over_t)
