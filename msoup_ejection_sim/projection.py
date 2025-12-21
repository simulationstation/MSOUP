"""Projection utilities for expansion surrogates."""
from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def compute_H_global(cfg: SimulationConfig, X_V: float, mean_dm3: float, mean_rho_tot: float) -> float:
    H = cfg.H_base * (1.0 + cfg.beta * X_V)
    if mean_rho_tot > 0:
        H *= (1.0 - cfg.beta_dm * mean_dm3 / (mean_rho_tot + 1e-8))
    return H


def robust_mean(values, weights=None, clip_sigma: float = 3.0):
    arr = np.asarray(values)
    if weights is None:
        weights = np.ones_like(arr)
    weights = np.asarray(weights)
    avg = np.average(arr, weights=weights)
    std = np.sqrt(np.average((arr - avg) ** 2, weights=weights))
    if std == 0:
        return float(avg)
    mask = np.abs(arr - avg) < clip_sigma * std
    if not np.any(mask):
        return float(avg)
    return float(np.average(arr[mask], weights=weights[mask]))
