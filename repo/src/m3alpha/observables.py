"""Observables for the XY model."""
from __future__ import annotations

import numpy as np


def helicity_modulus(theta: np.ndarray, temperature: float, j_coupling: float = 1.0) -> float:
    """Compute the helicity modulus for a 2D XY configuration.

    Uses the standard estimator with periodic boundary conditions and averages
    over x and y directions.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if theta.ndim != 2:
        raise ValueError("theta must be a 2D array")

    l_size = theta.shape[0]
    if theta.shape[1] != l_size:
        raise ValueError("theta must be square")

    # x bonds
    theta_x = np.roll(theta, shift=-1, axis=1) - theta
    cos_x = np.cos(theta_x)
    sin_x = np.sin(theta_x)

    # y bonds
    theta_y = np.roll(theta, shift=-1, axis=0) - theta
    cos_y = np.cos(theta_y)
    sin_y = np.sin(theta_y)

    n_sites = l_size * l_size
    sum_cos = cos_x.sum() + cos_y.sum()
    sum_sin_x = sin_x.sum()
    sum_sin_y = sin_y.sum()

    term_cos = j_coupling * sum_cos / (2.0 * n_sites)
    term_sin = (j_coupling**2 / (2.0 * temperature * n_sites)) * (
        sum_sin_x**2 + sum_sin_y**2
    )
    return term_cos - term_sin
