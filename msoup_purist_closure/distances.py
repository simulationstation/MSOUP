from __future__ import annotations

import functools
import numpy as np
from scipy import integrate

from .config import DEFAULT_H_EARLY, DEFAULT_OMEGA_L0, DEFAULT_OMEGA_M0
from .model import h_eff, leak_fraction


def comoving_distance(
    z: np.ndarray,
    delta_m: float,
    h_early: float = DEFAULT_H_EARLY,
    omega_m0: float = DEFAULT_OMEGA_M0,
    omega_L0: float = DEFAULT_OMEGA_L0,
    c_km_s: float = 299792.458,
    num_points: int | None = None,
) -> np.ndarray:
    """Compute comoving distance via quadrature."""
    z = np.asarray(z, dtype=float)
    num_points = num_points or max(128, len(z) * 4)
    z_grid = np.linspace(0.0, z.max(), num_points)
    hz = h_eff(z_grid, delta_m=delta_m, h_early=h_early, omega_m0=omega_m0, omega_L0=omega_L0)
    integral = integrate.cumulative_trapezoid(1.0 / hz, z_grid, initial=0.0)
    interp = np.interp(z, z_grid, integral)
    return c_km_s * interp


def luminosity_distance(z: np.ndarray, *args, **kwargs) -> np.ndarray:
    chi = comoving_distance(z, *args, **kwargs)
    return (1.0 + np.asarray(z, dtype=float)) * chi


def angular_diameter_distance(z: np.ndarray, *args, **kwargs) -> np.ndarray:
    chi = comoving_distance(z, *args, **kwargs)
    z = np.asarray(z, dtype=float)
    return chi / (1.0 + z)


def h_eff_ratio(z: np.ndarray, delta_m: float, h_early: float = DEFAULT_H_EARLY, omega_m0: float = DEFAULT_OMEGA_M0, omega_L0: float = DEFAULT_OMEGA_L0) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    lcdm = h_early * np.sqrt(omega_m0 * (1.0 + z) ** 3 + omega_L0)
    return h_eff(z, delta_m, h_early, omega_m0, omega_L0) / lcdm
