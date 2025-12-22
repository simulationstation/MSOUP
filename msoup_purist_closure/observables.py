from __future__ import annotations

import numpy as np

from .distances import angular_diameter_distance, comoving_distance, luminosity_distance


def distance_modulus(z: np.ndarray, delta_m: float, nuisance_M: float = 0.0, **kwargs) -> np.ndarray:
    """Compute distance modulus mu."""
    dl = luminosity_distance(z, delta_m=delta_m, **kwargs)  # km/s units cancel via c/H
    mu = 5.0 * np.log10(np.maximum(dl, 1e-9)) + 25.0 + nuisance_M
    return mu


def bao_dv(z: np.ndarray, delta_m: float, **kwargs) -> np.ndarray:
    """Compute BAO volume distance D_V."""
    z = np.asarray(z, dtype=float)
    da = angular_diameter_distance(z, delta_m=delta_m, **kwargs)
    from .model import h_eff

    h_eff_vals = h_eff(
        z,
        delta_m=delta_m,
        h_early=kwargs.get("h_early"),
        omega_m0=kwargs.get("omega_m0"),
        omega_L0=kwargs.get("omega_L0"),
    )
    dv = ((1 + z) ** 2 * da ** 2 * (kwargs.get("c_km_s", 299792.458) * z / h_eff_vals)) ** (1.0 / 3.0)
    return dv


def lens_time_delay_scaling(z_lens: float, z_source: float, delta_m: float, **kwargs) -> float:
    """Simplified lensing geometry scaling using angular diameter distances."""
    da_l = angular_diameter_distance(np.array([z_lens]), delta_m=delta_m, **kwargs)[0]
    da_s = angular_diameter_distance(np.array([z_source]), delta_m=delta_m, **kwargs)[0]
    da_ls = comoving_distance(np.array([z_source]), delta_m=delta_m, **kwargs)[0] - comoving_distance(np.array([z_lens]), delta_m=delta_m, **kwargs)[0]
    if da_l <= 0 or da_s <= 0 or da_ls <= 0:
        raise ValueError("Distances must be positive")
    return da_l * da_ls / da_s
