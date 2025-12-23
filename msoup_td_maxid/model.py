from __future__ import annotations

import numpy as np

from msoup_purist_closure.distances import angular_diameter_distance, comoving_distance
from msoup_purist_closure.model import leak_fraction


def D_dt_pred(z_lens: float, z_source: float, delta_m: float, h_early: float, omega_m0: float, omega_L0: float) -> float:
    """Predict time-delay distance (D_dt)."""
    d_l = angular_diameter_distance(np.array([z_lens]), delta_m=delta_m, h_early=h_early, omega_m0=omega_m0, omega_L0=omega_L0)[0]
    d_s = angular_diameter_distance(np.array([z_source]), delta_m=delta_m, h_early=h_early, omega_m0=omega_m0, omega_L0=omega_L0)[0]
    chi_l = comoving_distance(np.array([z_lens]), delta_m=delta_m, h_early=h_early, omega_m0=omega_m0, omega_L0=omega_L0)[0]
    chi_s = comoving_distance(np.array([z_source]), delta_m=delta_m, h_early=h_early, omega_m0=omega_m0, omega_L0=omega_L0)[0]
    d_ls = (chi_s - chi_l) / (1 + z_source)
    if d_ls <= 0:
        return np.nan
    return (1 + z_lens) * d_l * d_s / d_ls


def f0_from_delta_m(delta_m: np.ndarray | float, omega_m0: float, omega_L0: float) -> np.ndarray:
    dm_arr = np.asarray(delta_m, dtype=float).reshape(-1)
    z0 = np.array([0.0])
    f_list = [float(leak_fraction(z0, float(dm), omega_m0, omega_L0)[0]) for dm in dm_arr]
    return np.asarray(f_list, dtype=float).reshape(np.shape(delta_m))
