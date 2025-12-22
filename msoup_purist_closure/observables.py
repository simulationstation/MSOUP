from __future__ import annotations

import numpy as np

from .distances import angular_diameter_distance, comoving_distance, luminosity_distance


def distance_modulus(z: np.ndarray, delta_m: float, nuisance_M: float = 0.0, **kwargs) -> np.ndarray:
    """Compute distance modulus mu."""
    dl = luminosity_distance(z, delta_m=delta_m, **kwargs)  # km/s units cancel via c/H
    mu = 5.0 * np.log10(np.maximum(dl, 1e-9)) + 25.0 + nuisance_M
    return mu


def bao_dv(z: np.ndarray, delta_m: float, **kwargs) -> np.ndarray:
    """
    Compute BAO volume distance D_V in Mpc.

    D_V(z) = [ (1+z)^2 * D_A(z)^2 * c*z / H(z) ]^(1/3)
    """
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
    c_km_s = kwargs.get("c_km_s", 299792.458)
    dv = ((1 + z) ** 2 * da ** 2 * (c_km_s * z / h_eff_vals)) ** (1.0 / 3.0)
    return dv


def bao_dm(z: np.ndarray, delta_m: float, **kwargs) -> np.ndarray:
    """
    Compute BAO comoving distance D_M in Mpc.

    D_M(z) = (1+z) * D_A(z)
    """
    z = np.asarray(z, dtype=float)
    da = angular_diameter_distance(z, delta_m=delta_m, **kwargs)
    return (1 + z) * da


def bao_dh(z: np.ndarray, delta_m: float, **kwargs) -> np.ndarray:
    """
    Compute BAO Hubble distance D_H in Mpc.

    D_H(z) = c / H(z)
    where c is in km/s and H(z) is in km/s/Mpc, so D_H is in Mpc.
    """
    z = np.asarray(z, dtype=float)
    from .model import h_eff

    h_eff_vals = h_eff(
        z,
        delta_m=delta_m,
        h_early=kwargs.get("h_early"),
        omega_m0=kwargs.get("omega_m0"),
        omega_L0=kwargs.get("omega_L0"),
    )
    c_km_s = kwargs.get("c_km_s", 299792.458)
    return c_km_s / h_eff_vals


def bao_predict(z: float, observable: str, delta_m: float, rd_mpc: float, **kwargs) -> float:
    """
    Compute BAO observable prediction for a single redshift.

    Args:
        z: Redshift
        observable: One of 'DV/rd', 'DM/rd', 'DH/rd' (case-insensitive)
        delta_m: MSOUP delta_m parameter
        rd_mpc: Sound horizon at drag epoch in Mpc (fixed, not fitted)
        **kwargs: Passed to distance functions (h_early, omega_m0, omega_L0, c_km_s)

    Returns:
        Predicted observable value (dimensionless ratio)

    Raises:
        ValueError: If observable type is not recognized
    """
    obs_upper = observable.upper().replace(" ", "")
    z_arr = np.array([z])

    if obs_upper in ("DV/RD", "DV_RD", "DVRD"):
        dv = bao_dv(z_arr, delta_m=delta_m, **kwargs)[0]
        return dv / rd_mpc
    elif obs_upper in ("DM/RD", "DM_RD", "DMRD"):
        dm = bao_dm(z_arr, delta_m=delta_m, **kwargs)[0]
        return dm / rd_mpc
    elif obs_upper in ("DH/RD", "DH_RD", "DHRD"):
        dh = bao_dh(z_arr, delta_m=delta_m, **kwargs)[0]
        return dh / rd_mpc
    else:
        raise ValueError(f"Unknown BAO observable: '{observable}'. Supported: DV/rd, DM/rd, DH/rd")


def lens_time_delay_scaling(z_lens: float, z_source: float, delta_m: float, **kwargs) -> float:
    """Simplified lensing geometry scaling using angular diameter distances."""
    da_l = angular_diameter_distance(np.array([z_lens]), delta_m=delta_m, **kwargs)[0]
    da_s = angular_diameter_distance(np.array([z_source]), delta_m=delta_m, **kwargs)[0]
    da_ls = comoving_distance(np.array([z_source]), delta_m=delta_m, **kwargs)[0] - comoving_distance(np.array([z_lens]), delta_m=delta_m, **kwargs)[0]
    if da_l <= 0 or da_s <= 0 or da_ls <= 0:
        raise ValueError("Distances must be positive")
    return da_l * da_ls / da_s
