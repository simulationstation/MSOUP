from __future__ import annotations

import numpy as np

from .config import DEFAULT_H_EARLY, DEFAULT_OMEGA_L0, DEFAULT_OMEGA_M0


def pinned_fraction(z: np.ndarray, delta_m: float) -> np.ndarray:
    """
    Compute pinned fraction A(z).

    The shape is governed by redshift only; delta_m is accepted for API
    compatibility but does not change the baseline curve. This makes the
    leak-free limit explicit (see leak_fraction).
    """
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(np.log(2.0) * (1.0 - (1.0 + z) ** 3)))


def omega_m_z(z: np.ndarray, omega_m0: float = DEFAULT_OMEGA_M0, omega_L0: float = DEFAULT_OMEGA_L0) -> np.ndarray:
    """Matter density fraction as a function of redshift."""
    z = np.asarray(z, dtype=float)
    numerator = omega_m0 * (1.0 + z) ** 3
    return numerator / (numerator + omega_L0)


def leak_fraction(z: np.ndarray, delta_m: float, omega_m0: float = DEFAULT_OMEGA_M0, omega_L0: float = DEFAULT_OMEGA_L0) -> np.ndarray:
    """
    Leak fraction f(z).

    By construction f(z, delta_m=0) == 0 for all z, so the LCDM baseline
    is free of artificial leakage. delta_m linearly controls the amplitude
    while the pinned_fraction shape sets the redshift dependence.
    """
    z = np.asarray(z, dtype=float)
    if float(delta_m) == 0.0:
        return np.zeros_like(z, dtype=float)

    base = (1.0 - pinned_fraction(z, delta_m)) * omega_m_z(z, omega_m0, omega_L0)
    return float(delta_m) * base


def h_eff(z: np.ndarray, delta_m: float, h_early: float = DEFAULT_H_EARLY, omega_m0: float = DEFAULT_OMEGA_M0, omega_L0: float = DEFAULT_OMEGA_L0) -> np.ndarray:
    """Effective Hubble parameter H_eff(z)."""
    z = np.asarray(z, dtype=float)
    lcdm = h_early * np.sqrt(omega_m0 * (1.0 + z) ** 3 + omega_L0)
    f = leak_fraction(z, delta_m, omega_m0, omega_L0)
    # Guard against numerical issues if |f| approaches 1
    f = np.clip(f, -0.999999999999, 0.999999999999)
    return lcdm / np.sqrt(1.0 - f)


def h_inferred_from_kernel(f_eff: float, h_early: float = DEFAULT_H_EARLY) -> float:
    """Infer H from effective leak fraction."""
    return h_early / np.sqrt(max(1e-15, 1.0 - f_eff))
