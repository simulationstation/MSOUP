"""BAO template models."""

from __future__ import annotations

import numpy as np


def _transfer_no_wiggle(k: np.ndarray, omega_m: float, h: float) -> np.ndarray:
    """Eisenstein & Hu (1998) no-wiggle transfer function."""
    theta = 2.7255 / 2.7
    omhh = omega_m * h**2
    k_eq = 0.0746 * omhh / theta**2
    q = k / (13.41 * k_eq)
    l0 = np.log(np.e + 1.8 * q)
    c0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    return l0 / (l0 + c0 * q**2)


def bao_template(
    s: np.ndarray,
    r_d: float,
    sigma_nl: float,
    omega_m: float,
    omega_b: float,
    h: float,
    n_s: float,
    k_min: float = 1.0e-3,
    k_max: float = 0.6,
    n_k: int = 512,
) -> np.ndarray:
    """Compute a standard BAO template from a damped linear power spectrum."""
    k = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    t0 = _transfer_no_wiggle(k, omega_m=omega_m, h=h)
    f_b = omega_b / omega_m
    wiggle = 1.0 + f_b * np.exp(-(k * sigma_nl) ** 2) * np.sin(k * r_d) / np.maximum(k * r_d, 1.0e-8)
    pk = k**n_s * (t0**2) * wiggle

    ks = np.outer(k, s)
    j0 = np.sinc(ks / np.pi)
    integrand = k[:, None] ** 2 * pk[:, None] * j0
    xi = np.trapz(integrand, k, axis=0) / (2.0 * np.pi**2)
    return xi


def model_xi(
    s: np.ndarray,
    alpha: float,
    bias: float,
    r_d: float,
    sigma_nl: float,
    omega_m: float,
    omega_b: float,
    h: float,
    n_s: float,
    nuisance: np.ndarray,
) -> np.ndarray:
    template = bao_template(
        alpha * s,
        r_d=r_d,
        sigma_nl=sigma_nl,
        omega_m=omega_m,
        omega_b=omega_b,
        h=h,
        n_s=n_s,
    )
    return bias**2 * template + nuisance
