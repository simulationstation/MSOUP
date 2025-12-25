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


def _tophat_window(x: np.ndarray) -> np.ndarray:
    """Spherical top-hat window function in Fourier space."""
    x = np.asarray(x)
    window = np.ones_like(x)
    mask = x != 0.0
    x_m = x[mask]
    window[mask] = 3.0 * (np.sin(x_m) - x_m * np.cos(x_m)) / x_m**3
    return window


def _sigma_r(k: np.ndarray, pk: np.ndarray, radius: float) -> float:
    """Return sigma(R) for a power spectrum P(k) and radius in Mpc/h."""
    kr = k * radius
    window = _tophat_window(kr)
    integrand = k**2 * pk * window**2
    sigma2 = np.trapz(integrand, k) / (2.0 * np.pi**2)
    return float(np.sqrt(max(sigma2, 0.0)))


def bao_template(
    s: np.ndarray,
    r_d: float,
    sigma_nl: float,
    omega_m: float,
    omega_b: float,
    h: float,
    n_s: float,
    sigma8: float,
    k_min: float = 1.0e-3,
    k_max: float = 10.0,
    n_k: int = 1024,
) -> np.ndarray:
    """Compute a standard BAO template from a damped linear power spectrum."""
    k = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    t0 = _transfer_no_wiggle(k, omega_m=omega_m, h=h)
    f_b = omega_b / omega_m
    r_d_h = r_d * h
    wiggle = 1.0 + f_b * np.exp(-(k * sigma_nl) ** 2) * np.sin(k * r_d_h) / np.maximum(k * r_d_h, 1.0e-8)
    pk_shape = k**n_s * (t0**2) * wiggle
    sigma8_unscaled = _sigma_r(k, pk_shape, radius=8.0)
    if sigma8_unscaled > 0:
        pk = pk_shape * (sigma8 / sigma8_unscaled) ** 2
    else:
        pk = pk_shape

    ks = np.outer(k, s)
    j0 = np.sinc(ks / np.pi)
    integrand = k[:, None] ** 2 * pk[:, None] * j0
    xi = np.trapz(integrand, k, axis=0) / (2.0 * np.pi**2)
    return xi


def bao_template_damped(
    s: np.ndarray,
    r_d: float,
    sigma_nl: float,
    omega_m: float,
    omega_b: float,
    h: float,
    n_s: float,
    sigma8: float,
    k_min: float = 1.0e-3,
    k_max: float = 10.0,
    n_k: int = 1024,
) -> np.ndarray:
    """Compute BAO template using damped wiggle/no-wiggle decomposition."""
    k = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
    t0 = _transfer_no_wiggle(k, omega_m=omega_m, h=h)
    f_b = omega_b / omega_m
    r_d_h = r_d * h

    p_nowiggle = k**n_s * (t0**2)
    wiggle = 1.0 + f_b * np.sin(k * r_d_h) / np.maximum(k * r_d_h, 1.0e-8)
    p_lin = p_nowiggle * wiggle
    p_wiggle = p_lin - p_nowiggle

    sigma8_unscaled = _sigma_r(k, p_lin, radius=8.0)
    if sigma8_unscaled > 0:
        scale = (sigma8 / sigma8_unscaled) ** 2
        p_nowiggle = p_nowiggle * scale
        p_wiggle = p_wiggle * scale

    damping = np.exp(-0.5 * (k * sigma_nl) ** 2)
    pk = p_nowiggle + p_wiggle * damping

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
    sigma8: float,
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
        sigma8=sigma8,
    )
    return bias**2 * template + nuisance
