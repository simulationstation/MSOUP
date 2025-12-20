"""High-pass filtering utilities for ring profiles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .ring_profile import RingProfile


@dataclass
class HighpassConfig:
    """Configuration for high-pass filtering."""

    m_cut: int = 3
    m_max: int | None = None
    method: str = "wls_fourier"


def _design_matrix(theta: np.ndarray, m_max: int) -> np.ndarray:
    """Return the Fourier design matrix up to order ``m_max``."""

    cols = [np.ones_like(theta)]
    for m in range(1, m_max + 1):
        cols.append(np.cos(m * theta))
        cols.append(np.sin(m * theta))
    return np.vstack(cols).T


def _evaluate_fourier(theta: np.ndarray, coeffs: np.ndarray, m_limit: int) -> np.ndarray:
    """Evaluate the fitted Fourier series up to ``m_limit``."""

    values = np.full_like(theta, coeffs[0] if coeffs.size else 0.0, dtype=float)
    cos_coeffs = coeffs[1::2]
    sin_coeffs = coeffs[2::2]
    max_use = min(m_limit, len(cos_coeffs))
    for idx in range(max_use):
        m = idx + 1
        values += cos_coeffs[idx] * np.cos(m * theta)
        if idx < len(sin_coeffs):
            values += sin_coeffs[idx] * np.sin(m * theta)
    return values


def mask_ring_profile(profile: RingProfile, valid_mask: np.ndarray) -> RingProfile:
    """Return a copy of ``profile`` with invalid bins blanked to NaN/zero."""

    mask = np.asarray(valid_mask, dtype=bool)
    r_theta = np.where(mask, profile.r_theta, np.nan)
    weights = np.where(mask, profile.weights, 0.0)
    coverage = np.where(mask, profile.coverage, 0.0)
    return RingProfile(
        theta_edges=profile.theta_edges,
        r_theta=r_theta,
        uncertainty=profile.uncertainty,
        coverage=coverage,
        weights=weights,
    )


def highpass_fourier(
    r_theta: np.ndarray,
    valid_mask: np.ndarray | None = None,
    m_cut: int = 3,
    theta: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    m_max: int | None = None,
    method: str = "wls_fourier",
) -> Tuple[np.ndarray, Dict]:
    """
    Apply a weighted Fourier high-pass filter to ``r_theta``.

    The low-order Fourier modes up to ``m_cut`` are estimated via weighted
    least squares using only valid bins, then subtracted to yield the
    high-pass component.
    """

    values = np.asarray(r_theta, dtype=float)
    n = values.size
    if theta is None:
        theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    theta = np.asarray(theta, dtype=float)

    if valid_mask is None:
        valid_mask = np.isfinite(values)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool)
    valid_mask = valid_mask & np.isfinite(values)

    if weights is None:
        weights = np.ones_like(values)
    weights = np.nan_to_num(np.asarray(weights, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    weights = np.where(valid_mask, weights, 0.0)

    n_valid = int(np.count_nonzero(valid_mask))
    if n_valid == 0:
        return np.full_like(values, np.nan), {
            "m_max": 0,
            "m_cut": m_cut,
            "coefficients": None,
            "fitted": np.full_like(values, np.nan),
            "low_mode": np.full_like(values, np.nan),
            "valid_fraction": 0.0,
        }

    if m_max is None:
        m_max = min(20, max(1, n_valid // 4))
    m_max = max(m_cut, m_max)

    if method != "wls_fourier":
        raise ValueError(f"Unknown high-pass method {method}")

    design = _design_matrix(theta, m_max)
    design_valid = design[valid_mask]
    weights_valid = weights[valid_mask]
    values_valid = values[valid_mask]

    w_sqrt = np.sqrt(np.clip(weights_valid, 0.0, None))
    a = design_valid * w_sqrt[:, None]
    b = values_valid * w_sqrt
    coeffs, *_ = np.linalg.lstsq(a, b, rcond=None)

    fitted = design @ coeffs
    low_mode = _evaluate_fourier(theta, coeffs, m_cut)
    r_hp = values - low_mode
    r_hp = np.where(valid_mask, r_hp, np.nan)

    diagnostics = {
        "m_max": m_max,
        "m_cut": m_cut,
        "coefficients": coeffs,
        "fitted": fitted,
        "low_mode": low_mode,
        "valid_fraction": n_valid / max(1, n),
        "weights": weights,
    }
    return r_hp, diagnostics


def apply_highpass_filter(
    profile: RingProfile,
    config: HighpassConfig,
    valid_mask: np.ndarray | None = None,
    weight_override: np.ndarray | None = None,
) -> Tuple[RingProfile, Dict]:
    """Apply :func:`highpass_fourier` to a :class:`RingProfile`."""

    theta_edges = np.asarray(profile.theta_edges, dtype=float)
    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    theta_rad = np.deg2rad(theta_centers)

    weights = weight_override if weight_override is not None else profile.weights
    mask = valid_mask
    if mask is None:
        mask = np.isfinite(profile.r_theta) & (np.nan_to_num(weights, nan=0.0) >= 0)

    r_hp, diagnostics = highpass_fourier(
        profile.r_theta,
        valid_mask=mask,
        m_cut=config.m_cut,
        theta=theta_rad,
        weights=weights,
        m_max=config.m_max,
        method=config.method,
    )
    filtered_profile = mask_ring_profile(
        RingProfile(
            theta_edges=profile.theta_edges,
            r_theta=r_hp,
            uncertainty=profile.uncertainty,
            coverage=profile.coverage,
            weights=profile.weights,
        ),
        mask,
    )
    diagnostics["theta"] = theta_rad
    return filtered_profile, diagnostics
