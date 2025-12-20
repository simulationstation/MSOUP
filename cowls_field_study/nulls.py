"""Null distributions for field-level statistics."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from .stats_field import compute_field_stats
from .ring_profile import RingProfile


def circular_shift(profile: RingProfile, rng: np.random.Generator) -> RingProfile:
    """Generate a circularly-shifted null profile."""
    shift = int(rng.integers(0, len(profile.r_theta)))
    r_shift = np.roll(profile.r_theta, shift)
    return RingProfile(profile.theta_edges, r_shift, profile.uncertainty, profile.coverage, profile.weights)


def resample_window(
    profile: RingProfile,
    residual_samples: np.ndarray,
    rng: np.random.Generator,
) -> RingProfile:
    """
    Resample residuals into theta bins using window weights.

    ``residual_samples`` should reflect the one-point distribution of
    residual/noise inside the arc mask.
    """
    n_bins = len(profile.r_theta)
    residual_samples = np.asarray(residual_samples, dtype=float)
    if residual_samples.size == 0:
        residual_samples = np.array([0.0])

    weights = np.nan_to_num(profile.weights, nan=0.0)
    if not np.any(weights):
        weights = np.ones_like(weights)
    weights = weights / (np.sum(weights) + 1e-9)

    # Sample an effective number of pixels per bin proportional to S(theta)
    effective_counts = rng.poisson(lam=weights * max(1, residual_samples.size))
    r_new = np.zeros(n_bins, dtype=float)
    for i in range(n_bins):
        count = max(1, int(effective_counts[i]))
        draws = rng.choice(residual_samples, size=count, replace=True)
        r_new[i] = float(np.nanmedian(draws))
    return RingProfile(profile.theta_edges, r_new, profile.uncertainty, profile.coverage, profile.weights)


def draw_null_statistics(
    profile: RingProfile,
    mode: str,
    residual_samples: np.ndarray,
    lag_max: int,
    hf_fraction: float,
    draws: int = 300,
    seed: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate null realizations for correlation and power statistics.
    """
    rng = np.random.default_rng(seed)
    t_corr_null = np.zeros(draws, dtype=float)
    t_pow_null = np.zeros(draws, dtype=float)

    for i in range(draws):
        if mode == "shift":
            null_profile = circular_shift(profile, rng)
        elif mode == "both":
            if i % 2 == 0:
                null_profile = circular_shift(profile, rng)
            else:
                null_profile = resample_window(profile, residual_samples, rng)
        else:
            null_profile = resample_window(profile, residual_samples, rng)

        stats = compute_field_stats(null_profile, lag_max=lag_max, hf_fraction=hf_fraction)
        t_corr_null[i] = stats.t_corr
        t_pow_null[i] = stats.t_pow
    return t_corr_null, t_pow_null
