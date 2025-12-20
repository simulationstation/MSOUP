"""Null distributions for field-level statistics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .stats_field import compute_field_stats
from .ring_profile import RingProfile
from .highpass import HighpassConfig, mask_ring_profile


@dataclass
class NullDraws:
    t_corr: np.ndarray
    t_pow: np.ndarray
    t_corr_hp: np.ndarray | None = None
    t_pow_hp: np.ndarray | None = None


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
    highpass_config: HighpassConfig | None = None,
    valid_mask: np.ndarray | None = None,
) -> NullDraws:
    """
    Generate null realizations for correlation and power statistics.
    """
    rng = np.random.default_rng(seed)
    t_corr_null = np.zeros(draws, dtype=float)
    t_pow_null = np.zeros(draws, dtype=float)
    t_corr_hp_null = np.zeros(draws, dtype=float)
    t_pow_hp_null = np.zeros(draws, dtype=float)

    # use provided masks/configs

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

        if valid_mask is not None:
            null_profile = mask_ring_profile(null_profile, valid_mask)

        stats = compute_field_stats(
            null_profile,
            lag_max=lag_max,
            hf_fraction=hf_fraction,
            highpass_config=highpass_config,
            highpass_mask=valid_mask,
        )
        t_corr_null[i] = stats.t_corr
        t_pow_null[i] = stats.t_pow
        if stats.t_corr_hp is not None:
            t_corr_hp_null[i] = stats.t_corr_hp
            t_pow_hp_null[i] = stats.t_pow_hp if stats.t_pow_hp is not None else 0.0
    return NullDraws(
        t_corr=t_corr_null,
        t_pow=t_pow_null,
        t_corr_hp=t_corr_hp_null if highpass_config is not None else None,
        t_pow_hp=t_pow_hp_null if highpass_config is not None else None,
    )
