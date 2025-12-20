"""Control experiments and falsifiers."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np

from .nulls import draw_null_statistics
from .stats_field import FieldStats, aggregate_z_scores


def shuffle_control(stats: FieldStats, draws: int, lag_max: int, hf_fraction: float, seed: int = 0) -> Tuple[float, float]:
    """
    Replace the observed profile with a null draw and verify Z ≈ 0.
    """
    rng = np.random.default_rng(seed)
    fake_profile = stats.r_theta.copy()
    rng.shuffle(fake_profile)
    residual_samples = fake_profile
    from .ring_profile import RingProfile

    null_profile = RingProfile(stats.theta_edges, fake_profile, stats.weights, stats.weights, stats.weights)
    nulls = draw_null_statistics(
        null_profile, mode="resample", residual_samples=residual_samples, lag_max=lag_max, hf_fraction=hf_fraction, draws=draws, seed=seed
    )
    z_corr, _, _ = aggregate_z_scores([0.0], nulls.t_corr)
    z_pow, _, _ = aggregate_z_scores([0.0], nulls.t_pow)
    return z_corr, z_pow


def window_stress_test(stats: FieldStats, sharpen_factor: float = 1.5) -> FieldStats:
    """
    Amplify the window weights to ensure statistics respond smoothly.
    """
    stressed = stats.weights * sharpen_factor
    stressed = stressed / (np.nanmean(stressed) + 1e-9)
    stressed_profile = stats.r_theta * stressed
    return FieldStats(
        t_corr=stats.t_corr,
        t_pow=stats.t_pow,
        r_theta=stressed_profile,
        weights=stressed,
        theta_edges=stats.theta_edges,
    )


def band_robustness(z_scores: Dict[str, Tuple[float, float]]) -> float:
    """
    Check variability of Z across bands; small values indicate robustness.
    """
    if not z_scores:
        return 0.0
    vals = [abs(z[0]) + abs(z[1]) for z in z_scores.values()]
    return float(np.nanstd(vals))
