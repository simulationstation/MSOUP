"""Field-level statistics on ring residuals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np

from .ring_profile import RingProfile
from .highpass import HighpassConfig, apply_highpass_filter


@dataclass
class FieldStats:
    t_corr: float
    t_pow: float
    r_theta: np.ndarray
    weights: np.ndarray
    theta_edges: np.ndarray
    t_corr_hp: float | None = None
    t_pow_hp: float | None = None
    r_theta_hp: np.ndarray | None = None


def _normalized_autocorr(series: np.ndarray, max_lag: int) -> Tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(series)
    series = np.nan_to_num(series, nan=0.0)
    if np.all(series == 0) or np.count_nonzero(valid) == 0:
        return np.zeros(max_lag), np.zeros(max_lag)

    series = series - np.nanmean(series[valid])
    denom = np.nanvar(series[valid]) + 1e-9
    ac = []
    for lag in range(1, max_lag + 1):
        shifted = np.roll(series, lag)
        mask = valid & np.roll(valid, lag)
        if not np.any(mask):
            ac.append(0.0)
            continue
        ac_val = float(np.sum(series[mask] * shifted[mask]) / (np.sum(mask) * denom))
        ac.append(ac_val)
    return np.array(ac), np.arange(1, max_lag + 1)


def _high_frequency_power(series: np.ndarray, weights: np.ndarray, fraction: float) -> float:
    series = np.nan_to_num(series, nan=0.0)
    weights = np.nan_to_num(weights, nan=0.0)
    if series.size == 0:
        return 0.0
    if not np.any(weights):
        weights = np.ones_like(series)
    series = series - np.average(series, weights=weights)
    padded = series * weights
    fft = np.fft.rfft(padded)
    power = np.abs(fft) ** 2
    high_start = int((1.0 - fraction) * power.size)
    high_start = max(1, min(power.size - 1, high_start))
    return float(np.sum(power[high_start:])) if high_start < power.size else 0.0


def compute_field_stats(
    profile: RingProfile,
    lag_max: int,
    hf_fraction: float,
    highpass_config: HighpassConfig | None = None,
    highpass_mask: np.ndarray | None = None,
) -> FieldStats:
    """Compute autocorrelation- and power-based test statistics."""
    ac, _ = _normalized_autocorr(profile.r_theta, lag_max)
    t_corr = float(np.nanmean(ac))
    t_pow = _high_frequency_power(profile.r_theta, profile.weights, hf_fraction)
    field_stats = FieldStats(
        t_corr=t_corr,
        t_pow=t_pow,
        r_theta=profile.r_theta,
        weights=profile.weights,
        theta_edges=profile.theta_edges,
    )

    if highpass_config is not None:
        hp_profile, _ = apply_highpass_filter(profile, highpass_config, valid_mask=highpass_mask)
        ac_hp, _ = _normalized_autocorr(hp_profile.r_theta, lag_max)
        t_corr_hp = float(np.nanmean(ac_hp))
        t_pow_hp = _high_frequency_power(hp_profile.r_theta, hp_profile.weights, hf_fraction)
        field_stats.t_corr_hp = t_corr_hp
        field_stats.t_pow_hp = t_pow_hp
        field_stats.r_theta_hp = hp_profile.r_theta
    return field_stats


def zscore(value: float, null_values: Iterable[float]) -> Tuple[float, float, float]:
    """Return Z = (value - mean(null)) / std(null)."""
    nulls = np.asarray(list(null_values), dtype=float)
    mean_null = float(np.nanmean(nulls)) if nulls.size else 0.0
    std_null = float(np.nanstd(nulls) + 1e-9) if nulls.size else 1.0
    z = float((value - mean_null) / std_null)
    return z, mean_null, std_null


def aggregate_z_scores(values: Iterable[float], null_values: Iterable[float]) -> Tuple[float, float, float]:
    """
    Compute Z-score for a statistic relative to null distribution.

    Returns z, mean_null, std_null.
    """
    vals = np.asarray(list(values), dtype=float)
    nulls = np.asarray(list(null_values), dtype=float)
    mean_null = float(np.nanmean(nulls)) if nulls.size else 0.0
    std_null = float(np.nanstd(nulls) + 1e-9) if nulls.size else 1.0
    z = float((np.nanmean(vals) - mean_null) / std_null)
    return z, mean_null, std_null
