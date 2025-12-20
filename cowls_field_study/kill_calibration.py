"""Calibration helpers for the COWLS kill pipeline."""

from __future__ import annotations

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .nulls import circular_shift, resample_window, draw_null_statistics, NullDraws
from .ring_profile import RingProfile
from .stats_field import compute_field_stats
from .highpass import HighpassConfig, mask_ring_profile


def _enforce_draw_floor(requested: int, default: int, allow_reduction: bool, name: str) -> int:
    """Ensure draw counts are not silently reduced."""

    if requested < default and not allow_reduction:
        raise ValueError(
            f"{name} requested {requested} draws but default minimum is {default}; "
            "pass allow_reduction=True to override."
        )
    return requested


def empirical_p_value(samples: Sequence[float], observed: float) -> float:
    """Compute the smoothed empirical p-value."""

    samples = np.asarray(samples, dtype=float)
    exceed = np.sum(samples >= observed)
    return float((exceed + 1) / (samples.size + 1)) if samples.size else math.nan


def compute_low_m_ratio(profile: RingProfile, m0: int = 3) -> float:
    """Return the low-m dominance ratio using weighted circular interpolation."""

    theta_edges = np.asarray(profile.theta_edges, dtype=float)
    values = np.asarray(profile.r_theta, dtype=float)
    weights = np.nan_to_num(np.asarray(profile.weights, dtype=float), nan=0.0)

    if values.size == 0 or theta_edges.size < 2:
        return math.nan

    theta_centers = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    valid = np.isfinite(values) & (weights > 0)

    if np.count_nonzero(valid) == 0:
        return math.nan

    filled = values.copy()
    if not np.all(valid):
        # Wrap the valid points to allow circular interpolation
        valid_thetas = theta_centers[valid]
        valid_vals = values[valid]

        order = np.argsort(valid_thetas)
        valid_thetas = valid_thetas[order]
        valid_vals = valid_vals[order]

        theta_wrap = np.concatenate([valid_thetas - 360, valid_thetas, valid_thetas + 360])
        vals_wrap = np.concatenate([valid_vals, valid_vals, valid_vals])
        interpolated = np.interp(theta_centers, theta_wrap, vals_wrap)
        filled = np.where(valid, filled, interpolated)

    weight_norm = weights if np.any(weights) else np.ones_like(filled)
    weight_norm = weight_norm / (np.sum(weight_norm) + 1e-9)
    weighted_series = filled * weight_norm

    fft = np.fft.rfft(weighted_series)
    power = np.abs(fft) ** 2

    if power.size <= 1:
        return math.nan

    total_power = np.sum(power[1:])
    max_low_m = min(m0, power.size - 1)
    low_m_power = np.sum(power[1 : max_low_m + 1])

    if total_power <= 0:
        return math.nan
    return float(low_m_power / (total_power + 1e-12))


@dataclass
class LensNullContext:
    """Reusable state for null draws for a single lens/band."""

    lens_id: str
    band: str
    profile: RingProfile
    residual_samples: np.ndarray
    null_means: Dict[str, float] = field(default_factory=dict)
    null_stds: Dict[str, float] = field(default_factory=dict)
    null_hp_means: Dict[str, float] = field(default_factory=dict)
    null_hp_stds: Dict[str, float] = field(default_factory=dict)
    valid_mask: np.ndarray | None = None


def prepare_null_baseline(
    profile: RingProfile,
    residual_samples: np.ndarray,
    method: str,
    lag_max: int,
    hf_fraction: float,
    draws: int,
    default_draws: int,
    allow_reduction: bool,
    highpass_config: HighpassConfig | None = None,
    valid_mask: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[NullDraws, float, float, float | None, float | None]:
    """Compute the per-lens baseline null distribution."""

    draws = _enforce_draw_floor(draws, default_draws, allow_reduction, f"{method} per-lens null")
    nulls = draw_null_statistics(
        profile,
        mode=method,
        residual_samples=residual_samples,
        lag_max=lag_max,
        hf_fraction=hf_fraction,
        draws=draws,
        highpass_config=highpass_config,
        valid_mask=valid_mask,
        seed=seed,
    )
    mean = float(np.nanmean(nulls.t_corr)) if nulls.t_corr.size else 0.0
    std = float(np.nanstd(nulls.t_corr) + 1e-9) if nulls.t_corr.size else 1.0
    mean_hp = float(np.nanmean(nulls.t_corr_hp)) if nulls.t_corr_hp is not None and nulls.t_corr_hp.size else None
    std_hp = float(np.nanstd(nulls.t_corr_hp) + 1e-9) if nulls.t_corr_hp is not None and nulls.t_corr_hp.size else None
    return nulls, mean, std, mean_hp, std_hp


def _single_global_null_draw(
    contexts: Iterable[LensNullContext],
    method: str,
    lag_max: int,
    hf_fraction: float,
    highpass_config: HighpassConfig | None,
    seed: int | None,
) -> tuple[float, float | None]:
    rng = np.random.default_rng(seed)
    z_values: List[float] = []
    z_hp_values: List[float] = []

    for ctx in contexts:
        lens_rng = np.random.default_rng(rng.integers(0, 2**32 - 1))
        if method == "shift":
            null_profile = circular_shift(ctx.profile, lens_rng)
        else:
            null_profile = resample_window(ctx.profile, ctx.residual_samples, lens_rng)

        if ctx.valid_mask is not None:
            null_profile = mask_ring_profile(null_profile, ctx.valid_mask)

        stats = compute_field_stats(
            null_profile,
            lag_max=lag_max,
            hf_fraction=hf_fraction,
            highpass_config=highpass_config,
            highpass_mask=ctx.valid_mask,
        )
        mean = ctx.null_means[method]
        std = ctx.null_stds[method]
        z_values.append((stats.t_corr - mean) / (std + 1e-9))
        if ctx.null_hp_means.get(method) is not None and stats.t_corr_hp is not None:
            z_hp_values.append((stats.t_corr_hp - ctx.null_hp_means[method]) / (ctx.null_hp_stds[method] + 1e-9))

    z_values = np.asarray(z_values, dtype=float)
    base = float(np.nanmean(z_values) * math.sqrt(z_values.size)) if z_values.size else float("nan")

    z_hp_values = np.asarray(z_hp_values, dtype=float)
    hp_val = float(np.nanmean(z_hp_values) * math.sqrt(z_hp_values.size)) if z_hp_values.size else float("nan")
    return base, hp_val


def build_global_null_distribution(
    contexts: Sequence[LensNullContext],
    method: str,
    lag_max: int,
    hf_fraction: float,
    draws: int,
    default_draws: int,
    allow_reduction: bool,
    max_workers: int | None = None,
    seed: int | None = None,
    highpass_config: HighpassConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate global-null draws of Z_corr across lenses."""

    draws = _enforce_draw_floor(draws, default_draws, allow_reduction, f"{method} global-null")
    seeds = np.random.default_rng(seed).integers(0, 2**32 - 1, size=draws)
    worker = lambda s: _single_global_null_draw(contexts, method, lag_max, hf_fraction, highpass_config, int(s))

    if max_workers is None:
        max_workers = os.cpu_count() or 1

    hp_samples: list[float] = []
    base_samples: list[float] = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            for base, hp in executor.map(worker, seeds):
                base_samples.append(base)
                hp_samples.append(hp)
    else:
        for s in seeds:
            base, hp = worker(int(s))
            base_samples.append(base)
            hp_samples.append(hp)

    return np.asarray(base_samples, dtype=float), np.asarray(hp_samples, dtype=float)
