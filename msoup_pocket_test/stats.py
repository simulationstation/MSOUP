"""
Statistics for the MV-O1B pocket test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import CandidateConfig, StatsConfig
from .nulls import generate_null_catalogs


@dataclass
class GlobalStats:
    """Container for all computed statistics."""
    # Fano factor statistics
    fano_obs: float
    fano_null_mean: float
    fano_null_std: float
    fano_z: float
    p_fano: float

    # Clustering statistics
    c_obs: float
    c_null_mean: float
    c_null_std: float
    c_excess: float
    c_excess_std: float
    c_z: float
    p_clustering: float

    # Metadata
    n_resamples: int
    sensor_weights: Dict[str, float]


def compute_fano(counts: Iterable[int]) -> float:
    """Compute Fano factor (variance/mean)."""
    counts = np.asarray(list(counts))
    if counts.size == 0:
        return np.nan
    mean = counts.mean()
    var = counts.var(ddof=1) if counts.size > 1 else 0.0
    if mean == 0:
        return np.nan
    return float(var / mean)


def _pairwise_clustering(df: pd.DataFrame, neighbor_time_s: float, neighbor_angle_deg: float) -> float:
    """
    Compute pair-count clustering metric across sensors with optional angular
    dependence (stored in df['angle'] if available).
    """
    if df.empty:
        return np.nan
    times = pd.to_datetime(df["peak_time"]).astype("int64") / 1e9
    angles = df.get("angle", pd.Series(np.zeros(len(df))))
    count = 0
    pairs = 0
    n = len(df)
    for i in range(n):
        for j in range(i + 1, n):
            dt = abs(times[i] - times[j])
            if dt > neighbor_time_s:
                continue
            if neighbor_angle_deg is not None and "angle" in df:
                if abs(angles.iloc[i] - angles.iloc[j]) > neighbor_angle_deg:
                    continue
            count += 1
        pairs += n - i - 1
    norm = pairs if pairs > 0 else 1
    return count / norm


def compute_clustering(
    candidates: pd.DataFrame,
    config: CandidateConfig,
) -> float:
    """Compute observed clustering metric."""
    return _pairwise_clustering(
        candidates,
        neighbor_time_s=config.neighbor_time_seconds,
        neighbor_angle_deg=config.neighbor_angle_degrees if "angle" in candidates else None,
    )


def compute_empirical_pvalue(observed: float, null_samples: List[float]) -> float:
    """Empirical p-value with add-one smoothing."""
    if not null_samples:
        return 1.0
    null_arr = np.asarray(null_samples)
    more_extreme = np.sum(null_arr >= observed)
    return float((more_extreme + 1) / (len(null_samples) + 1))


def compute_debiased_stats(
    candidates: pd.DataFrame,
    windows: pd.DataFrame,
    stats_cfg: StatsConfig,
    cand_cfg: CandidateConfig,
) -> GlobalStats:
    """Compute C_excess and Fano with null debiasing."""
    if candidates.empty:
        return GlobalStats(
            fano_obs=np.nan, fano_null_mean=1.0, fano_null_std=0.0, fano_z=0.0, p_fano=1.0,
            c_obs=0.0, c_null_mean=0.0, c_null_std=0.0, c_excess=0.0, c_excess_std=0.0,
            c_z=0.0, p_clustering=1.0, n_resamples=0, sensor_weights={},
        )

    counts = candidates.groupby("sensor").size()
    fano_obs = compute_fano(counts)
    c_obs = compute_clustering(candidates, cand_cfg)

    # Generate null catalogs
    null_catalogs = generate_null_catalogs(
        candidates,
        windows,
        stats_cfg.null_realizations,
        stats_cfg.block_length,
        stats_cfg.allow_time_shift_null,
        stats_cfg.random_seed,
    )

    # Compute Fano and clustering for each null
    fano_null = []
    c_null = []
    for cat in null_catalogs:
        if cat.empty:
            continue
        null_counts = cat.groupby("sensor").size()
        fano_null.append(compute_fano(null_counts))
        c_null.append(compute_clustering(cat, cand_cfg))

    # Fano statistics
    fano_null_mean = float(np.nanmean(fano_null)) if fano_null else 1.0
    fano_null_std = float(np.nanstd(fano_null)) if fano_null else 0.0
    fano_z = (fano_obs - fano_null_mean) / fano_null_std if fano_null_std > 0 else 0.0
    p_fano = compute_empirical_pvalue(fano_obs, fano_null)

    # Clustering statistics
    c_null_mean = float(np.nanmean(c_null)) if c_null else 0.0
    c_null_std = float(np.nanstd(c_null)) if c_null else 0.0
    c_excess = c_obs - c_null_mean
    c_excess_std = c_null_std  # Uncertainty from null distribution
    c_z = c_excess / c_null_std if c_null_std > 0 else 0.0
    p_clustering = compute_empirical_pvalue(c_obs, c_null)

    # Weight sensors inversely by their counts to avoid dominance
    weights = {sensor: 1.0 / (count if count else 1) for sensor, count in counts.items()}
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}

    return GlobalStats(
        fano_obs=fano_obs,
        fano_null_mean=fano_null_mean,
        fano_null_std=fano_null_std,
        fano_z=fano_z,
        p_fano=p_fano,
        c_obs=c_obs,
        c_null_mean=c_null_mean,
        c_null_std=c_null_std,
        c_excess=c_excess,
        c_excess_std=c_excess_std,
        c_z=c_z,
        p_clustering=p_clustering,
        n_resamples=len(null_catalogs),
        sensor_weights=weights,
    )
