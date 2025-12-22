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
    fano_factor: float
    c_obs: float
    c_excess: float
    p_value: float
    sensor_weights: Dict[str, float]
    null_mean: float
    null_std: float


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
    counts = candidates.groupby("sensor").size()
    fano = compute_fano(counts)

    c_obs = compute_clustering(candidates, cand_cfg)

    null_catalogs = generate_null_catalogs(
        candidates,
        windows,
        stats_cfg.null_realizations,
        stats_cfg.block_length,
        stats_cfg.allow_time_shift_null,
        stats_cfg.random_seed,
    )
    c_null = [compute_clustering(cat, cand_cfg) for cat in null_catalogs if not cat.empty]
    null_mean = float(np.nanmean(c_null)) if c_null else 0.0
    null_std = float(np.nanstd(c_null)) if c_null else 0.0
    c_excess = c_obs - null_mean
    p_value = compute_empirical_pvalue(c_obs, c_null)

    # Weight sensors inversely by their counts to avoid dominance
    weights = {sensor: 1.0 / (count if count else 1) for sensor, count in counts.items()}
    total = sum(weights.values()) or 1.0
    weights = {k: v / total for k, v in weights.items()}

    return GlobalStats(
        fano_factor=fano,
        c_obs=c_obs,
        c_excess=c_excess,
        p_value=p_value,
        sensor_weights=weights,
        null_mean=null_mean,
        null_std=null_std,
    )
