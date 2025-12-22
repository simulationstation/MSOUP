"""
Statistics for the MV-O1B pocket test.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .config import CandidateConfig, StatsConfig
from .nulls import iter_null_catalogs


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
    p_clustering_two_sided: float

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
    Compute pair-count clustering metric using KDTree for O(n log n) performance.

    Uses spatial indexing to find time-neighbors efficiently, then filters by
    angle constraint if present. Produces mathematically identical results to
    brute-force O(n²) approach.
    """
    if df.empty:
        return np.nan

    n = len(df)
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return np.nan

    # Convert times to seconds (float array for KDTree)
    # Handle both tz-aware and tz-naive datetimes
    times = pd.to_datetime(df["peak_time"], utc=True).astype("int64") / 1e9
    times_arr = times.to_numpy().reshape(-1, 1)

    # Build KDTree on times - O(n log n)
    tree = cKDTree(times_arr)

    # Find all pairs within time threshold - O(n log n + k) where k = matching pairs
    time_neighbor_pairs = tree.query_pairs(r=neighbor_time_s, output_type='set')

    if neighbor_angle_deg is None or "angle" not in df.columns:
        # No angle constraint - time-neighbor count is the answer
        count = len(time_neighbor_pairs)
    else:
        # Filter by angle constraint - O(k) where k << n²
        angles = df["angle"].to_numpy()
        count = 0
        for i, j in time_neighbor_pairs:
            if abs(angles[i] - angles[j]) <= neighbor_angle_deg:
                count += 1

    return count / total_pairs


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


def _compute_null_stats(cat: pd.DataFrame, cand_cfg: CandidateConfig) -> Tuple[float, float]:
    """Compute Fano and clustering for a single null catalog (for parallel execution)."""
    if cat.empty:
        return np.nan, np.nan
    null_counts = cat.groupby("sensor").size()
    fano = compute_fano(null_counts)
    clust = compute_clustering(cat, cand_cfg)
    return fano, clust


def _get_memory_usage_mb() -> float:
    """Get current process memory usage in MB."""
    try:
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _get_available_memory_mb() -> float:
    """Get available system memory in MB."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 * 1024)
    except ImportError:
        return float('inf')


def compute_debiased_stats(
    candidates: pd.DataFrame,
    windows: pd.DataFrame,
    stats_cfg: StatsConfig,
    cand_cfg: CandidateConfig,
    memory_limit_mb: float = 0,
) -> GlobalStats:
    """
    Compute C_excess and Fano with null debiasing.

    Uses streaming to process one null catalog at a time, avoiding memory explosion.

    Args:
        memory_limit_mb: If > 0, abort if process memory exceeds this limit.
                        Default 0 means no limit (but will warn at 80% system RAM).
    """
    if candidates.empty:
        return GlobalStats(
            fano_obs=0.0, fano_null_mean=1.0, fano_null_std=0.0, fano_z=0.0, p_fano=1.0,
            c_obs=0.0, c_null_mean=0.0, c_null_std=0.0, c_excess=0.0, c_excess_std=0.0,
            c_z=0.0, p_clustering=1.0, p_clustering_two_sided=1.0, n_resamples=0, sensor_weights={},
        )

    counts = candidates.groupby("sensor").size()
    fano_obs = float(np.nan_to_num(compute_fano(counts), nan=0.0, posinf=0.0, neginf=0.0))
    c_obs = float(np.nan_to_num(compute_clustering(candidates, cand_cfg), nan=0.0, posinf=0.0, neginf=0.0))

    # Stream null catalogs one at a time (memory efficient)
    fano_null = []
    c_null = []
    n_processed = 0

    # Get initial memory state for monitoring
    initial_mem = _get_memory_usage_mb()
    try:
        import psutil
        total_mem = psutil.virtual_memory().total / (1024 * 1024)
        mem_warning_threshold = total_mem * 0.8
    except ImportError:
        mem_warning_threshold = float('inf')

    null_iter = iter_null_catalogs(
        candidates,
        windows,
        stats_cfg.null_realizations,
        stats_cfg.block_length,
        stats_cfg.allow_time_shift_null,
        stats_cfg.random_seed,
        show_progress=True,
        null_type=getattr(stats_cfg, "null_type", None),
    )

    for cat in null_iter:
        if cat.empty:
            continue

        # Compute stats for this single catalog
        null_counts = cat.groupby("sensor").size()
        fano_val = compute_fano(null_counts)
        clust_val = compute_clustering(cat, cand_cfg)
        if np.isnan(fano_val) or np.isnan(clust_val):
            continue
        fano_null.append(fano_val)
        c_null.append(clust_val)
        n_processed += 1

        # Memory monitoring (check every 10 iterations to reduce overhead)
        if n_processed % 10 == 0:
            current_mem = _get_memory_usage_mb()
            if memory_limit_mb > 0 and current_mem > memory_limit_mb:
                print(f"\n⚠️  Memory limit reached ({current_mem:.0f} MB > {memory_limit_mb:.0f} MB)")
                print(f"    Stopping early after {n_processed} null realizations")
                break
            if current_mem > mem_warning_threshold:
                print(f"\n⚠️  High memory usage: {current_mem:.0f} MB (80% of system RAM)")
                print(f"    Consider reducing null_realizations or data size")

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
    # two-sided approximation from Z (Gaussian tail)
    p_clustering_two_sided = float(min(1.0, np.math.erfc(abs(c_z) / np.sqrt(2))))

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
        p_clustering_two_sided=p_clustering_two_sided,
        n_resamples=n_processed,
        sensor_weights=weights,
    )
