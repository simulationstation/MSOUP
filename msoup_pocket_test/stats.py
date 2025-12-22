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
from .resources import ResourceMonitor


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
    pvalue_resolution: float
    sensor_weights: Dict[str, float]
    peak_rss_gb: float


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


def _binned_clustering(
    df: pd.DataFrame,
    neighbor_time_s: float,
    neighbor_angle_deg: float,
    time_bin_seconds: float,
) -> float:
    """Approximate clustering using coarse bins for near O(n) scaling."""
    if df.empty:
        return np.nan

    n = len(df)
    total_pairs = n * (n - 1) // 2
    if total_pairs == 0:
        return np.nan

    times = pd.to_datetime(df["peak_time"], utc=True).astype("int64") / 1e9
    t_bins = np.floor(times / time_bin_seconds).astype(np.int64)

    if neighbor_angle_deg is None or "angle" not in df.columns:
        counts: Dict[int, int] = {}
        for t in t_bins:
            counts[t] = counts.get(t, 0) + 1
        t_radius = max(0, int(np.ceil(neighbor_time_s / time_bin_seconds)))
        pair_count = 0.0
        for t, c in counts.items():
            pair_count += c * (c - 1) / 2
            for dt in range(1, t_radius + 1):
                if t + dt in counts:
                    pair_count += c * counts[t + dt]
        return pair_count / total_pairs

    angle_bin_size = max(1.0, neighbor_angle_deg)
    a_bins = np.floor(df["angle"].to_numpy() / angle_bin_size).astype(np.int64)
    counts: Dict[Tuple[int, int], int] = {}
    for t, a in zip(t_bins, a_bins):
        key = (t, a)
        counts[key] = counts.get(key, 0) + 1

    t_radius = max(0, int(np.ceil(neighbor_time_s / time_bin_seconds)))
    a_radius = max(0, int(np.ceil(neighbor_angle_deg / angle_bin_size)))
    pair_count = 0.0
    for (t, a), c in counts.items():
        pair_count += c * (c - 1) / 2
        for dt in range(0, t_radius + 1):
            for da in range(-a_radius, a_radius + 1):
                if dt == 0 and da <= 0:
                    continue
                neighbor = (t + dt, a + da)
                if neighbor in counts:
                    pair_count += c * counts[neighbor]
    return pair_count / total_pairs


def compute_clustering(
    candidates: pd.DataFrame,
    config: CandidateConfig,
    pair_mode: str = "binned",
    time_bin_seconds: float = 60.0,
    max_candidates_in_memory: int = 750000,
    unsafe: bool = False,
) -> float:
    """Compute observed clustering metric."""
    if len(candidates) > max_candidates_in_memory and pair_mode == "exact" and not unsafe:
        raise RuntimeError(
            f"{len(candidates)} candidates exceed exact-pair threshold {max_candidates_in_memory}; "
            "use binned pair mode or increase the limit with --unsafe."
        )
    angle = config.neighbor_angle_degrees if "angle" in candidates else None
    if pair_mode == "exact":
        return _pairwise_clustering(
            candidates,
            neighbor_time_s=config.neighbor_time_seconds,
            neighbor_angle_deg=angle,
        )
    return _binned_clustering(
        candidates,
        neighbor_time_s=config.neighbor_time_seconds,
        neighbor_angle_deg=angle,
        time_bin_seconds=time_bin_seconds,
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
    monitor: ResourceMonitor | None = None,
    unsafe: bool = False,
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
            fano_obs=0.0,
            fano_null_mean=1.0,
            fano_null_std=0.0,
            fano_z=0.0,
            p_fano=1.0,
            c_obs=0.0,
            c_null_mean=0.0,
            c_null_std=0.0,
            c_excess=0.0,
            c_excess_std=0.0,
            c_z=0.0,
            p_clustering=1.0,
            p_clustering_two_sided=1.0,
            n_resamples=0,
            pvalue_resolution=1.0,
            sensor_weights={},
            peak_rss_gb=monitor.peak_rss_gb if monitor else 0.0,
        )

    counts = candidates.groupby("sensor").size()
    fano_obs = float(np.nan_to_num(compute_fano(counts), nan=0.0, posinf=0.0, neginf=0.0))
    c_obs = float(
        np.nan_to_num(
            compute_clustering(
                candidates,
                cand_cfg,
                pair_mode=getattr(stats_cfg, "pair_mode", "binned"),
                time_bin_seconds=getattr(stats_cfg, "time_bin_seconds", 60),
                max_candidates_in_memory=getattr(stats_cfg, "max_candidates_in_memory", 750000),
                unsafe=unsafe,
            ),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
    )

    if monitor:
        monitor.check("after_observed")

    class RunningStats:
        def __init__(self) -> None:
            self.n = 0
            self.mean = 0.0
            self.m2 = 0.0

        def update(self, value: float) -> None:
            self.n += 1
            delta = value - self.mean
            self.mean += delta / self.n
            delta2 = value - self.mean
            self.m2 += delta * delta2

        @property
        def variance(self) -> float:
            return self.m2 / self.n if self.n > 0 else 0.0

        @property
        def std(self) -> float:
            return np.sqrt(self.variance)

    fano_stats = RunningStats()
    c_stats = RunningStats()
    fano_null_values: List[float] = []
    c_null_values: List[float] = []
    n_processed = 0

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

        null_counts = cat.groupby("sensor").size()
        fano_val = compute_fano(null_counts)
        clust_val = compute_clustering(
            cat,
            cand_cfg,
            pair_mode=getattr(stats_cfg, "pair_mode", "binned"),
            time_bin_seconds=getattr(stats_cfg, "time_bin_seconds", 60),
            max_candidates_in_memory=getattr(stats_cfg, "max_candidates_in_memory", 750000),
            unsafe=unsafe,
        )
        if np.isnan(fano_val) or np.isnan(clust_val):
            continue
        fano_stats.update(fano_val)
        c_stats.update(clust_val)
        fano_null_values.append(fano_val)
        c_null_values.append(clust_val)
        n_processed += 1

        if monitor and n_processed % 5 == 0:
            monitor.check("null_resample")

    fano_null_mean = fano_stats.mean if fano_stats.n > 0 else 1.0
    fano_null_std = fano_stats.std if fano_stats.n > 0 else 0.0
    fano_z = (fano_obs - fano_null_mean) / fano_null_std if fano_null_std > 0 else 0.0

    more_extreme_fano = sum(1 for v in fano_null_values if v >= fano_obs) if fano_null_values else 0
    p_fano = float((more_extreme_fano + 1) / (len(fano_null_values) + 1)) if fano_null_values else 1.0

    c_null_mean = c_stats.mean if c_stats.n > 0 else 0.0
    c_null_std = c_stats.std if c_stats.n > 0 else 0.0
    c_excess = c_obs - c_null_mean
    c_excess_std = c_null_std
    c_z = c_excess / c_null_std if c_null_std > 0 else 0.0
    more_extreme = sum(1 for v in c_null_values if v >= c_obs)
    p_clustering = float((more_extreme + 1) / (len(c_null_values) + 1)) if c_null_values else 1.0
    two_sided_extreme = sum(1 for v in c_null_values if abs(v - c_null_mean) >= abs(c_obs - c_null_mean))
    p_clustering_two_sided = (
        float((two_sided_extreme + 1) / (len(c_null_values) + 1)) if c_null_values else 1.0
    )

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
        pvalue_resolution=1.0 / (n_processed + 1) if n_processed else 1.0,
        sensor_weights=weights,
        peak_rss_gb=monitor.peak_rss_gb if monitor else 0.0,
    )
