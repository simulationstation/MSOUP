"""
Propagation geometry consistency tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .config import GeometryConfig


@dataclass
class GeometryResult:
    """Container for geometry test results."""
    chi2_obs: float
    null_chi2s: np.ndarray  # Array of chi2 from shuffled nulls
    p_value: float
    n_events: int
    speed_kmps: float = 0.0
    normal: np.ndarray = None


def _fit_plane_times(positions: pd.DataFrame, times: pd.Series) -> Tuple[np.ndarray, float]:
    """
    Fit a plane wavefront (n · r = v t) to sensor positions and arrival times.
    Returns normal vector (unit) and speed (km/s).
    """
    r = positions[["x", "y", "z"]].to_numpy() / 1000.0  # m -> km
    t = (times - times.min()).dt.total_seconds().to_numpy()
    A = np.column_stack([r, -t])
    # Solve A [n_x, n_y, n_z, v]^T = 0 with constraint ||n||=1
    _, _, vh = np.linalg.svd(A)
    params = vh[-1, :]
    normal = params[:3]
    speed = params[3]
    norm = np.linalg.norm(normal) or 1.0
    normal /= norm
    speed = abs(speed)
    return normal, speed


def _compute_chi2(positions: pd.DataFrame, times: pd.Series, normal: np.ndarray, speed: float) -> float:
    """Compute chi-squared residual for plane wave fit."""
    if positions.empty or len(times) == 0:
        return np.nan
    r = positions[["x", "y", "z"]].to_numpy() / 1000.0  # m -> km
    t = (times - times.min()).dt.total_seconds().to_numpy()
    # Predicted arrival times from plane wave: t_pred = (n · r) / v
    if speed == 0:
        return np.nan
    t_pred = np.dot(r, normal) / speed
    t_pred = t_pred - t_pred.min()  # Normalize to start at 0
    residuals = t - t_pred
    chi2 = np.sum(residuals ** 2)
    return float(chi2)


def evaluate_geometry(
    candidate_frame: pd.DataFrame,
    positions: pd.DataFrame,
    cfg: GeometryConfig,
) -> GeometryResult:
    """
    Evaluate propagation geometry by comparing fitted plane speed/normal against
    shuffled arrival times. Uses chi-squared as the test statistic.
    """
    df = candidate_frame
    n_unique = len(df["sensor"].unique()) if not df.empty else 0

    if df.empty or n_unique < cfg.min_sensors:
        return GeometryResult(
            chi2_obs=np.nan, null_chi2s=None, p_value=1.0, n_events=0,
            speed_kmps=np.nan, normal=np.zeros(3)
        )

    # Use first arrival per sensor
    # Normalize timestamps to UTC for consistent comparison
    df = df.copy()
    df["peak_time"] = pd.to_datetime(df["peak_time"], utc=True)
    first_hits = df.sort_values("peak_time").groupby("sensor").head(1)
    # Use first position per satellite (positions may have multiple epochs)
    unique_positions = positions.drop_duplicates(subset="satellite").set_index("satellite")
    sensor_positions = unique_positions.reindex(first_hits["sensor"])
    valid = sensor_positions.dropna()
    first_hits = first_hits.set_index("sensor").loc[valid.index].reset_index()

    if len(first_hits) < cfg.min_sensors:
        return GeometryResult(
            chi2_obs=np.nan, null_chi2s=None, p_value=1.0, n_events=len(first_hits),
            speed_kmps=np.nan, normal=np.zeros(3)
        )

    # Fit plane wave to observed data
    normal, speed = _fit_plane_times(valid.reset_index(), first_hits["peak_time"])
    chi2_obs = _compute_chi2(valid.reset_index(), first_hits["peak_time"], normal, speed)

    # Generate null distribution by shuffling arrival times
    rng = np.random.default_rng(cfg.n_shuffles + cfg.min_sensors)
    null_chi2s = []
    for _ in tqdm(range(cfg.n_shuffles), desc="Geometry shuffles", leave=False):
        shuffled_times = first_hits["peak_time"].sample(frac=1.0, replace=False, random_state=rng.integers(0, 1e9))
        shuffled_times = shuffled_times.reset_index(drop=True)
        null_normal, null_speed = _fit_plane_times(valid.reset_index(), shuffled_times)
        null_chi2 = _compute_chi2(valid.reset_index(), shuffled_times, null_normal, null_speed)
        null_chi2s.append(null_chi2)

    null_chi2s = np.array(null_chi2s)
    # P-value: fraction of null chi2 values <= observed (lower chi2 = better fit)
    p_value = (np.sum(null_chi2s <= chi2_obs) + 1) / (len(null_chi2s) + 1)

    return GeometryResult(
        chi2_obs=chi2_obs,
        null_chi2s=null_chi2s,
        p_value=p_value,
        n_events=len(first_hits),
        speed_kmps=speed,
        normal=normal
    )
