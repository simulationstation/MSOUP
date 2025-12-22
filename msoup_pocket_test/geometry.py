"""
Propagation geometry consistency tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd

from .config import GeometryConfig


@dataclass
class GeometryResult:
    speed_kmps: float
    normal: np.ndarray
    p_value: float
    n_sensors: int


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


def evaluate_geometry(
    candidate_frame: pd.DataFrame,
    positions: pd.DataFrame,
    cfg: GeometryConfig,
) -> GeometryResult:
    """
    Evaluate propagation geometry by comparing fitted plane speed/normal against
    shuffled arrival times.
    """
    df = candidate_frame
    if df.empty or len(df["sensor"].unique()) < cfg.min_sensors:
        return GeometryResult(speed_kmps=np.nan, normal=np.zeros(3), p_value=1.0, n_sensors=len(df["sensor"].unique()))

    # Use first arrival per sensor
    first_hits = df.sort_values("peak_time").groupby("sensor").head(1)
    sensor_positions = positions.set_index("satellite").reindex(first_hits["sensor"])
    valid = sensor_positions.dropna()
    first_hits = first_hits.set_index("sensor").loc[valid.index].reset_index()
    if len(first_hits) < cfg.min_sensors:
        return GeometryResult(speed_kmps=np.nan, normal=np.zeros(3), p_value=1.0, n_sensors=len(first_hits))

    normal, speed = _fit_plane_times(valid.reset_index(), first_hits["peak_time"])
    observed = speed

    rng = np.random.default_rng(cfg.n_shuffles + cfg.min_sensors)
    shuffled_speeds = []
    for _ in range(cfg.n_shuffles):
        shuffled_times = first_hits["peak_time"].sample(frac=1.0, replace=False, random_state=rng.integers(0, 1e9))
        _, shuffle_speed = _fit_plane_times(valid.reset_index(), shuffled_times.reset_index(drop=True))
        shuffled_speeds.append(shuffle_speed)

    p_value = (np.sum(np.array(shuffled_speeds) >= observed) + 1) / (len(shuffled_speeds) + 1)
    return GeometryResult(speed_kmps=observed, normal=normal, p_value=p_value, n_sensors=len(first_hits))
