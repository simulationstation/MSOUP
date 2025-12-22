"""
Per-sensor sensitivity window construction.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .config import PreprocessConfig, WindowConfig


@dataclass
class Window:
    sensor: str
    start: pd.Timestamp
    end: pd.Timestamp
    cadence_seconds: float
    quality_score: float

    def duration_seconds(self) -> float:
        return (self.end - self.start).total_seconds()


def _split_by_gaps(df: pd.DataFrame, cadence_seconds: float, gap_factor: float) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    times = df["time"].reset_index(drop=True)
    diffs = times.diff().dt.total_seconds().fillna(0)
    gap = cadence_seconds * gap_factor
    boundaries = np.where(diffs > gap)[0]
    segments = []
    start_idx = 0
    for b in boundaries:
        segments.append((times.iloc[start_idx], times.iloc[b - 1]))
        start_idx = b
    segments.append((times.iloc[start_idx], times.iloc[len(times) - 1]))
    return segments


def derive_windows(
    per_sensor: Dict[str, pd.DataFrame],
    preprocess_cfg: PreprocessConfig,
    window_cfg: WindowConfig,
) -> List[Window]:
    """
    Build per-sensor sensitivity windows accounting for cadence gaps and quality.
    """
    windows: List[Window] = []
    for sensor, df in per_sensor.items():
        if df.empty:
            continue
        segments = _split_by_gaps(df, preprocess_cfg.cadence_seconds, window_cfg.cadence_tolerance)
        for start, end in segments:
            duration = (end - start).total_seconds()
            if duration < window_cfg.require_fractional_coverage * preprocess_cfg.min_window_minutes * 60:
                continue
            quality_score = min(1.0, duration / (preprocess_cfg.min_window_minutes * 60))
            windows.append(
                Window(
                    sensor=sensor,
                    start=start,
                    end=end,
                    cadence_seconds=preprocess_cfg.cadence_seconds,
                    quality_score=quality_score,
                )
            )
    return windows


def windows_to_frame(windows: Iterable[Window]) -> pd.DataFrame:
    """Convert list of Window dataclasses to DataFrame."""
    return pd.DataFrame([dataclasses.asdict(w) for w in windows])


def filter_windows(windows: List[Window], min_quality: float) -> List[Window]:
    """Retain windows exceeding a quality threshold."""
    return [w for w in windows if w.quality_score >= min_quality]


def window_index_for_time(windows: List[Window], sensor: str, time: pd.Timestamp) -> int:
    """Return index of window containing time, or -1."""
    for i, w in enumerate(windows):
        if w.sensor == sensor and w.start <= time <= w.end:
            return i
    return -1
