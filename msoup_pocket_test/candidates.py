"""
Candidate extraction across sensors.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from .config import CandidateConfig
from .windows import Window, window_index_for_time


@dataclass
class Candidate:
    sensor: str
    start: pd.Timestamp
    end: pd.Timestamp
    peak_time: pd.Timestamp
    peak_value: float
    sigma: float
    is_subthreshold: bool
    window_index: int


def _detect_events(series: pd.DataFrame, config: CandidateConfig) -> List[Candidate]:
    values = series["value"]
    if values.empty:
        return []

    mean = values.mean()
    std = values.std(ddof=1) or 1.0
    threshold = mean + config.threshold_sigma * std
    subthreshold = mean + config.subthreshold_sigma * std

    events: List[Candidate] = []
    active = False
    start_idx = 0
    for idx, val in enumerate(values):
        if val >= subthreshold and not active:
            active = True
            start_idx = idx
        if active and val < subthreshold:
            active = False
            end_idx = idx
            segment = series.iloc[start_idx:end_idx]
            peak_idx = segment["value"].idxmax()
            peak_value = segment.loc[peak_idx, "value"]
            is_sub = peak_value < threshold
            events.append(
                Candidate(
                    sensor=series["sensor"].iloc[0],
                    start=segment["time"].iloc[0],
                    end=segment["time"].iloc[-1],
                    peak_time=segment.loc[peak_idx, "time"],
                    peak_value=float(peak_value),
                    sigma=(peak_value - mean) / std,
                    is_subthreshold=is_sub,
                    window_index=-1,
                )
            )

    # Handle trailing active segment
    if active:
        segment = series.iloc[start_idx:]
        peak_idx = segment["value"].idxmax()
        peak_value = segment.loc[peak_idx, "value"]
        is_sub = peak_value < threshold
        events.append(
            Candidate(
                sensor=series["sensor"].iloc[0],
                start=segment["time"].iloc[0],
                end=segment["time"].iloc[-1],
                peak_time=segment.loc[peak_idx, "time"],
                peak_value=float(peak_value),
                sigma=(peak_value - mean) / std,
                is_subthreshold=is_sub,
                window_index=-1,
            )
        )
    return events


def extract_candidates(
    per_sensor_series: Dict[str, pd.DataFrame],
    windows: List[Window],
    config: CandidateConfig,
) -> List[Candidate]:
    """Detect candidate events per sensor while preserving sub-threshold events."""
    all_candidates: List[Candidate] = []
    for sensor, df in per_sensor_series.items():
        if df.empty:
            continue
        events = _detect_events(df, config)
        for cand in events:
            cand.window_index = window_index_for_time(windows, sensor, cand.peak_time)
        if len(events) > config.max_candidates_per_sensor:
            events = events[: config.max_candidates_per_sensor]
        all_candidates.extend(events)
    return all_candidates


def candidates_to_frame(candidates: List[Candidate]) -> pd.DataFrame:
    """Convert candidate dataclasses to DataFrame."""
    return pd.DataFrame([dataclasses.asdict(c) for c in candidates])
