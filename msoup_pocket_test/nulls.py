"""
Null construction conditioned on sensitivity windows.
"""

from __future__ import annotations

from enum import Enum
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm


class NullType(Enum):
    """Type of null construction to use."""
    CONDITIONED_RESAMPLE = "conditioned_resample"
    BLOCK_BOOTSTRAP = "block_bootstrap"
    TIME_SHIFT = "time_shift"


def can_time_shift(windows: pd.DataFrame) -> bool:
    """
    Determine whether time-shift nulls are safe: requires near-continuous
    coverage per sensor with no overlapping windows.
    """
    if windows.empty:
        return False
    total_range = (windows["end"].max() - windows["start"].min()).total_seconds()
    grouped = windows.groupby("sensor")
    for _, group in grouped:
        if len(group) != 1:
            return False
        coverage = (group["end"].iloc[0] - group["start"].iloc[0]).total_seconds()
        if coverage < 0.8 * total_range:
            return False
    return True


def window_conditioned_resample(
    candidates: pd.DataFrame,
    windows: pd.DataFrame,
    block_length: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Resample candidate times within each window using block bootstrap to
    preserve autocorrelation and censoring.
    """
    resampled_rows = []
    for sensor, group in candidates.groupby("sensor"):
        sensor_windows = windows[windows["sensor"] == sensor]
        if sensor_windows.empty or group.empty:
            continue
        for _, win in sensor_windows.iterrows():
            in_window = group[(group["peak_time"] >= win["start"]) & (group["peak_time"] <= win["end"])]
            if in_window.empty:
                continue
            times = in_window["peak_time"].sort_values()
            # Convert to seconds since window start
            offsets = (times - win["start"]).dt.total_seconds().to_numpy()
            n = len(offsets)
            min_offset = offsets.min()
            max_offset = offsets.max()
            duration = max(1.0, max_offset - min_offset)
            new_offsets = []
            while len(new_offsets) < n:
                if n <= block_length:
                    # For short segments, draw uniformly within the window
                    new_offsets.extend(rng.uniform(min_offset, min_offset + duration, size=n - len(new_offsets)))
                    continue
                start_idx = rng.integers(0, n)
                end_idx = start_idx + block_length
                if end_idx <= n:
                    block = offsets[start_idx:end_idx]
                else:
                    block = np.concatenate([offsets[start_idx:], offsets[: end_idx - n]])
                shift = rng.uniform(min_offset, min_offset + duration)
                shifted = np.mod(block - min_offset + shift, duration) + min_offset
                new_offsets.extend(shifted.tolist())
            new_offsets = np.array(new_offsets[:n])
            jitter_scale = min(win.get("cadence_seconds", 30), duration * 0.1)
            if jitter_scale > 0:
                new_offsets = np.clip(
                    new_offsets + rng.uniform(-jitter_scale / 2, jitter_scale / 2, size=n),
                    min_offset,
                    min_offset + duration,
                )
            new_times = win["start"] + pd.to_timedelta(new_offsets, unit="s")
            sample = in_window.copy()
            sample.loc[:, "peak_time"] = new_times.to_numpy()
            resampled_rows.append(sample)
    if not resampled_rows:
        return pd.DataFrame(columns=candidates.columns)
    return pd.concat(resampled_rows, ignore_index=True)


def time_shift_resample(candidates: pd.DataFrame, windows: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Apply a global circular time shift within each sensor's coverage if allowed.
    """
    resampled_rows = []
    for sensor, group in candidates.groupby("sensor"):
        sensor_windows = windows[windows["sensor"] == sensor]
        if sensor_windows.empty or group.empty:
            continue
        coverage = (
            sensor_windows["end"].max() - sensor_windows["start"].min()
        ).total_seconds()
        shift = rng.uniform(0, coverage)
        shifted = group.copy()
        shifted.loc[:, "peak_time"] = group["peak_time"] + pd.to_timedelta(shift, unit="s")
        resampled_rows.append(shifted)
    if not resampled_rows:
        return pd.DataFrame(columns=candidates.columns)
    return pd.concat(resampled_rows, ignore_index=True)


def generate_null_catalogs(
    candidates: pd.DataFrame,
    windows: pd.DataFrame,
    n_realizations: int,
    block_length: int,
    allow_time_shift: bool,
    seed: int,
    show_progress: bool = True,
) -> List[pd.DataFrame]:
    """Generate null catalogs conditioned on windows."""
    rng = np.random.default_rng(seed)
    use_time_shift = allow_time_shift and can_time_shift(windows)
    catalogs: List[pd.DataFrame] = []
    iterator = range(n_realizations)
    if show_progress:
        iterator = tqdm(iterator, desc="Generating nulls", leave=False)
    for i in iterator:
        if use_time_shift and i % 2 == 0:
            catalogs.append(time_shift_resample(candidates, windows, rng))
        else:
            catalogs.append(window_conditioned_resample(candidates, windows, block_length, rng))
    return catalogs
