"""
Preprocessing and quality control for GNSS and magnetometer series.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal

from .config import PreprocessConfig


def detrend_series(series: pd.Series, config: PreprocessConfig) -> pd.Series:
    """
    Detrend a time series using a robust median baseline followed by a linear
    detrend on short timescales.
    """
    if series.empty:
        return series

    window = max(1, int(config.detrend_window_minutes * 60 / config.cadence_seconds))
    median = series.rolling(window=window, center=True, min_periods=1).median()
    residual = series - median

    # High-pass using SciPy detrend to remove slow drifts
    residual = pd.Series(signal.detrend(residual.to_numpy()), index=series.index)
    return residual


def compute_quality_mask(times: pd.Series, config: PreprocessConfig) -> pd.Series:
    """
    Compute a boolean mask indicating samples that reside in stable cadence
    windows. Cadence gaps beyond cadence_tolerance * cadence_seconds break
    windows.
    """
    if times.empty:
        return pd.Series(dtype=bool)
    diffs = times.diff().dt.total_seconds().fillna(0)
    expected = config.cadence_seconds
    mask = diffs <= config.max_gap_factor * expected
    mask.iloc[0] = True
    return mask


def apply_quality(series: pd.DataFrame, value_column: str, config: PreprocessConfig) -> pd.DataFrame:
    """Apply detrending and quality masks."""
    if series.empty:
        return series
    cleaned = series.copy()
    cleaned[value_column] = detrend_series(cleaned[value_column], config)
    cleaned["quality_mask"] = compute_quality_mask(cleaned["time"], config)
    return cleaned


def standardize_magnetometer(df: pd.DataFrame, config: PreprocessConfig) -> pd.DataFrame:
    """Return magnetometer magnitude residual with quality mask applied."""
    mag = df.copy()
    mag["btot"] = np.sqrt(mag["bx"] ** 2 + mag["by"] ** 2 + mag["bz"] ** 2)
    mag = apply_quality(mag, "btot", config)
    return mag


def drop_bad_windows(df: pd.DataFrame, mask_column: str = "quality_mask") -> pd.DataFrame:
    """Remove rows failing the quality mask."""
    if mask_column not in df:
        return df
    return df[df[mask_column]].reset_index(drop=True)
