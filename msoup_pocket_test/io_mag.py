"""
Magnetometer ingestion utilities.

The pipeline expects magnetometer measurements in CSV or simple whitespace
formats with at least columns: time, station, bx, by, bz (in nT). Additional
columns are preserved. Streaming generators allow processing long spans
without loading everything at once.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional

import pandas as pd


REQUIRED_COLUMNS = {"time", "station", "bx", "by", "bz"}


def load_magnetometer_file(path: Path) -> pd.DataFrame:
    """Load a single magnetometer file."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, parse_dates=["time"])
    else:
        df = pd.read_table(path, delim_whitespace=True, parse_dates=["time"])

    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Magnetometer file {path} missing required columns {missing}")

    return df


def iter_magnetometer(paths: Iterable[Path], chunk_hours: float) -> Generator[pd.DataFrame, None, None]:
    """Yield magnetometer data in chronological batches."""
    for path in paths:
        df = load_magnetometer_file(path).sort_values("time")
        start = df["time"].min()
        end = df["time"].max()
        chunk = dt.timedelta(hours=chunk_hours)
        current = start
        while current <= end:
            window_end = current + chunk
            mask = (df["time"] >= current) & (df["time"] < window_end)
            batch = df.loc[mask].copy()
            if not batch.empty:
                yield batch
            current = window_end


def group_by_station(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Split magnetometer samples by station for parallel-friendly processing."""
    return {name: group.sort_values("time").reset_index(drop=True) for name, group in df.groupby("station")}
