"""
Magnetometer ingestion utilities.

The pipeline expects magnetometer measurements in CSV or simple whitespace
formats with at least columns: time, station, bx, by, bz (in nT). Additional
columns are preserved. Streaming generators allow processing long spans
without loading everything at once.

Supports HAPI-format CSVs (no headers, columns: time, X, Y, Z, F) where
the station name is inferred from the file path.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"time", "station", "bx", "by", "bz"}
HAPI_COLUMNS = ["time", "bx", "by", "bz", "btot"]  # INTERMAGNET HAPI format
FILL_VALUE = 99999.0


def _infer_station_from_path(path: Path) -> str:
    """Infer station code from file path like data/mag/abg/2022/..."""
    parts = path.parts
    for i, p in enumerate(parts):
        if p == "mag" and i + 1 < len(parts):
            return parts[i + 1].upper()
    # Fallback: use filename prefix
    return path.stem.split("_")[0].upper()


def load_magnetometer_file(path: Path) -> pd.DataFrame:
    """Load a single magnetometer file.

    Supports two formats:
    1. Standard format with headers: time, station, bx, by, bz
    2. HAPI format without headers: time, X, Y, Z, F (station inferred from path)
    """
    # Try to detect format by checking first line
    with open(path, "r") as f:
        first_line = f.readline().strip()

    # Check if first line looks like a header
    has_header = any(col in first_line.lower() for col in ["time", "station", "bx"])

    if has_header:
        # Standard format with headers
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path, parse_dates=["time"])
        else:
            df = pd.read_table(path, sep=r"\s+", parse_dates=["time"])
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            raise ValueError(f"Magnetometer file {path} missing required columns {missing}")
        return df

    # HAPI format: no headers, columns are time, X, Y, Z, F
    df = pd.read_csv(path, names=HAPI_COLUMNS, parse_dates=["time"])

    # Replace fill values with NaN
    for col in ["bx", "by", "bz", "btot"]:
        df.loc[df[col] >= FILL_VALUE - 1, col] = np.nan

    # Add station from path
    df["station"] = _infer_station_from_path(path)

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
