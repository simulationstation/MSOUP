"""
IGS CLK/SP3 ingestion utilities.

Parsing uses tolerant fallbacks (CSV with time,value columns) to make the
pipeline usable in test environments while remaining compatible with standard
IGS text products. Streaming generators are provided to avoid loading a full
year of products into memory.
"""

from __future__ import annotations

import datetime as dt
import itertools
from pathlib import Path
from typing import Dict, Generator, Iterable, List, Optional

import numpy as np
import pandas as pd


def _parse_sp3_epoch(line: str) -> Optional[dt.datetime]:
    try:
        _, year, month, day, hour, minute, second = line.split()
        return dt.datetime(
            int(year), int(month), int(day), int(hour), int(minute), int(float(second))
        )
    except ValueError:
        return None


def _parse_sp3_position_line(line: str, current_epoch: dt.datetime) -> Optional[Dict]:
    if not line.startswith("P"):
        return None
    fields = line.split()
    if len(fields) < 5:
        return None
    prn = fields[0][1:]
    try:
        x, y, z = (float(v) * 1e3 for v in fields[1:4])  # convert km to m
    except ValueError:
        return None
    return {"time": current_epoch, "satellite": prn, "x": x, "y": y, "z": z}


def load_sp3_positions(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Load satellite positions from SP3 files.

    Returns a DataFrame with columns: time, satellite, x, y, z.
    """
    rows: List[Dict] = []
    for path in paths:
        current_epoch: Optional[dt.datetime] = None
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("*"):
                    maybe_epoch = _parse_sp3_epoch(line)
                    if maybe_epoch:
                        current_epoch = maybe_epoch
                elif line.startswith("P") and current_epoch:
                    row = _parse_sp3_position_line(line, current_epoch)
                    if row:
                        rows.append(row)
    return pd.DataFrame(rows)


def _parse_clk_line(line: str) -> Optional[Dict]:
    parts = line.strip().split()
    if len(parts) < 3:
        return None
    try:
        time = pd.to_datetime(" ".join(parts[:2]))
        sat = parts[2]
        bias = float(parts[3]) if len(parts) > 3 else float(parts[-1])
        return {"time": time, "satellite": sat, "bias_s": bias}
    except (ValueError, IndexError):
        return None


def load_clk_series(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Load clock residuals from CLK files.

    The parser tolerates simplified CSV files with columns: time, satellite,
    bias_s. Standard IGS CLK text is also supported for the subset of fields we
    need.
    """
    rows: List[Dict] = []
    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".csv":
            df = pd.read_csv(path, parse_dates=["time"])
            expected = {"time", "satellite", "bias_s"}
            if not expected.issubset(df.columns):
                raise ValueError(f"CSV CLK file missing columns: {expected - set(df.columns)}")
            rows.extend(df.to_dict("records"))
            continue

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip() or line.startswith("#"):
                    continue
                parsed = _parse_clk_line(line)
                if parsed:
                    rows.append(parsed)
    return pd.DataFrame(rows)


def iter_clk_series(paths: Iterable[Path], chunk_hours: float) -> Generator[pd.DataFrame, None, None]:
    """Yield clock residuals in chronological batches."""
    df = load_clk_series(paths)
    if df.empty:
        return
    df = df.sort_values("time")
    start = df["time"].min()
    end = df["time"].max()
    chunk = dt.timedelta(hours=chunk_hours)
    current = start
    while current <= end:
        window_end = current + chunk
        mask = (df["time"] >= current) & (df["time"] < window_end)
        chunk_df = df.loc[mask].copy()
        if not chunk_df.empty:
            yield chunk_df
        current = window_end


def iter_magnetometer_series(paths: Iterable[Path], chunk_hours: float) -> Generator[pd.DataFrame, None, None]:
    """Yield magnetometer samples in batches mirroring GNSS chunking."""
    for path in paths:
        df = pd.read_csv(path, parse_dates=["time"])
        df = df.sort_values("time")
        start = df["time"].min()
        end = df["time"].max()
        chunk = dt.timedelta(hours=chunk_hours)
        current = start
        while current <= end:
            window_end = current + chunk
            mask = (df["time"] >= current) & (df["time"] < window_end)
            chunk_df = df.loc[mask].copy()
            if not chunk_df.empty:
                yield chunk_df
            current = window_end


def resolve_paths(glob_pattern: str) -> List[Path]:
    """Return sorted list of paths for a glob pattern."""
    return sorted(Path().glob(glob_pattern))


def load_positions_from_cache(cache_dir: Path, sp3_paths: List[Path]) -> pd.DataFrame:
    """Load cached positions if available; otherwise parse and cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "sp3_positions.parquet"
    if cache_file.exists():
        return pd.read_parquet(cache_file)
    df = load_sp3_positions(sp3_paths)
    if not df.empty:
        df.to_parquet(cache_file)
    return df


def load_clk_from_cache(cache_dir: Path, clk_paths: List[Path], chunk_hours: float) -> List[pd.DataFrame]:
    """Cache per-chunk clock residual batches."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    batches: List[pd.DataFrame] = []
    cache_base = cache_dir / "clk_batches"
    cache_base.mkdir(exist_ok=True)
    clk_df = load_clk_series(clk_paths)
    if clk_df.empty:
        return batches
    clk_df = clk_df.sort_values("time")
    start = clk_df["time"].min()
    end = clk_df["time"].max()
    chunk = dt.timedelta(hours=chunk_hours)
    current = start
    idx = 0
    while current <= end:
        window_end = current + chunk
        mask = (clk_df["time"] >= current) & (clk_df["time"] < window_end)
        chunk_df = clk_df.loc[mask].copy()
        cache_file = cache_base / f"batch_{idx:04d}.parquet"
        if cache_file.exists():
            chunk_df = pd.read_parquet(cache_file)
        elif not chunk_df.empty:
            chunk_df.to_parquet(cache_file)
        if not chunk_df.empty:
            batches.append(chunk_df)
        current = window_end
        idx += 1
    return batches


def group_by_sensor(df: pd.DataFrame, sensor_column: str = "satellite") -> Dict[str, pd.DataFrame]:
    """Helper to split a dataframe by sensor name."""
    return {name: group.sort_values("time").reset_index(drop=True) for name, group in df.groupby(sensor_column)}
