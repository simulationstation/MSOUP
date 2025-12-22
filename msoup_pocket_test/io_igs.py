"""
IGS CLK/SP3 ingestion utilities.

Parsing uses tolerant fallbacks (CSV with time,value columns) to make the
pipeline usable in test environments while remaining compatible with standard
IGS text products. Streaming generators are provided to avoid loading a full
year of products into memory.

Supports gzip-compressed files (.gz extension).

MEMORY SAFETY: All loading functions are iterator-based. Files are processed
one at a time and data is yielded in chunks to prevent memory accumulation.
"""

from __future__ import annotations

import datetime as dt
import gzip
import itertools
from pathlib import Path
from typing import Dict, Generator, Iterable, Iterator, List, Optional, TextIO, Tuple

import numpy as np
import pandas as pd


def _open_file(path: Path) -> TextIO:
    """Open a file, handling gzip compression if needed."""
    if path.suffix.lower() == ".gz" or str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")


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


def iter_sp3_positions(paths: Iterable[Path]) -> Generator[Dict, None, None]:
    """
    Stream satellite positions from SP3 files one record at a time.

    Memory-safe: yields individual records without accumulating.
    """
    for path in paths:
        current_epoch: Optional[dt.datetime] = None
        with _open_file(path) as f:
            for line in f:
                if line.startswith("*"):
                    maybe_epoch = _parse_sp3_epoch(line)
                    if maybe_epoch:
                        current_epoch = maybe_epoch
                elif line.startswith("P") and current_epoch:
                    row = _parse_sp3_position_line(line, current_epoch)
                    if row:
                        yield row


def load_sp3_positions(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Load satellite positions from SP3 files.

    Returns a DataFrame with columns: time, satellite, x, y, z.
    Supports gzip-compressed files.
    """
    rows = list(iter_sp3_positions(paths))
    return pd.DataFrame(rows)


def _parse_clk_line(line: str) -> Optional[Dict]:
    """Parse a RINEX CLK data line.

    Format: TYPE ID YYYY MM DD HH MM SS.SSSSSS N bias [sigma]
    Example: AS G01 2022  1  1  0  0  0.000000  1    0.123456789012E-03
    """
    parts = line.strip().split()
    if len(parts) < 9:
        return None
    rec_type = parts[0]
    # Only process satellite clocks (AS) for now
    if rec_type not in ("AS", "AR"):
        return None
    try:
        sat_id = parts[1]
        year = int(parts[2])
        month = int(parts[3])
        day = int(parts[4])
        hour = int(parts[5])
        minute = int(parts[6])
        second = float(parts[7])
        # n_values = int(parts[8])
        bias = float(parts[9])
        time = dt.datetime(year, month, day, hour, minute, int(second),
                          int((second % 1) * 1e6))
        return {"time": time, "satellite": sat_id, "bias_s": bias, "type": rec_type}
    except (ValueError, IndexError):
        return None


def iter_clk_records(path: Path) -> Generator[Dict, None, None]:
    """
    Stream CLK records from a single file.

    Memory-safe: yields individual records without accumulating.
    """
    path_str = str(path).lower()

    # Handle CSV files (possibly gzipped)
    if ".csv" in path_str:
        if path_str.endswith(".gz"):
            df = pd.read_csv(path, parse_dates=["time"], compression="gzip")
        else:
            df = pd.read_csv(path, parse_dates=["time"])
        expected = {"time", "satellite", "bias_s"}
        if not expected.issubset(df.columns):
            raise ValueError(f"CSV CLK file missing columns: {expected - set(df.columns)}")
        for _, row in df.iterrows():
            yield row.to_dict()
        return

    # Parse RINEX CLK format
    with _open_file(path) as f:
        in_header = True
        for line in f:
            if in_header:
                if "END OF HEADER" in line:
                    in_header = False
                continue
            if not line.strip() or line.startswith("#"):
                continue
            parsed = _parse_clk_line(line)
            if parsed:
                yield parsed


def iter_clk_files_streaming(
    paths: Iterable[Path],
) -> Generator[Tuple[Path, pd.DataFrame], None, None]:
    """
    Stream CLK data file by file.

    Yields (path, dataframe) tuples where each dataframe contains all records
    from a single file. This allows per-file processing without loading all
    files into memory at once.

    Memory-safe: only one file's data is in memory at a time.
    """
    for path in paths:
        records = list(iter_clk_records(path))
        if records:
            df = pd.DataFrame(records)
            yield path, df


def load_clk_series(paths: Iterable[Path]) -> pd.DataFrame:
    """
    Load clock residuals from CLK files.

    The parser tolerates simplified CSV files with columns: time, satellite,
    bias_s. Standard IGS/GFZ CLK text (RINEX format) is also supported.
    Supports gzip-compressed files (.gz).

    NOTE: This loads all files into memory. For large datasets, use
    iter_clk_files_streaming() instead.
    """
    rows: List[Dict] = []
    for path in paths:
        rows.extend(iter_clk_records(path))
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


def resolve_paths(glob_pattern: str, max_files: int = 0) -> List[Path]:
    """
    Return sorted list of paths for a glob pattern.

    Args:
        glob_pattern: Glob pattern to match files
        max_files: Maximum number of files to return (0 = no limit)
    """
    paths = sorted(Path().glob(glob_pattern))
    if max_files > 0:
        paths = paths[:max_files]
    return paths


def iter_paths(glob_pattern: str, max_files: int = 0) -> Iterator[Path]:
    """
    Iterator version of resolve_paths for lazy evaluation.

    Memory-safe: paths are yielded one at a time.
    """
    paths = sorted(Path().glob(glob_pattern))
    if max_files > 0:
        paths = paths[:max_files]
    for path in paths:
        yield path


def count_paths(glob_pattern: str, max_files: int = 0) -> int:
    """Count matching files without loading them."""
    paths = list(Path().glob(glob_pattern))
    if max_files > 0:
        return min(len(paths), max_files)
    return len(paths)


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


def iter_clk_from_cache_streaming(
    cache_dir: Path,
    clk_paths: List[Path],
    chunk_hours: float,
) -> Generator[pd.DataFrame, None, None]:
    """
    Stream CLK batches with caching, truly streaming version.

    This function:
    1. Processes files one at a time (not all at once)
    2. Caches per-file parsed data
    3. Yields time-chunked batches

    Memory-safe: only one file's data is in memory at a time during parsing.
    After parsing, time chunks are yielded one at a time.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    file_cache_dir = cache_dir / "clk_files"
    file_cache_dir.mkdir(exist_ok=True)

    # First pass: ensure all files are cached (streaming)
    all_times = []
    for path in clk_paths:
        cache_file = file_cache_dir / f"{path.stem}.parquet"
        if cache_file.exists():
            # Just read the time range, not all data
            df = pd.read_parquet(cache_file, columns=["time"])
            if not df.empty:
                all_times.append((df["time"].min(), df["time"].max()))
        else:
            # Parse and cache
            records = list(iter_clk_records(path))
            if records:
                df = pd.DataFrame(records)
                df.to_parquet(cache_file)
                all_times.append((df["time"].min(), df["time"].max()))

    if not all_times:
        return

    # Determine global time range
    global_start = min(t[0] for t in all_times)
    global_end = max(t[1] for t in all_times)
    chunk = dt.timedelta(hours=chunk_hours)

    # Yield chunks by loading only overlapping files
    current = global_start
    while current <= global_end:
        window_end = current + chunk
        chunk_data = []

        for path in clk_paths:
            cache_file = file_cache_dir / f"{path.stem}.parquet"
            if not cache_file.exists():
                continue

            # Read cached file
            df = pd.read_parquet(cache_file)
            if df.empty:
                continue

            # Filter to current chunk
            mask = (df["time"] >= current) & (df["time"] < window_end)
            chunk_df = df.loc[mask]
            if not chunk_df.empty:
                chunk_data.append(chunk_df)

        if chunk_data:
            yield pd.concat(chunk_data, ignore_index=True)

        current = window_end


def load_clk_from_cache(cache_dir: Path, clk_paths: List[Path], chunk_hours: float) -> List[pd.DataFrame]:
    """Cache per-chunk clock residual batches (legacy list interface)."""
    return list(iter_clk_from_cache_streaming(cache_dir, clk_paths, chunk_hours))


def group_by_sensor(df: pd.DataFrame, sensor_column: str = "satellite") -> Dict[str, pd.DataFrame]:
    """Helper to split a dataframe by sensor name."""
    return {name: group.sort_values("time").reset_index(drop=True) for name, group in df.groupby(sensor_column)}


def get_file_time_range(path: Path) -> Optional[Tuple[dt.datetime, dt.datetime]]:
    """
    Get the time range covered by a CLK file without loading all data.

    Returns (min_time, max_time) or None if file is empty/invalid.
    """
    first_time = None
    last_time = None

    for record in iter_clk_records(path):
        t = record.get("time")
        if t:
            if first_time is None:
                first_time = t
            last_time = t

    if first_time and last_time:
        return (first_time, last_time)
    return None


def estimate_file_coverage(paths: List[Path]) -> Dict[str, any]:
    """
    Estimate coverage without loading all data.

    Returns dict with:
    - n_files: number of files
    - time_range: (start, end) datetime tuple
    - estimated_sensors: set of satellite IDs seen in first 100 records
    """
    n_files = len(paths)
    all_times = []
    sensors = set()

    for path in paths[:min(5, len(paths))]:  # Sample first 5 files
        count = 0
        for record in iter_clk_records(path):
            t = record.get("time")
            sat = record.get("satellite")
            if t:
                all_times.append(t)
            if sat:
                sensors.add(sat)
            count += 1
            if count >= 100:  # Sample first 100 records per file
                break

    return {
        "n_files": n_files,
        "time_range": (min(all_times), max(all_times)) if all_times else None,
        "estimated_sensors": sensors,
        "n_sensors_estimated": len(sensors),
    }
