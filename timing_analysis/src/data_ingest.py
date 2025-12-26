"""
Data Ingestion Module for Timing Analysis Pipeline

Handles downloading and parsing of:
- IGS GNSS clock products (.clk files)
- BIPM Circular T clock comparison data

Author: Claude Code
"""

import os
import gzip
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# Base URLs for data sources
IGS_BASE_URL = "https://igs.bkg.bund.de/root_ftp/IGS/products"
BIPM_BASE_URL = "https://webtai.bipm.org/ftp/pub/tai"


def download_file(url: str, dest_path: Path, timeout: int = 60) -> bool:
    """Download a file from URL to destination path."""
    try:
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        print(f"  Warning: Failed to download {url}: {e}")
        return False


def gps_week_day_from_date(date: datetime) -> Tuple[int, int]:
    """Convert datetime to GPS week and day of week."""
    gps_epoch = datetime(1980, 1, 6)
    delta = date - gps_epoch
    gps_week = delta.days // 7
    day_of_week = delta.days % 7
    return gps_week, day_of_week


def doy_from_date(date: datetime) -> int:
    """Get day of year from datetime."""
    return date.timetuple().tm_yday


def download_igs_clock_files(
    start_date: datetime,
    end_date: datetime,
    output_dir: Path,
    sample_rate: str = "30S"  # "30S" for 30-second, "05M" for 5-minute
) -> List[Path]:
    """
    Download IGS clock files for date range.

    Args:
        start_date: Start date
        end_date: End date
        output_dir: Directory to save files
        sample_rate: "30S" or "05M"

    Returns:
        List of downloaded file paths
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    current = start_date

    while current <= end_date:
        gps_week, _ = gps_week_day_from_date(current)
        year = current.year
        doy = doy_from_date(current)

        # IGS filename format: IGS0OPSFIN_YYYYDDDHHMMSS_01D_XXS_CLK.CLK.gz
        filename = f"IGS0OPSFIN_{year}{doy:03d}0000_01D_{sample_rate}_CLK.CLK.gz"
        url = f"{IGS_BASE_URL}/{gps_week}/{filename}"

        dest_gz = output_dir / filename
        dest_clk = output_dir / filename.replace('.gz', '')

        if dest_clk.exists():
            print(f"  Already exists: {dest_clk.name}")
            downloaded.append(dest_clk)
        elif dest_gz.exists() or download_file(url, dest_gz):
            # Decompress
            if dest_gz.exists():
                try:
                    with gzip.open(dest_gz, 'rb') as f_in:
                        with open(dest_clk, 'wb') as f_out:
                            f_out.write(f_in.read())
                    downloaded.append(dest_clk)
                    print(f"  Downloaded: {dest_clk.name}")
                except Exception as e:
                    print(f"  Warning: Failed to decompress {dest_gz}: {e}")

        current += timedelta(days=1)

    return downloaded


def parse_igs_clock_file(filepath: Path) -> Dict[str, pd.DataFrame]:
    """
    Parse IGS RINEX clock file (.clk).

    Returns dict of DataFrames, one per clock (station or satellite).
    Each DataFrame has columns: ['epoch', 'bias', 'bias_sigma']
    """
    clocks = {}

    with open(filepath, 'r') as f:
        in_header = True

        for line in f:
            # Skip header until END OF HEADER
            if in_header:
                if 'END OF HEADER' in line:
                    in_header = False
                continue

            # Parse data records
            # Format: AR/AS name epoch(6 fields) num_values bias [sigma] [rate] [rate_sigma]
            parts = line.split()
            if len(parts) < 9:
                continue

            record_type = parts[0]  # AR=receiver, AS=satellite
            clock_name = parts[1]

            try:
                year = int(parts[2])
                month = int(parts[3])
                day = int(parts[4])
                hour = int(parts[5])
                minute = int(parts[6])
                second = float(parts[7])

                epoch = datetime(year, month, day, hour, minute, int(second))
                epoch = epoch.replace(microsecond=int((second % 1) * 1e6))

                num_values = int(parts[8])
                bias = float(parts[9])  # seconds
                bias_sigma = float(parts[10]) if len(parts) > 10 else np.nan

            except (ValueError, IndexError):
                continue

            if clock_name not in clocks:
                clocks[clock_name] = []

            clocks[clock_name].append({
                'epoch': epoch,
                'bias': bias,
                'bias_sigma': bias_sigma,
                'type': record_type
            })

    # Convert to DataFrames
    result = {}
    for name, records in clocks.items():
        df = pd.DataFrame(records)
        df = df.sort_values('epoch').reset_index(drop=True)
        result[name] = df

    return result


def download_bipm_circular_t(
    year: int,
    output_dir: Path
) -> Optional[Path]:
    """
    Download BIPM Circular T ASCII file for a given year.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to get the directory listing first
    url = f"{BIPM_BASE_URL}/Circular-T/cirt/"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Find files for this year
        import re
        pattern = rf'href="(cirt\d+\.txt)"'
        files = re.findall(pattern, response.text)

        downloaded = []
        for fname in files:
            file_url = f"{url}{fname}"
            dest = output_dir / fname
            if not dest.exists():
                if download_file(file_url, dest):
                    downloaded.append(dest)
            else:
                downloaded.append(dest)

        return downloaded

    except Exception as e:
        print(f"Warning: Failed to download BIPM Circular T: {e}")
        return None


def parse_bipm_circular_t(filepath: Path) -> pd.DataFrame:
    """
    Parse BIPM Circular T file.

    Returns DataFrame with columns: ['mjd', 'lab', 'utc_minus_utck']
    where utc_minus_utck is in nanoseconds.
    """
    records = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or line.startswith('*'):
                continue

            parts = line.split()
            if len(parts) < 3:
                continue

            try:
                mjd = float(parts[0])
                lab = parts[1]
                utc_diff = float(parts[2])  # nanoseconds typically
                records.append({
                    'mjd': mjd,
                    'lab': lab,
                    'utc_minus_utck': utc_diff
                })
            except ValueError:
                continue

    return pd.DataFrame(records)


def combine_clock_streams(
    clock_data: Dict[str, pd.DataFrame],
    clock_type: str = "AR"  # AR=receiver, AS=satellite
) -> pd.DataFrame:
    """
    Combine multiple clock streams into a single DataFrame with one column per clock.

    Args:
        clock_data: Dict from parse_igs_clock_file
        clock_type: Filter by record type

    Returns:
        DataFrame indexed by epoch, columns are clock names
    """
    # Filter by type and get unique epochs
    all_epochs = set()
    filtered = {}

    for name, df in clock_data.items():
        type_df = df[df['type'] == clock_type].copy()
        if len(type_df) > 0:
            filtered[name] = type_df.set_index('epoch')['bias']
            all_epochs.update(type_df['epoch'].tolist())

    if not filtered:
        return pd.DataFrame()

    # Create combined DataFrame
    epochs = sorted(all_epochs)
    combined = pd.DataFrame(index=epochs)

    for name, series in filtered.items():
        combined[name] = series

    combined.index.name = 'epoch'
    return combined


def compute_clock_residuals(combined: pd.DataFrame) -> pd.DataFrame:
    """
    Compute clock residuals by removing linear drift.

    For each clock, fits a linear trend and returns residuals.
    """
    residuals = pd.DataFrame(index=combined.index)

    for col in combined.columns:
        series = combined[col].dropna()
        if len(series) < 10:
            continue

        # Convert to numeric time (seconds from start)
        t0 = series.index[0]
        t_seconds = np.array([(t - t0).total_seconds() for t in series.index])
        y = series.values

        # Fit linear trend
        valid = np.isfinite(y)
        if np.sum(valid) < 10:
            continue

        coeffs = np.polyfit(t_seconds[valid], y[valid], 1)
        trend = np.polyval(coeffs, t_seconds)

        # Store residuals
        residual_series = pd.Series(y - trend, index=series.index)
        residuals[col] = residual_series

    return residuals


def get_stream_info(clock_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Get summary information about each clock stream.
    """
    info = []
    for name, df in clock_data.items():
        if len(df) == 0:
            continue
        info.append({
            'name': name,
            'type': df['type'].iloc[0] if 'type' in df.columns else 'unknown',
            'n_samples': len(df),
            'start': df['epoch'].min(),
            'end': df['epoch'].max(),
            'mean_bias': df['bias'].mean(),
            'std_bias': df['bias'].std()
        })

    return pd.DataFrame(info)


if __name__ == "__main__":
    # Test data ingestion
    from datetime import datetime

    output_dir = Path("/home/primary/MSOUP/timing_analysis/data_ingest")

    # Download one day of IGS data
    start = datetime(2024, 12, 1)
    end = datetime(2024, 12, 1)

    print("Testing IGS clock download...")
    files = download_igs_clock_files(start, end, output_dir)

    if files:
        print(f"\nParsing {files[0]}...")
        clocks = parse_igs_clock_file(files[0])
        print(f"Found {len(clocks)} clock streams")

        info = get_stream_info(clocks)
        print("\nStream summary:")
        print(info.head(10))
