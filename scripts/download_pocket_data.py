#!/usr/bin/env python3
"""
Download real public datasets for MV-O1B pocket/intermittency test.

Datasets:
A) GFZ GNSS precise clock (CLK) and orbit (SP3) products for 2022
B) INTERMAGNET magnetometer data for ~50+ globally distributed stations (2022)

All downloads are resumable and cache-aware.
Generates SHA256 checksums for verification.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin
import urllib.request
import urllib.error


# Configuration
DATA_DIR = Path(__file__).parent.parent / "data"
GFZ_BASE_URL = "https://isdc-data.gfz.de/gnss/products/final/"
INTERMAGNET_HAPI_BASE = "https://imag-data.bgs.ac.uk/GIN_V1/hapi"

# Date range for 2022
START_DATE = date(2022, 1, 1)
END_DATE = date(2022, 12, 31)

# GPS epoch for week calculation
GPS_EPOCH = date(1980, 1, 6)


def gps_week(d: date) -> int:
    """Calculate GPS week number for a given date."""
    return (d - GPS_EPOCH).days // 7


def day_of_year(d: date) -> int:
    """Calculate day of year (1-366)."""
    return (d - date(d.year, 1, 1)).days + 1


def sha256_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, resume: bool = True) -> bool:
    """
    Download a file with optional resume support.
    Returns True if download succeeded or file already exists.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Check if file already exists and is complete
    if dest.exists():
        # Try HEAD request to check size
        try:
            req = urllib.request.Request(url, method='HEAD')
            with urllib.request.urlopen(req, timeout=30) as resp:
                remote_size = int(resp.headers.get('Content-Length', 0))
                local_size = dest.stat().st_size
                if remote_size > 0 and local_size == remote_size:
                    return True  # Already complete
        except Exception:
            pass  # Proceed with download if HEAD fails

    # Download with resume support
    headers = {}
    mode = 'wb'
    if resume and dest.exists():
        existing_size = dest.stat().st_size
        headers['Range'] = f'bytes={existing_size}-'
        mode = 'ab'
    else:
        existing_size = 0

    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=60) as resp:
            # Check if server supports range requests
            if resp.status == 206:  # Partial content
                pass
            elif resp.status == 200:
                # Server doesn't support resume, start fresh
                mode = 'wb'
                existing_size = 0

            total_size = int(resp.headers.get('Content-Length', 0)) + existing_size

            with open(dest, mode) as f:
                downloaded = existing_size
                for chunk in iter(lambda: resp.read(8192), b''):
                    f.write(chunk)
                    downloaded += len(chunk)

        return True
    except urllib.error.HTTPError as e:
        if e.code == 416:  # Range not satisfiable = file complete
            return True
        print(f"  HTTP Error {e.code}: {url}", file=sys.stderr)
        return False
    except Exception as e:
        print(f"  Download error: {e}", file=sys.stderr)
        return False


def download_gfz_gnss(year: int = 2022, dry_run: bool = False) -> dict:
    """
    Download GFZ GNSS SP3 and CLK files for a full year.

    Returns dict with file paths and checksums.
    """
    manifest = {"sp3": [], "clk": []}

    start = date(year, 1, 1)
    end = date(year, 12, 31)

    # Calculate week range
    start_week = gps_week(start)
    end_week = gps_week(end)

    print(f"\n=== Downloading GFZ GNSS data for {year} ===")
    print(f"GPS weeks {start_week} to {end_week}")

    current = start
    file_count = 0
    total_files = (end - start).days + 1

    while current <= end:
        week = gps_week(current)
        doy = day_of_year(current)
        year_str = str(current.year)
        doy_str = f"{doy:03d}"

        # GFZ naming: GFZ0OPSFIN_YYYYDOY0000_01D_15M_ORB.SP3.gz
        sp3_filename = f"GFZ0OPSFIN_{year_str}{doy_str}0000_01D_15M_ORB.SP3.gz"
        clk_filename = f"GFZ0OPSFIN_{year_str}{doy_str}0000_01D_30S_CLK.CLK.gz"

        week_dir = f"w{week}/"
        sp3_url = urljoin(GFZ_BASE_URL, week_dir + sp3_filename)
        clk_url = urljoin(GFZ_BASE_URL, week_dir + clk_filename)

        # Organize by YYYY/DOY structure
        sp3_dest = DATA_DIR / "gfz" / "sp3" / year_str / sp3_filename
        clk_dest = DATA_DIR / "gfz" / "clk" / year_str / clk_filename

        file_count += 1
        print(f"\r[{file_count}/{total_files}] {current.isoformat()}: ", end="", flush=True)

        if dry_run:
            print(f"(dry run) SP3: {sp3_url}")
            print(f"               CLK: {clk_url}")
        else:
            # Download SP3
            if download_file(sp3_url, sp3_dest):
                checksum = sha256_file(sp3_dest)
                manifest["sp3"].append({
                    "file": str(sp3_dest.relative_to(DATA_DIR)),
                    "url": sp3_url,
                    "size": sp3_dest.stat().st_size,
                    "sha256": checksum,
                    "date": current.isoformat()
                })
                print("SP3 ✓ ", end="", flush=True)
            else:
                print("SP3 ✗ ", end="", flush=True)

            # Download CLK
            if download_file(clk_url, clk_dest):
                checksum = sha256_file(clk_dest)
                manifest["clk"].append({
                    "file": str(clk_dest.relative_to(DATA_DIR)),
                    "url": clk_url,
                    "size": clk_dest.stat().st_size,
                    "sha256": checksum,
                    "date": current.isoformat()
                })
                print("CLK ✓", flush=True)
            else:
                print("CLK ✗", flush=True)

        current += timedelta(days=1)

    return manifest


def get_intermagnet_stations(limit: Optional[int] = None) -> list:
    """
    Get list of INTERMAGNET stations with definitive minute data.
    Returns list of station codes.
    """
    catalog_url = f"{INTERMAGNET_HAPI_BASE}/catalog"

    try:
        with urllib.request.urlopen(catalog_url, timeout=60) as resp:
            data = json.load(resp)
    except Exception as e:
        print(f"Error fetching catalog: {e}", file=sys.stderr)
        return []

    # Extract unique stations with definitive/PT1M/xyzf data
    stations = set()
    for item in data.get('catalog', []):
        dataset_id = item.get('id', '')
        parts = dataset_id.split('/')
        if len(parts) >= 4:
            station = parts[0].upper()
            data_type = parts[1]
            cadence = parts[2]
            orientation = parts[3]
            # Prefer definitive minute data in XYZF
            if data_type == 'definitive' and cadence == 'PT1M' and orientation == 'xyzf':
                stations.add(station)

    stations = sorted(stations)
    if limit:
        stations = stations[:limit]

    return stations


def download_intermagnet_station(station: str, year: int = 2022, dry_run: bool = False) -> Optional[dict]:
    """
    Download INTERMAGNET data for a single station for a year.
    Uses HAPI data endpoint with CSV output.
    """
    dataset_id = f"{station.lower()}/definitive/PT1M/xyzf"

    # First check if this dataset exists and covers our period
    info_url = f"{INTERMAGNET_HAPI_BASE}/info?id={dataset_id}"

    try:
        with urllib.request.urlopen(info_url, timeout=30) as resp:
            info = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None  # Dataset doesn't exist
        raise
    except Exception:
        return None

    # Check date coverage
    start_date = info.get('startDate', '')
    stop_date = info.get('stopDate', '')

    if not start_date or not stop_date:
        return None

    # Parse dates (format: YYYY-MM-DDTHH:MM:SSZ)
    try:
        data_start = datetime.fromisoformat(start_date.replace('Z', '+00:00')).date()
        data_stop = datetime.fromisoformat(stop_date.replace('Z', '+00:00')).date()
    except:
        return None

    # Check if 2022 is covered
    year_start = date(year, 1, 1)
    year_end = date(year, 12, 31)

    if data_start > year_end or data_stop < year_start:
        return None  # No data for this year

    # Adjust query range to available data
    query_start = max(data_start, year_start)
    query_end = min(data_stop, year_end)

    if dry_run:
        print(f"  Would download {station}: {query_start} to {query_end}")
        return {"station": station, "start": query_start.isoformat(), "end": query_end.isoformat()}

    # Download in monthly chunks to avoid timeout
    station_dir = DATA_DIR / "mag" / station.lower() / str(year)
    station_dir.mkdir(parents=True, exist_ok=True)

    files = []
    current_month_start = query_start.replace(day=1)

    while current_month_start <= query_end:
        # Calculate month end
        if current_month_start.month == 12:
            next_month = date(current_month_start.year + 1, 1, 1)
        else:
            next_month = date(current_month_start.year, current_month_start.month + 1, 1)
        month_end = min(next_month - timedelta(days=1), query_end)

        # Adjust start if needed
        month_start = max(current_month_start, query_start)

        # File for this month
        filename = f"{station.lower()}_{month_start.strftime('%Y%m%d')}_{month_end.strftime('%Y%m%d')}.csv"
        dest = station_dir / filename

        if not dest.exists():
            # HAPI data endpoint
            data_url = (
                f"{INTERMAGNET_HAPI_BASE}/data?"
                f"id={dataset_id}&"
                f"time.min={month_start.isoformat()}T00:00:00Z&"
                f"time.max={month_end.isoformat()}T23:59:00Z&"
                f"format=csv"
            )

            try:
                req = urllib.request.Request(data_url)
                with urllib.request.urlopen(req, timeout=120) as resp:
                    content = resp.read()
                    with open(dest, 'wb') as f:
                        f.write(content)
            except Exception as e:
                print(f"    Error downloading {station} {month_start.strftime('%Y-%m')}: {e}")
                current_month_start = next_month
                continue

        if dest.exists() and dest.stat().st_size > 0:
            files.append({
                "file": str(dest.relative_to(DATA_DIR)),
                "size": dest.stat().st_size,
                "sha256": sha256_file(dest),
                "start": month_start.isoformat(),
                "end": month_end.isoformat()
            })

        current_month_start = next_month

    if not files:
        return None

    return {
        "station": station,
        "lat": info.get('x_latitude'),
        "lon": info.get('x_longitude'),
        "elevation": info.get('x_elevation'),
        "files": files
    }


def download_intermagnet(year: int = 2022, station_limit: int = 60, dry_run: bool = False) -> dict:
    """
    Download INTERMAGNET magnetometer data for multiple stations.

    Returns dict with station data and checksums.
    """
    manifest = {"stations": []}

    print(f"\n=== Downloading INTERMAGNET magnetometer data for {year} ===")
    print("Fetching station catalog...")

    all_stations = get_intermagnet_stations()
    print(f"Found {len(all_stations)} stations with definitive minute data")

    # We'll try all stations but some may not have 2022 data
    successful = 0

    for i, station in enumerate(all_stations):
        if station_limit and successful >= station_limit:
            break

        print(f"\r[{i+1}/{len(all_stations)}] Checking {station}... ", end="", flush=True)

        result = download_intermagnet_station(station, year, dry_run)

        if result:
            manifest["stations"].append(result)
            successful += 1
            if not dry_run:
                n_files = len(result.get('files', []))
                print(f"✓ ({n_files} files)", flush=True)
            else:
                print("✓ (would download)", flush=True)
        else:
            print("✗ (no 2022 data)", flush=True)

    print(f"\nSuccessfully processed {successful} stations")

    return manifest


def generate_manifest(gfz_manifest: dict, mag_manifest: dict) -> str:
    """Generate MANIFEST.md content."""

    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    # Calculate totals
    sp3_files = gfz_manifest.get("sp3", [])
    clk_files = gfz_manifest.get("clk", [])
    mag_stations = mag_manifest.get("stations", [])

    sp3_total = sum(f.get("size", 0) for f in sp3_files)
    clk_total = sum(f.get("size", 0) for f in clk_files)
    mag_total = sum(
        sum(f.get("size", 0) for f in s.get("files", []))
        for s in mag_stations
    )

    def human_size(size: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} TB"

    content = f"""# Data Manifest

**Generated:** {now}

## Overview

This manifest documents the datasets downloaded for the MV-O1B pocket/intermittency test.

## Datasets

### A. GFZ GNSS Precise Products (2022)

**Provider:** GFZ German Research Centre for Geosciences
**Source:** https://isdc-data.gfz.de/gnss/products/final/
**License:** Free for scientific use (see GFZ data policy)
**Retrieved:** {now}

#### SP3 Orbit Files
- **Count:** {len(sp3_files)} files
- **Total size:** {human_size(sp3_total)}
- **Format:** SP3c (gzip compressed)
- **Cadence:** Daily, 15-minute sampling
- **Location:** `gfz/sp3/2022/`

#### CLK Clock Files
- **Count:** {len(clk_files)} files
- **Total size:** {human_size(clk_total)}
- **Format:** RINEX CLK (gzip compressed)
- **Cadence:** Daily, 30-second sampling
- **Location:** `gfz/clk/2022/`

### B. INTERMAGNET Magnetometer Data (2022)

**Provider:** INTERMAGNET / British Geological Survey
**Source:** https://imag-data.bgs.ac.uk/GIN_V1/hapi
**License:** See https://intermagnet.org/data_conditions.html
**Retrieved:** {now}

- **Stations:** {len(mag_stations)}
- **Total size:** {human_size(mag_total)}
- **Format:** CSV (HAPI format)
- **Cadence:** 1-minute
- **Components:** XYZF (definitive)
- **Location:** `mag/<station>/2022/`

## File Checksums

### SP3 Files
| File | Size | SHA256 |
|------|------|--------|
"""

    for f in sp3_files[:10]:  # Show first 10, summarize rest
        content += f"| `{f['file']}` | {human_size(f['size'])} | `{f['sha256'][:16]}...` |\n"

    if len(sp3_files) > 10:
        content += f"| ... | ({len(sp3_files) - 10} more files) | ... |\n"

    content += """
### CLK Files
| File | Size | SHA256 |
|------|------|--------|
"""

    for f in clk_files[:10]:
        content += f"| `{f['file']}` | {human_size(f['size'])} | `{f['sha256'][:16]}...` |\n"

    if len(clk_files) > 10:
        content += f"| ... | ({len(clk_files) - 10} more files) | ... |\n"

    content += """
### Magnetometer Stations
| Station | Lat | Lon | Files | Total Size |
|---------|-----|-----|-------|------------|
"""

    for s in mag_stations:
        lat = s.get('lat', 'N/A')
        lon = s.get('lon', 'N/A')
        files = s.get('files', [])
        size = sum(f.get('size', 0) for f in files)
        content += f"| {s['station']} | {lat} | {lon} | {len(files)} | {human_size(size)} |\n"

    content += """
## Full Checksums (JSON)

See `checksums.json` for complete SHA256 checksums of all files.
"""

    return content


def generate_readme() -> str:
    """Generate README_DATA.md content."""
    return """# Data Directory

This directory contains real observational datasets for the MV-O1B pocket/intermittency test.

## Datasets

### GFZ GNSS Products (`gfz/`)

Precise satellite orbit (SP3) and clock (CLK) products from GFZ for 2022.

- **SP3 files:** Satellite positions at 15-minute intervals
- **CLK files:** Satellite and station clock corrections at 30-second intervals

### INTERMAGNET Magnetometer Data (`mag/`)

Ground-based magnetometer measurements from the INTERMAGNET network for 2022.

- **Format:** CSV with XYZF components (nanotesla)
- **Cadence:** 1-minute definitive data
- **Coverage:** ~50+ globally distributed observatories

## Re-downloading

To re-download all data:

```bash
cd /path/to/MSOUP
python scripts/download_pocket_data.py --all
```

To download only specific datasets:

```bash
# GFZ GNSS data only
python scripts/download_pocket_data.py --gfz

# INTERMAGNET magnetometer data only
python scripts/download_pocket_data.py --mag

# Limit number of magnetometer stations
python scripts/download_pocket_data.py --mag --station-limit 20
```

## Verification

To verify downloaded files against checksums:

```bash
python scripts/verify_data.py
```

## Data Sources

- **GFZ:** https://isdc-data.gfz.de/gnss/products/final/
- **INTERMAGNET:** https://imag-data.bgs.ac.uk/GIN_V1/hapi

## Notes

- All data files are excluded from git (see .gitignore)
- Downloads are resumable - re-running will skip completed files
- Total download size is approximately 1-2 GB
"""


def main():
    parser = argparse.ArgumentParser(
        description="Download datasets for MV-O1B pocket/intermittency test"
    )
    parser.add_argument('--all', action='store_true', help='Download all datasets')
    parser.add_argument('--gfz', action='store_true', help='Download GFZ GNSS data')
    parser.add_argument('--mag', action='store_true', help='Download INTERMAGNET data')
    parser.add_argument('--year', type=int, default=2022, help='Year to download (default: 2022)')
    parser.add_argument('--station-limit', type=int, default=60,
                        help='Max magnetometer stations (default: 60)')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be downloaded')

    args = parser.parse_args()

    # Default to --all if no specific dataset selected
    if not (args.all or args.gfz or args.mag):
        args.all = True

    gfz_manifest = {"sp3": [], "clk": []}
    mag_manifest = {"stations": []}

    if args.all or args.gfz:
        gfz_manifest = download_gfz_gnss(args.year, args.dry_run)

    if args.all or args.mag:
        mag_manifest = download_intermagnet(args.year, args.station_limit, args.dry_run)

    if not args.dry_run:
        # Generate manifest
        print("\n=== Generating manifest files ===")

        manifest_content = generate_manifest(gfz_manifest, mag_manifest)
        manifest_path = DATA_DIR / "MANIFEST.md"
        with open(manifest_path, 'w') as f:
            f.write(manifest_content)
        print(f"Written: {manifest_path}")

        # Save full checksums as JSON
        checksums = {
            "generated": datetime.utcnow().isoformat() + "Z",
            "gfz": gfz_manifest,
            "intermagnet": mag_manifest
        }
        checksums_path = DATA_DIR / "checksums.json"
        with open(checksums_path, 'w') as f:
            json.dump(checksums, f, indent=2)
        print(f"Written: {checksums_path}")

        # Generate README
        readme_content = generate_readme()
        readme_path = DATA_DIR / "README_DATA.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"Written: {readme_path}")

        print("\n=== Download complete ===")
        print(f"SP3 files: {len(gfz_manifest['sp3'])}")
        print(f"CLK files: {len(gfz_manifest['clk'])}")
        print(f"Magnetometer stations: {len(mag_manifest['stations'])}")


if __name__ == "__main__":
    main()
