#!/usr/bin/env python3
"""
Verify downloaded datasets against checksums in checksums.json.

Usage:
    python scripts/verify_data.py           # Verify all files
    python scripts/verify_data.py --quick   # Quick check (file existence and size only)
    python scripts/verify_data.py --fix     # Re-download corrupted files
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def sha256_file(filepath: Path) -> str:
    """Compute SHA256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_file(file_info: dict, base_dir: Path, quick: bool = False) -> tuple:
    """
    Verify a single file.
    Returns (status, message) where status is 'ok', 'missing', 'size_mismatch', or 'checksum_mismatch'.
    """
    rel_path = file_info.get('file', '')
    expected_size = file_info.get('size', 0)
    expected_sha256 = file_info.get('sha256', '')

    filepath = base_dir / rel_path

    if not filepath.exists():
        return ('missing', f"File not found: {rel_path}")

    actual_size = filepath.stat().st_size
    if actual_size != expected_size:
        return ('size_mismatch', f"Size mismatch: {rel_path} (expected {expected_size}, got {actual_size})")

    if quick:
        return ('ok', f"OK (quick): {rel_path}")

    actual_sha256 = sha256_file(filepath)
    if actual_sha256 != expected_sha256:
        return ('checksum_mismatch', f"Checksum mismatch: {rel_path}")

    return ('ok', f"OK: {rel_path}")


def main():
    parser = argparse.ArgumentParser(description="Verify downloaded datasets")
    parser.add_argument('--quick', action='store_true',
                        help='Quick check (file existence and size only, no checksums)')
    parser.add_argument('--fix', action='store_true',
                        help='Re-download corrupted or missing files')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show all files, not just errors')

    args = parser.parse_args()

    checksums_path = DATA_DIR / "checksums.json"

    if not checksums_path.exists():
        print("Error: checksums.json not found. Run download_pocket_data.py first.", file=sys.stderr)
        sys.exit(1)

    with open(checksums_path) as f:
        checksums = json.load(f)

    print(f"Verifying datasets (generated: {checksums.get('generated', 'unknown')})")
    print(f"Mode: {'quick (size only)' if args.quick else 'full (SHA256 checksums)'}\n")

    stats = {'ok': 0, 'missing': 0, 'size_mismatch': 0, 'checksum_mismatch': 0}
    failed_files = []

    # Verify GFZ SP3 files
    sp3_files = checksums.get('gfz', {}).get('sp3', [])
    print(f"=== GFZ SP3 files ({len(sp3_files)}) ===")
    for f in sp3_files:
        status, msg = verify_file(f, DATA_DIR, args.quick)
        stats[status] += 1
        if status != 'ok':
            failed_files.append(f)
            print(f"  ✗ {msg}")
        elif args.verbose:
            print(f"  ✓ {msg}")

    # Verify GFZ CLK files
    clk_files = checksums.get('gfz', {}).get('clk', [])
    print(f"\n=== GFZ CLK files ({len(clk_files)}) ===")
    for f in clk_files:
        status, msg = verify_file(f, DATA_DIR, args.quick)
        stats[status] += 1
        if status != 'ok':
            failed_files.append(f)
            print(f"  ✗ {msg}")
        elif args.verbose:
            print(f"  ✓ {msg}")

    # Verify INTERMAGNET files
    mag_stations = checksums.get('intermagnet', {}).get('stations', [])
    print(f"\n=== INTERMAGNET stations ({len(mag_stations)}) ===")
    for station in mag_stations:
        station_code = station.get('station', 'unknown')
        station_files = station.get('files', [])
        station_ok = True
        for f in station_files:
            status, msg = verify_file(f, DATA_DIR, args.quick)
            stats[status] += 1
            if status != 'ok':
                failed_files.append(f)
                station_ok = False
                print(f"  ✗ {msg}")
            elif args.verbose:
                print(f"  ✓ {msg}")
        if station_ok and not args.verbose:
            print(f"  ✓ {station_code}: all {len(station_files)} files OK")

    # Summary
    total = sum(stats.values())
    print(f"\n=== Summary ===")
    print(f"Total files: {total}")
    print(f"  OK:                {stats['ok']}")
    print(f"  Missing:           {stats['missing']}")
    print(f"  Size mismatch:     {stats['size_mismatch']}")
    print(f"  Checksum mismatch: {stats['checksum_mismatch']}")

    if failed_files:
        print(f"\n{len(failed_files)} file(s) need attention.")
        if args.fix:
            print("\nRe-downloading failed files...")
            # Import and use the download script
            from download_pocket_data import download_file
            from download_pocket_data import sha256_file as dl_sha256

            for f in failed_files:
                url = f.get('url', '')
                rel_path = f.get('file', '')
                if url and rel_path:
                    dest = DATA_DIR / rel_path
                    print(f"  Downloading: {rel_path}...")
                    if download_file(url, dest, resume=False):
                        new_sha256 = sha256_file(dest)
                        if new_sha256 == f.get('sha256', ''):
                            print(f"    ✓ Fixed")
                        else:
                            print(f"    ✗ Still mismatched after re-download")
                    else:
                        print(f"    ✗ Download failed")
        else:
            print("Run with --fix to re-download failed files.")
        sys.exit(1)
    else:
        print("\nAll files verified successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
