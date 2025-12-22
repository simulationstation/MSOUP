# Data Directory

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
