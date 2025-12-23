# BAO Analysis LSS Catalog Data Documentation

## Overview

This repository contains the data staging infrastructure for a preregistered environment-overlap BAO (Baryon Acoustic Oscillations) analysis using wedges and BAO template fits. The analysis uses:

- **Baseline**: eBOSS DR16 (public, SDSS)
- **High-power follow-on**: DESI EDR and DR1 (public)

## Data Sources

### eBOSS DR16 (SDSS-IV)

| Dataset | Description | Redshift | Status |
|---------|-------------|----------|--------|
| LRGpCMASS | BOSS CMASS + eBOSS LRG combined | 0.6 < z < 1.0 | Primary |
| QSO | Quasar sample | 0.8 < z < 2.2 | Cross-check |

**Official Documentation:**
- LSS Overview: https://www.sdss4.org/dr17/spectro/lss/
- Data Model: https://data.sdss.org/datamodel/files/EBOSS_LSS/catalogs/DR16/
- SAS Directory: https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/

### DESI EDR (v2.0)

| Dataset | Description | Redshift | Status |
|---------|-------------|----------|--------|
| LRG | Luminous Red Galaxies | 0.4 < z < 1.1 | Primary |
| ELGnotqso | Emission Line Galaxies | 0.8 < z < 1.6 | Primary |
| QSO | Quasars | 0.8 < z < 2.1 | Primary |

**Official Documentation:**
- VAC Overview: https://data.desi.lbl.gov/doc/releases/edr/vac/lss/
- Data Model: https://desidatamodel.readthedocs.io/
- Data Directory: https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/

### DESI DR1 (v1.5)

| Dataset | Description | Redshift | Status |
|---------|-------------|----------|--------|
| LRG | Luminous Red Galaxies | 0.4 < z < 1.1 | Primary |
| ELG_LOPnotqso | Emission Line Galaxies | 0.8 < z < 1.6 | Primary |
| QSO | Quasars | 0.8 < z < 2.1 | Primary |

**Official Documentation:**
- DR1 Overview: https://data.desi.lbl.gov/doc/releases/dr1/
- LSS VAC: https://data.desi.lbl.gov/doc/releases/dr1/vac/lss/
- Data Directory: https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/LSS/iron/LSScats/v1.5/

### DESI DR2 Status

As of December 2023:
- **Cosmology chains**: Released (BAO + cosmology inference results)
- **Full spectroscopic catalogs**: Not yet public (expected in coming months)
- **LSS catalogs**: Pending full DR2 release

Reference: https://www.desi.lbl.gov/2025/10/06/desi-dr2-cosmology-chains-and-data-products-released/

## Directory Structure

```
MSOUP/
├── data/                          # Raw downloaded catalogs
│   ├── eboss/
│   │   └── dr16/
│   │       ├── LRGpCMASS/v1/     # Primary LRG sample
│   │       └── QSO/v1/           # Cross-check sample
│   └── desi/
│       ├── edr/
│       │   ├── LRG/v2.0/
│       │   ├── ELG/v2.0/
│       │   └── QSO/v2.0/
│       └── dr1/
│           ├── LRG/v1.5/
│           ├── ELG/v1.5/
│           └── QSO/v1.5/
├── data_processed/               # Normalized parquet files
│   ├── eboss/dr16/
│   └── desi/{edr,dr1}/
├── data_manifest.csv            # File inventory with checksums
├── catalog_config.yaml          # Analysis configuration
├── download_all.sh              # Reproducible download script
└── README_data.md               # This file
```

## How to Re-Download

### Prerequisites

```bash
# Required tools
sudo apt-get install curl wget

# Python packages for verification and processing
pip install astropy numpy pandas pyarrow pyyaml
```

### Download Commands

```bash
# Make script executable
chmod +x download_all.sh

# Download everything (~20 GB for minimal set)
./download_all.sh --all

# Or download specific surveys
./download_all.sh --eboss-only      # ~4 GB
./download_all.sh --desi-edr-only   # ~3 GB
./download_all.sh --desi-dr1-only   # ~15 GB

# Verify downloads
./download_all.sh --verify-only
```

### Manual Download (Alternative)

#### eBOSS DR16
```bash
BASE="https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16"

# LRGpCMASS
wget -P data/eboss/dr16/LRGpCMASS/v1/ \
  ${BASE}/eBOSS_LRGpCMASS_clustering_data-NGC-vDR16.fits \
  ${BASE}/eBOSS_LRGpCMASS_clustering_data-SGC-vDR16.fits \
  ${BASE}/eBOSS_LRGpCMASS_clustering_random-NGC-vDR16.fits \
  ${BASE}/eBOSS_LRGpCMASS_clustering_random-SGC-vDR16.fits
```

#### DESI EDR
```bash
BASE="https://data.desi.lbl.gov/public/edr/vac/edr/lss/v2.0/LSScats/clustering"

# LRG
wget -P data/desi/edr/LRG/v2.0/ \
  ${BASE}/LRG_N_clustering.dat.fits \
  ${BASE}/LRG_S_clustering.dat.fits \
  ${BASE}/LRG_N_0_clustering.ran.fits \
  ${BASE}/LRG_S_0_clustering.ran.fits
```

## Catalog Column Reference

### eBOSS LRGpCMASS Columns

| Column | Type | Description |
|--------|------|-------------|
| RA | float64 | J2000 right ascension (deg) |
| DEC | float64 | J2000 declination (deg) |
| Z | float64 | Spectroscopic redshift |
| WEIGHT_SYSTOT | float64 | Imaging systematics weight |
| WEIGHT_CP | float64 | Close-pair (fiber collision) weight |
| WEIGHT_NOZ | float64 | Redshift failure weight |
| WEIGHT_FKP | float64 | FKP optimal weight (P0=10000) |
| NZ | float64 | Number density (h³ Mpc⁻³) |
| LRG_ID | int64 | Galaxy identifier |
| ISCMASS | bool | Flag: 1 if from CMASS sample |
| IN_EBOSS_FOOT | bool | Flag: 1 if in eBOSS footprint |

**Total weight for clustering**: `WEIGHT_SYSTOT * WEIGHT_CP * WEIGHT_NOZ * WEIGHT_FKP`

### DESI Clustering Columns

| Column | Type | Description |
|--------|------|-------------|
| RA | float64 | Target right ascension (deg) |
| DEC | float64 | Target declination (deg) |
| Z | float64 | Redshift from Redrock |
| TARGETID | int64 | Unique DESI target ID |
| WEIGHT | float64 | Combined weight (sys × comp × zfail) |
| WEIGHT_SYS | float64 | Imaging systematics (DR1 only) |
| WEIGHT_COMP | float64 | Fiber assignment completeness (DR1 only) |
| WEIGHT_ZFAIL | float64 | Redshift failure weight |
| WEIGHT_FKP | float64 | FKP optimal weight |
| NZ | float64 | Comoving number density (h³ Mpc⁻³) |
| NTILE | int64 | Number of available tiles |
| BITWEIGHTS | int64[2] | Fiber assignment alternative histories |
| PROB_OBS | float64 | Observation probability |

**Total weight for clustering**: `WEIGHT * WEIGHT_FKP`

## Weight Definitions

### eBOSS Weights

- **WEIGHT_SYSTOT**: Corrects for spatially varying completeness due to imaging conditions
- **WEIGHT_CP**: Upweights galaxies to account for close-pair fiber collisions
- **WEIGHT_NOZ**: Corrects for redshift measurement failures
- **WEIGHT_FKP**: `1 / (1 + n(z) * P0)` for optimal power spectrum estimation

### DESI Weights

- **WEIGHT_SYS**: Random forest regression correction for imaging systematics
- **WEIGHT_COMP**: Accounts for fiber assignment incompleteness
- **WEIGHT_ZFAIL**: Corrects for spectroscopic redshift failures
- **WEIGHT**: Pre-combined `WEIGHT_SYS * WEIGHT_COMP * WEIGHT_ZFAIL`
- **WEIGHT_FKP**: `1 / (1 + n(z) * P0)` with tracer-specific P0

## Integrity Checks

For each downloaded file, run:

```python
from astropy.io import fits
import numpy as np

def validate_catalog(filepath):
    with fits.open(filepath) as hdul:
        data = hdul[1].data

        # Row count
        print(f"Rows: {len(data)}")

        # RA/DEC ranges
        ra = data['RA']
        dec = data['DEC']
        assert np.all((ra >= 0) & (ra < 360)), "RA out of range"
        assert np.all((dec >= -90) & (dec <= 90)), "DEC out of range"

        # Redshift range
        z = data['Z']
        print(f"Z range: [{z.min():.3f}, {z.max():.3f}]")

        # Weight sanity
        for col in ['WEIGHT', 'WEIGHT_FKP', 'WEIGHT_SYSTOT']:
            if col in data.names:
                w = data[col]
                print(f"{col}: min={w.min():.4f}, max={w.max():.4f}, NaN={np.isnan(w).sum()}")
                assert np.all(np.isfinite(w)), f"{col} contains non-finite values"
                assert np.all(w >= 0), f"{col} contains negative values"
```

## Processing to Parquet

```python
import pandas as pd
from astropy.io import fits

def process_catalog(fits_path, output_path, columns, weight_expr):
    with fits.open(fits_path) as hdul:
        data = hdul[1].data

        df = pd.DataFrame({col: data[col] for col in columns})

        # Normalize coordinates
        df['RA'] = df['RA'] % 360.0

        # Compute total weight
        df['WEIGHT_TOTAL'] = eval(weight_expr, {'__builtins__': {}}, df.to_dict('series'))

        # Type consistency
        for col in ['RA', 'DEC', 'Z']:
            df[col] = df[col].astype('float64')

        df.to_parquet(output_path, compression='snappy', index=False)
```

## Required Software

| Tool | Version | Purpose |
|------|---------|---------|
| Python | ≥3.9 | Processing scripts |
| astropy | ≥5.0 | FITS file handling |
| numpy | ≥1.24 | Array operations |
| pandas | ≥2.0 | DataFrame processing |
| pyarrow | ≥14.0 | Parquet I/O |
| curl/wget | any | File downloads |
| sha256sum | any | Checksum verification |

## References

### eBOSS DR16
- Ross et al. (2020): "The Completed SDSS-IV extended Baryon Oscillation Spectroscopic Survey: Large-scale structure catalogues for cosmological analysis" [arXiv:2007.09000]
- SDSS DR16 Paper: Ahumada et al. (2020) [arXiv:1912.02905]

### DESI
- DESI EDR Paper: DESI Collaboration (2024) [arXiv:2306.06308]
- DESI DR1 Paper: DESI Collaboration (2025)
- LSS Catalog Construction: Ross et al. (2024) [arXiv:2405.16593]
- DESI Y1 BAO: DESI Collaboration (2024) [arXiv:2404.03000]

## Contact

For catalog-specific questions:
- eBOSS: SDSS Helpdesk (https://www.sdss.org/helpdesk/)
- DESI EDR/DR1 LSS: Ashley Ross (ross.1333@osu.edu)

## License

- SDSS Data: CC BY 4.0
- DESI Data: CC BY 4.0
