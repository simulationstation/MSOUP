# COWLS Dataset Integrity Report

Generated: 2025-12-20T19:40:11Z

## Repository Information

| Property | Value |
|----------|-------|
| Repository URL | https://github.com/Jammy2211/COWLS_COSMOS_Web_Lens_Survey |
| Commit Hash | `423ca21a0f4c3834e9ef299aa0300a52ee716bc7` |
| Commit Date | 2025-03-14 |
| Total Lenses | 435 |
| Total Files | 42,012 |
| Total Size | 20.32 GB |

## Dataset Overview

The COSMOS-Web Lens Survey (COWLS) provides JWST NIRCam imaging of strong gravitational lens candidates discovered in the COSMOS-Web field. This is the complete public release containing:

- **4 NIRCam bands**: F115W, F150W, F277W, F444W
- **Per-band files**: data.fits, noise_map.fits, psf.fits
- **Modeling outputs**: lens models, source reconstructions (where available)
- **Auxiliary data**: RGB images, HST archive data

## Score Bin Distribution

Lenses are categorized by quality score (M25 = highest quality modeled lenses, S12-S00 = candidates with decreasing confidence):

| Score Bin | Lens Count | Description |
|-----------|------------|-------------|
| **M25** | 17 | Highest quality - modeled lenses |
| **S12** | 2 | High confidence candidates |
| **S11** | 3 | High confidence candidates |
| **S10** | 12 | High confidence candidates |
| **S09** | 11 | Medium-high confidence |
| **S08** | 14 | Medium-high confidence |
| **S07** | 20 | Medium confidence |
| **S06** | 37 | Medium confidence |
| **S05** | 44 | Medium confidence |
| **S04** | 58 | Lower confidence |
| **S03** | 62 | Lower confidence |
| **S02** | 66 | Lower confidence |
| **S01** | 45 | Lowest confidence |
| **S00** | 44 | Lowest confidence |

**Total: 435 lenses**

## NIRCam Band Completeness

All 435 lenses have complete NIRCam coverage:

| Band | data.fits | noise_map.fits | psf.fits | Complete |
|------|-----------|----------------|----------|----------|
| F115W | 435/435 | 435/435 | 435/435 | 100% |
| F150W | 435/435 | 435/435 | 435/435 | 100% |
| F277W | 435/435 | 435/435 | 435/435 | 100% |
| F444W | 435/435 | 435/435 | 435/435 | 100% |

## Missing Files

**None.** All expected NIRCam imaging files (data.fits, noise_map.fits, psf.fits) are present for all 435 lenses across all 4 bands.

## Recommended Analysis Subsets

### Primary Subset (High Quality): M25 + S10-S12
- **34 lenses** total
- All have lens modeling results available
- Highest confidence lens candidates
- Recommended for initial scientific analysis

### Extended Subset: M25 + S07-S12
- **71 lenses** total
- Medium to high confidence candidates
- Suitable for statistical studies

### Full Dataset
- **435 lenses** total
- Includes all candidates from visual inspection
- Use for completeness studies or machine learning training

## File Structure per Lens

Each lens folder contains:
```
{LENS_CODE}/
├── F115W/
│   ├── data.fits          # Science image
│   ├── noise_map.fits     # Noise map
│   ├── psf.fits           # Point spread function
│   └── result/            # Modeling outputs (if available)
├── F150W/
│   └── (same structure)
├── F277W/
│   └── (same structure)
├── F444W/
│   └── (same structure)
├── archive_space/         # Additional HST/JWST bands
└── *.png, *.jpeg          # RGB and visualization images
```

## Data Provenance

- **Source**: COSMOS-Web (PID 1727; PIs: Kartaltepe & Casey)
- **Instrument**: JWST NIRCam
- **Survey Area**: 0.54 deg² in COSMOS field
- **Completion**: May 2024
- **Papers**: COWLS I (Nightingale+ 2025), COWLS II (Mahler+ 2025), COWLS III (Hogg+ 2025)

## Verification

To verify data integrity:
1. Check file hashes against `MANIFEST.json`
2. Each FITS file has SHA256 hash recorded
3. Re-run `generate_manifest.py` to validate

## Notes

- This is a complete download of the COWLS public release
- No partial download or fallback datasets were used
- All files verified present on local disk
- Dataset suitable for strong lensing analysis with Msoup/Mverse models
