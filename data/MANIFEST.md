# Data Manifest

**Generated:** 2025-12-22 03:27:43 UTC

## Overview

This manifest documents the datasets downloaded for the MV-O1B pocket/intermittency test.

## Datasets

### A. GFZ GNSS Precise Products (2022)

**Provider:** GFZ German Research Centre for Geosciences
**Source:** https://isdc-data.gfz.de/gnss/products/final/
**License:** Free for scientific use (see GFZ data policy)
**Retrieved:** 2025-12-22 03:27:43 UTC

#### SP3 Orbit Files
- **Count:** 365 files
- **Total size:** 61.6 MB
- **Format:** SP3c (gzip compressed)
- **Cadence:** Daily, 15-minute sampling
- **Location:** `gfz/sp3/2022/`

#### CLK Clock Files
- **Count:** 365 files
- **Total size:** 928.4 MB
- **Format:** RINEX CLK (gzip compressed)
- **Cadence:** Daily, 30-second sampling
- **Location:** `gfz/clk/2022/`

### B. INTERMAGNET Magnetometer Data (2022)

**Provider:** INTERMAGNET / British Geological Survey
**Source:** https://imag-data.bgs.ac.uk/GIN_V1/hapi
**License:** See https://intermagnet.org/data_conditions.html
**Retrieved:** 2025-12-22 03:27:43 UTC

- **Stations:** 60
- **Total size:** 1.8 GB
- **Format:** CSV (HAPI format)
- **Cadence:** 1-minute
- **Components:** XYZF (definitive)
- **Location:** `mag/<station>/2022/`

## File Checksums

### SP3 Files
| File | Size | SHA256 |
|------|------|--------|
| `gfz/sp3/2022/GFZ0OPSFIN_20220010000_01D_15M_ORB.SP3.gz` | 139.5 KB | `bf5798b920247219...` |
| `gfz/sp3/2022/GFZ0OPSFIN_20220020000_01D_15M_ORB.SP3.gz` | 139.5 KB | `08a59c6c9f41f4f8...` |
| `gfz/sp3/2022/GFZ0OPSFIN_20220030000_01D_15M_ORB.SP3.gz` | 139.5 KB | `d097edd5edd343eb...` |
| `gfz/sp3/2022/GFZ0OPSFIN_20220040000_01D_15M_ORB.SP3.gz` | 139.6 KB | `3bb339090f6f8095...` |
| `gfz/sp3/2022/GFZ0OPSFIN_20220050000_01D_15M_ORB.SP3.gz` | 139.5 KB | `2cf9905b8c5c1dcc...` |
| `gfz/sp3/2022/GFZ0OPSFIN_20220060000_01D_15M_ORB.SP3.gz` | 139.5 KB | `aa5d30b3b778bd3c...` |
| `gfz/sp3/2022/GFZ0OPSFIN_20220070000_01D_15M_ORB.SP3.gz` | 139.5 KB | `86c63de9790c58f8...` |
| `gfz/sp3/2022/GFZ0OPSFIN_20220080000_01D_15M_ORB.SP3.gz` | 139.5 KB | `b2994295756ee648...` |
| `gfz/sp3/2022/GFZ0OPSFIN_20220090000_01D_15M_ORB.SP3.gz` | 139.4 KB | `9d73df42fd54a968...` |
| `gfz/sp3/2022/GFZ0OPSFIN_20220100000_01D_15M_ORB.SP3.gz` | 139.4 KB | `d14407826eb37cee...` |
| ... | (355 more files) | ... |

### CLK Files
| File | Size | SHA256 |
|------|------|--------|
| `gfz/clk/2022/GFZ0OPSFIN_20220010000_01D_30S_CLK.CLK.gz` | 2.6 MB | `e08c5929dfa1a12d...` |
| `gfz/clk/2022/GFZ0OPSFIN_20220020000_01D_30S_CLK.CLK.gz` | 2.5 MB | `31447192e1156001...` |
| `gfz/clk/2022/GFZ0OPSFIN_20220030000_01D_30S_CLK.CLK.gz` | 2.5 MB | `83d70aea08693318...` |
| `gfz/clk/2022/GFZ0OPSFIN_20220040000_01D_30S_CLK.CLK.gz` | 2.5 MB | `31e3aac2bf690f56...` |
| `gfz/clk/2022/GFZ0OPSFIN_20220050000_01D_30S_CLK.CLK.gz` | 2.5 MB | `7dfeb6eff2cf3753...` |
| `gfz/clk/2022/GFZ0OPSFIN_20220060000_01D_30S_CLK.CLK.gz` | 2.5 MB | `857236a8a2b23433...` |
| `gfz/clk/2022/GFZ0OPSFIN_20220070000_01D_30S_CLK.CLK.gz` | 2.5 MB | `d15030c28a58fd4b...` |
| `gfz/clk/2022/GFZ0OPSFIN_20220080000_01D_30S_CLK.CLK.gz` | 2.5 MB | `a517cc42cc8bd721...` |
| `gfz/clk/2022/GFZ0OPSFIN_20220090000_01D_30S_CLK.CLK.gz` | 2.5 MB | `174bf5ee8e3ec6cb...` |
| `gfz/clk/2022/GFZ0OPSFIN_20220100000_01D_30S_CLK.CLK.gz` | 2.5 MB | `94f8ccc35bed528d...` |
| ... | (355 more files) | ... |

### Magnetometer Stations
| Station | Lat | Lon | Files | Total Size |
|---------|-----|-----|-------|------------|
| ABG | 18.638000000000005 | 72.872 | 12 | 30.1 MB |
| AIA | -65.245 | 295.742 | 12 | 31.1 MB |
| ASP | -23.762 | 133.883 | 12 | 31.1 MB |
| BDV | 49.08 | 14.02 | 12 | 30.6 MB |
| BEL | 51.836 | 20.789 | 11 | 28.0 MB |
| BFO | 48.331 | 8.325 | 12 | 30.6 MB |
| BMT | 40.3 | 116.2 | 11 | 28.5 MB |
| BOU | 40.14 | 254.767 | 12 | 30.6 MB |
| BRD | 49.87 | 260.026 | 12 | 30.6 MB |
| BRW | 71.32 | 203.38 | 12 | 30.1 MB |
| BSL | 30.35 | 270.36 | 12 | 30.6 MB |
| CKI | -12.188000000000002 | 96.834 | 12 | 31.6 MB |
| CLF | 48.025 | 2.26 | 12 | 30.1 MB |
| CMO | 64.87 | 212.14 | 12 | 30.6 MB |
| CNB | -35.31999999999999 | 149.36 | 12 | 31.1 MB |
| CPL | 17.293999999999997 | 78.919 | 12 | 30.6 MB |
| CSY | -66.28299999999999 | 110.533 | 12 | 31.6 MB |
| CTA | -20.090000000000003 | 146.264 | 12 | 31.1 MB |
| DOU | 50.1 | 4.6 | 12 | 30.1 MB |
| DUR | 41.65 | 14.47 | 12 | 30.6 MB |
| EBR | 40.957 | 0.333 | 12 | 30.1 MB |
| ESK | 55.314 | 356.794 | 12 | 30.6 MB |
| EYR | -43.47399999999999 | 172.393 | 12 | 31.1 MB |
| FCC | 58.759 | 265.912 | 12 | 30.2 MB |
| FRD | 38.21 | 282.633 | 12 | 31.1 MB |
| FRN | 37.09 | 240.28 | 12 | 30.6 MB |
| FUR | 48.17 | 11.28 | 12 | 30.6 MB |
| GCK | 44.633 | 20.767 | 12 | 30.6 MB |
| GDH | 69.252 | 306.467 | 12 | 30.6 MB |
| GNG | -31.355999999999995 | 115.715 | 12 | 31.1 MB |
| GUI | 28.320999999999998 | 343.559 | 12 | 31.1 MB |
| HAD | 51.0 | 355.52 | 12 | 30.6 MB |
| HBK | -25.879999999999995 | 27.71 | 12 | 31.5 MB |
| HER | -34.43000000000001 | 19.23 | 12 | 31.1 MB |
| HLP | 54.603 | 18.811 | 12 | 30.6 MB |
| HON | 21.319999999999993 | 202.0 | 12 | 30.6 MB |
| HRB | 47.873 | 18.19 | 12 | 30.6 MB |
| HRN | 77.0 | 15.55 | 12 | 30.1 MB |
| HUA | -12.049999999999997 | 284.67 | 12 | 30.6 MB |
| HYB | 17.42 | 78.55 | 12 | 30.6 MB |
| IQA | 63.753 | 291.482 | 12 | 31.1 MB |
| IZN | 40.5 | 29.72 | 12 | 30.6 MB |
| JCO | 70.356 | 211.201 | 12 | 30.1 MB |
| KAK | 36.232 | 140.186 | 12 | 31.1 MB |
| KDU | -12.689999999999998 | 132.47 | 12 | 31.1 MB |
| KEP | -54.28200000000001 | 323.507 | 12 | 31.6 MB |
| KIR | 67.843 | 20.42 | 12 | 30.1 MB |
| KMH | -26.540000000000006 | 18.11 | 12 | 31.5 MB |
| KNY | 31.42 | 130.88 | 12 | 31.1 MB |
| KOU | 5.209999999999994 | 307.27 | 12 | 30.6 MB |
| LER | 60.138000000000005 | 358.817 | 12 | 30.1 MB |
| LON | 45.408 | 16.659 | 12 | 30.6 MB |
| LRM | -22.22 | 114.1 | 12 | 30.2 MB |
| LYC | 64.612 | 18.748 | 12 | 30.6 MB |
| MAB | 50.298 | 5.682 | 12 | 30.6 MB |
| MAW | -67.6 | 62.88 | 12 | 31.6 MB |
| MCQ | -54.5 | 158.95 | 12 | 31.1 MB |
| MEA | 54.616 | 246.653 | 12 | 30.7 MB |
| MMB | 43.91 | 144.19 | 12 | 31.1 MB |
| NEW | 48.27 | 242.88 | 12 | 30.6 MB |

## Full Checksums (JSON)

See `checksums.json` for complete SHA256 checksums of all files.
