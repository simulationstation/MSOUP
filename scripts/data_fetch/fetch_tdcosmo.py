#!/usr/bin/env python3
"""
Create TDCOSMO/H0LiCOW time-delay lens data from published measurements.

Source: https://github.com/TDCOSMO/hierarchy_analysis_2020_public
Papers:
  - H0LiCOW XIII (Wong et al. 2020, MNRAS 498, 1420)
  - TDCOSMO IV (Birrer et al. 2020, A&A 643, A165)
  - Individual lens papers (see sources.txt)

Usage:
    python scripts/data_fetch/fetch_tdcosmo.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data_ignored/probes/tdcosmo")
RAW_DIR = OUT_DIR / "raw"


def create_tdcosmo_data():
    """Create TDCOSMO data from published D_dt measurements."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # D_dt values from H0LiCOW XIII and TDCOSMO papers
    # Units: Mpc
    tdcosmo_data = [
        # lens_id, z_lens, z_source, D_dt, sigma_D_dt, D_d, sigma_D_d, paper_tag
        ('B1608+656', 0.6304, 1.394, 5156, 236, 1228, 193, 'Suyu2010'),
        ('RXJ1131-1231', 0.295, 0.654, 1740, 155, 804, 226, 'Suyu2014'),
        ('HE0435-1223', 0.4546, 1.693, 2707, 183, np.nan, np.nan, 'Wong2017'),
        ('SDSS1206+4332', 0.745, 1.789, 5769, 589, 1805, 564, 'Birrer2019'),
        ('WFI2033-4723', 0.6575, 1.662, 4784, 399, np.nan, np.nan, 'Rusu2020'),
        ('PG1115+080', 0.311, 1.722, 1470, 137, np.nan, np.nan, 'Chen2019'),
        ('DES0408-5354', 0.597, 2.375, 3382, 146, 1711, 328, 'Shajib2020'),
    ]

    df = pd.DataFrame(tdcosmo_data, columns=[
        'lens_id', 'z_lens', 'z_source', 'D_dt', 'sigma_D_dt', 'D_d', 'sigma_D_d', 'paper_tag'
    ])

    df.to_csv(OUT_DIR / 'td.csv', index=False, float_format='%.1f')
    print(f"Saved: {OUT_DIR / 'td.csv'}")

    return df


def write_sources():
    """Write source documentation."""
    sources = """TDCOSMO/H0LiCOW Time-Delay Distance Data Sources
================================================

Primary Reference:
- H0LiCOW XIII (Wong et al. 2020, MNRAS 498, 1420)
  https://arxiv.org/abs/1907.04869

- TDCOSMO IV (Birrer et al. 2020, A&A 643, A165)
  https://arxiv.org/abs/2007.02941

Individual Lens Papers:
- B1608+656: Suyu et al. 2010, ApJ 711, 201
- RXJ1131-1231: Suyu et al. 2014, ApJ 788, L35
- HE0435-1223: Wong et al. 2017, MNRAS 465, 4895
- SDSS1206+4332: Birrer et al. 2019, MNRAS 484, 4726
- WFI2033-4723: Rusu et al. 2020, MNRAS 498, 1440
- PG1115+080: Chen et al. 2019, MNRAS 490, 1743
- DES0408-5354: Shajib et al. 2020, MNRAS 494, 6072

GitHub Repository:
- https://github.com/TDCOSMO/hierarchy_analysis_2020_public

Notes:
- D_dt is the time-delay distance in Mpc
- D_d is the angular diameter distance to the lens (where available)
- Uncertainties are symmetric approximations of asymmetric posteriors
"""
    with open(RAW_DIR / 'sources.txt', 'w') as f:
        f.write(sources)
    print(f"Saved: {RAW_DIR / 'sources.txt'}")


def main():
    print("=" * 60)
    print("TDCOSMO Data Creation Script")
    print("=" * 60)

    df = create_tdcosmo_data()
    write_sources()

    print("\nSummary:")
    print(f"  N = {len(df)} lenses")
    print(f"  z_lens range: [{df['z_lens'].min():.3f}, {df['z_lens'].max():.3f}]")
    print(f"  z_source range: [{df['z_source'].min():.3f}, {df['z_source'].max():.3f}]")
    print(f"  D_dt range: [{df['D_dt'].min():.0f}, {df['D_dt'].max():.0f}] Mpc")

    print("\nLens data:")
    print(df[['lens_id', 'z_lens', 'z_source', 'D_dt', 'sigma_D_dt']].to_string(index=False))


if __name__ == "__main__":
    main()
