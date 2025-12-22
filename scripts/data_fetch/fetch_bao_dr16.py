#!/usr/bin/env python3
"""
Create BAO DR16 data from published SDSS eBOSS measurements.

Source: https://www.sdss4.org/science/final-bao-and-rsd-measurements/
Papers:
  - Alam et al. 2017 (BOSS DR12)
  - Alam et al. 2021 (eBOSS DR16 final)
  - Various eBOSS tracer papers (LRG, ELG, QSO, Lya)

Usage:
    python scripts/data_fetch/fetch_bao_dr16.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

OUT_DIR = Path("data_ignored/probes/bao_dr16")


def create_bao_data():
    """Create BAO data from published measurements."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # BAO measurements from SDSS DR16 compilation
    # https://www.sdss4.org/science/final-bao-and-rsd-measurements/
    bao_data = [
        # z_eff, observable, value, sigma, tracer, paper_tag
        # MGS
        (0.15, 'DV/rd', 4.47, 0.17, 'MGS', 'Ross2015'),
        # BOSS Galaxy (z=0.38, 0.51)
        (0.38, 'DM/rd', 10.23, 0.17, 'BOSS_Galaxy', 'Alam2017'),
        (0.38, 'DH/rd', 25.00, 0.76, 'BOSS_Galaxy', 'Alam2017'),
        (0.51, 'DM/rd', 13.36, 0.21, 'BOSS_Galaxy', 'Alam2017'),
        (0.51, 'DH/rd', 22.33, 0.58, 'BOSS_Galaxy', 'Alam2017'),
        # eBOSS LRG
        (0.70, 'DM/rd', 17.86, 0.33, 'eBOSS_LRG', 'Bautista2021'),
        (0.70, 'DH/rd', 19.33, 0.53, 'eBOSS_LRG', 'Bautista2021'),
        # eBOSS ELG
        (0.85, 'DM/rd', 19.17, 0.99, 'eBOSS_ELG', 'Raichoor2021'),
        (0.85, 'DH/rd', 19.6, 2.1, 'eBOSS_ELG', 'Raichoor2021'),
        # eBOSS Quasar
        (1.48, 'DM/rd', 30.69, 0.80, 'eBOSS_QSO', 'Hou2021'),
        (1.48, 'DH/rd', 13.26, 0.55, 'eBOSS_QSO', 'Hou2021'),
        # Lya-Lya
        (2.33, 'DM/rd', 37.6, 1.9, 'Lya_Lya', 'duMas2020'),
        (2.33, 'DH/rd', 8.93, 0.28, 'Lya_Lya', 'duMas2020'),
        # Lya-Quasar cross
        (2.33, 'DM/rd', 37.3, 1.7, 'Lya_QSO', 'duMas2020'),
        (2.33, 'DH/rd', 9.08, 0.34, 'Lya_QSO', 'duMas2020'),
    ]

    df = pd.DataFrame(bao_data, columns=['z', 'observable', 'value', 'sigma', 'tracer', 'paper_tag'])
    df.to_csv(OUT_DIR / 'bao.csv', index=False, float_format='%.4f')
    print(f"Saved: {OUT_DIR / 'bao.csv'}")

    return df


def create_covariance(df):
    """Create approximate covariance matrix."""
    n = len(df)
    C = np.zeros((n, n))

    # Diagonal
    for i in range(n):
        C[i, i] = df.iloc[i]['sigma'] ** 2

    # Off-diagonal: DM-DH correlation at same (z, tracer)
    r_DM_DH = -0.4

    for (z, tracer), group in df.groupby(['z', 'tracer']):
        if len(group) == 2:
            obs = group['observable'].tolist()
            if 'DM/rd' in obs and 'DH/rd' in obs:
                idx = group.index.tolist()
                i, j = idx
                sigma_i = df.iloc[i]['sigma']
                sigma_j = df.iloc[j]['sigma']
                C[i, j] = r_DM_DH * sigma_i * sigma_j
                C[j, i] = C[i, j]

    # Lya cross-correlation at z=2.33
    r_Lya_cross = 0.3
    lya_z = 2.33
    for obs in ['DM/rd', 'DH/rd']:
        idx_list = df[(df['z'] == lya_z) & (df['observable'] == obs)].index.tolist()
        if len(idx_list) == 2:
            i, j = idx_list
            sigma_i = df.iloc[i]['sigma']
            sigma_j = df.iloc[j]['sigma']
            C[i, j] = r_Lya_cross * sigma_i * sigma_j
            C[j, i] = C[i, j]

    # Ensure symmetry
    C = (C + C.T) / 2

    # Save
    labels = [f"{row['tracer']}_{row['observable']}_{row['z']:.2f}" for _, row in df.iterrows()]
    np.savez_compressed(
        OUT_DIR / 'bao_cov.npz',
        C=C,
        labels=labels,
        z=df['z'].values,
        observable=df['observable'].values,
        tracer=df['tracer'].values
    )
    print(f"Saved: {OUT_DIR / 'bao_cov.npz'}")

    return C


def main():
    print("=" * 60)
    print("BAO DR16 Data Creation Script")
    print("=" * 60)

    df = create_bao_data()
    C = create_covariance(df)

    print("\nSummary:")
    print(f"  N = {len(df)}")
    print(f"  z range: [{df['z'].min():.2f}, {df['z'].max():.2f}]")
    print(f"  Tracers: {df['tracer'].unique().tolist()}")
    print(f"  Observables: {df['observable'].unique().tolist()}")

    eigvals = np.linalg.eigvalsh(C)
    print(f"  Covariance: {C.shape}, positive definite: {eigvals.min() > 0}")


if __name__ == "__main__":
    main()
