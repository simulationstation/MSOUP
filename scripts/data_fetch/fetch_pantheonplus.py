#!/usr/bin/env python3
"""
Fetch and process Pantheon+ SN Ia data from the public GitHub release.

Source: https://github.com/PantheonPlusSH0ES/DataRelease
Paper: Scolnic et al. 2022 (ApJ 938, 113) - Pantheon+ Analysis
       Riess et al. 2022 (ApJL 934, L7) - SH0ES H0 Measurement

Usage:
    python scripts/data_fetch/fetch_pantheonplus.py
"""

import os
import json
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path

# Configuration
RAW_DIR = Path("data_ignored/probes/pantheonplus/raw")
OUT_DIR = Path("data_ignored/probes/pantheonplus")

URLS = {
    "data": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat",
    "cov": "https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov",
}


def download_files():
    """Download raw data files from GitHub."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    for name, url in URLS.items():
        outfile = RAW_DIR / f"Pantheon_{name}.dat" if name == "data" else RAW_DIR / "Pantheon_STAT_SYS.cov"
        if name == "data":
            outfile = RAW_DIR / "Pantheon_SH0ES.dat"
        print(f"Downloading {name}...")
        subprocess.run(["curl", "-sL", url, "-o", str(outfile)], check=True)
        print(f"  Saved: {outfile} ({outfile.stat().st_size} bytes)")


def process_data():
    """Process raw data into standardized format."""
    # Load data
    df = pd.read_csv(RAW_DIR / "Pantheon_SH0ES.dat", sep=r'\s+', comment='#')
    print(f"Loaded {len(df)} SN entries")

    # Load covariance
    with open(RAW_DIR / "Pantheon_STAT_SYS.cov", 'r') as f:
        lines = f.readlines()
    n = int(lines[0].strip())
    cov_values = []
    for line in lines[1:]:
        cov_values.extend([float(x) for x in line.strip().split()])
    C = np.array(cov_values).reshape(n, n)
    C = (C + C.T) / 2  # Ensure symmetry

    # Create output CSV
    output_df = pd.DataFrame({
        'z': df['zCMB'].values,
        'mu_obs': df['MU_SH0ES'].values,
        'mu_err_diag': np.sqrt(C.diagonal())
    })
    output_df.to_csv(OUT_DIR / "pantheonplus_mu.csv", index=False, float_format='%.6f')
    print(f"Saved: {OUT_DIR / 'pantheonplus_mu.csv'}")

    # Save covariance
    np.savez_compressed(
        OUT_DIR / "pantheonplus_cov_stat_sys.npz",
        C=C,
        idx=np.arange(len(df)),
        CID=df['CID'].values
    )
    print(f"Saved: {OUT_DIR / 'pantheonplus_cov_stat_sys.npz'}")

    # Create splits
    splits = {
        'lowz': {'description': 'z < 0.1', 'indices': output_df[output_df['z'] < 0.1].index.tolist()},
        'midz': {'description': '0.1 <= z < 0.5', 'indices': output_df[(output_df['z'] >= 0.1) & (output_df['z'] < 0.5)].index.tolist()},
        'highz': {'description': 'z >= 0.8', 'indices': output_df[output_df['z'] >= 0.8].index.tolist()}
    }
    for key in splits:
        idx = splits[key]['indices']
        z_vals = output_df.loc[idx, 'z']
        splits[key].update({
            'N': len(idx),
            'z_min': float(z_vals.min()),
            'z_median': float(z_vals.median()),
            'z_max': float(z_vals.max())
        })

    with open(OUT_DIR / "pantheonplus_splits.json", 'w') as f:
        json.dump(splits, f, indent=2)
    print(f"Saved: {OUT_DIR / 'pantheonplus_splits.json'}")

    return output_df, C, splits


def main():
    print("=" * 60)
    print("Pantheon+ Data Fetch Script")
    print("=" * 60)

    download_files()
    df, C, splits = process_data()

    print("\nSummary:")
    print(f"  Total: N={len(df)}, z=[{df['z'].min():.4f}, {df['z'].median():.4f}, {df['z'].max():.4f}]")
    for key, val in splits.items():
        print(f"  {key}: N={val['N']}, z=[{val['z_min']:.4f}, {val['z_median']:.4f}, {val['z_max']:.4f}]")
    print(f"  Covariance: {C.shape}, eigenvalue range: [{np.linalg.eigvalsh(C).min():.4e}, {np.linalg.eigvalsh(C).max():.2e}]")


if __name__ == "__main__":
    main()
