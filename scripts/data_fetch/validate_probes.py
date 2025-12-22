#!/usr/bin/env python3
"""
Validate probe data files: check counts, z ranges, and covariance sanity.

Usage:
    python scripts/data_fetch/validate_probes.py
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def validate_pantheonplus():
    """Validate Pantheon+ data."""
    print("=" * 60)
    print("Pantheon+ SNe Ia")
    print("=" * 60)

    base = Path("data_ignored/probes/pantheonplus")

    # Load CSV
    df = pd.read_csv(base / "pantheonplus_mu.csv")
    print(f"  pantheonplus_mu.csv:")
    print(f"    N = {len(df)}")
    print(f"    z: min={df['z'].min():.4f}, median={df['z'].median():.4f}, max={df['z'].max():.4f}")

    # Load covariance
    data = np.load(base / "pantheonplus_cov_stat_sys.npz")
    C = data['C']
    print(f"\n  pantheonplus_cov_stat_sys.npz:")
    print(f"    Shape: {C.shape}")
    print(f"    Symmetric: {np.allclose(C, C.T)}")
    eigvals = np.linalg.eigvalsh(C)
    print(f"    Eigenvalues: min={eigvals.min():.4e}, max={eigvals.max():.2e}")
    print(f"    Positive definite: {eigvals.min() > 0}")

    # Load splits
    with open(base / "pantheonplus_splits.json") as f:
        splits = json.load(f)
    print(f"\n  pantheonplus_splits.json:")
    for key, val in splits.items():
        print(f"    {key}: N={val['N']}, z=[{val['z_min']:.4f}, {val['z_median']:.4f}, {val['z_max']:.4f}]")


def validate_bao():
    """Validate BAO data."""
    print("\n" + "=" * 60)
    print("BAO DR16")
    print("=" * 60)

    base = Path("data_ignored/probes/bao_dr16")

    # Load CSV
    df = pd.read_csv(base / "bao.csv")
    print(f"  bao.csv:")
    print(f"    N = {len(df)}")
    print(f"    z: min={df['z'].min():.2f}, median={df['z'].median():.2f}, max={df['z'].max():.2f}")
    print(f"    Tracers: {df['tracer'].unique().tolist()}")
    print(f"    Observables: {df['observable'].unique().tolist()}")

    # Load covariance
    data = np.load(base / "bao_cov.npz")
    C = data['C']
    print(f"\n  bao_cov.npz:")
    print(f"    Shape: {C.shape}")
    print(f"    Symmetric: {np.allclose(C, C.T)}")
    eigvals = np.linalg.eigvalsh(C)
    print(f"    Eigenvalues: min={eigvals.min():.4e}, max={eigvals.max():.2e}")
    print(f"    Positive definite: {eigvals.min() > 0}")


def validate_tdcosmo():
    """Validate TDCOSMO data."""
    print("\n" + "=" * 60)
    print("TDCOSMO/H0LiCOW")
    print("=" * 60)

    base = Path("data_ignored/probes/tdcosmo")

    # Load CSV
    df = pd.read_csv(base / "td.csv")
    print(f"  td.csv:")
    print(f"    N = {len(df)} lenses")
    print(f"    z_lens: min={df['z_lens'].min():.3f}, median={df['z_lens'].median():.3f}, max={df['z_lens'].max():.3f}")
    print(f"    z_source: min={df['z_source'].min():.3f}, median={df['z_source'].median():.3f}, max={df['z_source'].max():.3f}")
    print(f"    D_dt (Mpc): min={df['D_dt'].min():.0f}, max={df['D_dt'].max():.0f}")

    # Check for D_d availability
    n_with_dd = df['D_d'].notna().sum()
    print(f"    Lenses with D_d: {n_with_dd}/{len(df)}")


def main():
    print("\n" + "#" * 60)
    print("# PROBE DATA VALIDATION")
    print("#" * 60)

    validate_pantheonplus()
    validate_bao()
    validate_tdcosmo()

    print("\n" + "=" * 60)
    print("VALIDATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
