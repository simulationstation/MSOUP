#!/usr/bin/env python3
"""
Msoup Closure Model Pipeline: Main Entry Point

This script runs the complete analysis pipeline:
1. Validates the model implementation
2. Downloads SPARC data (or uses synthetic data)
3. Fits rotation curves with CDM and Msoup models
4. Checks lensing constraints
5. Generates summary report

Usage:
    python run_all.py [--smoke-test] [--skip-download]

Options:
    --smoke-test     Run quick validation (<5 minutes)
    --skip-download  Use synthetic data instead of downloading SPARC
"""

import argparse
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from msoup_model import (
    MsoupParams, CosmologyParams, MsoupGrowthSolver,
    RotationCurveFitter, fit_galaxy_sample, validate_lcdm_limit,
    HaloMassFunction
)
from msoup_model.growth import compute_half_mode_mass
from data.sparc import download_sparc, load_rotation_curve, get_dwarf_lsb_sample, create_synthetic_sparc_data
from data.lensing import LensingConstraints

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)


def ensure_results_dir():
    """Create results directory if it doesn't exist."""
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    return results_dir


def run_validation(verbose=True):
    """Phase 0: Validate model implementation."""
    if verbose:
        print("\n" + "="*60)
        print("PHASE 0: MODEL VALIDATION")
        print("="*60)

    # Test LCDM limit
    if verbose:
        print("\n1. Testing LCDM limit (c_*² = 0)...")
    passed = validate_lcdm_limit(verbose=verbose)

    if not passed:
        raise RuntimeError("LCDM limit validation failed!")

    # Test basic model
    if verbose:
        print("\n2. Testing Msoup growth solver...")

    params = MsoupParams(c_star_sq=100, Delta_M=0.5, z_t=2.0, w=0.5)
    cosmo = CosmologyParams()
    solver = MsoupGrowthSolver(params, cosmo, n_k=20, n_z=30)
    solution = solver.solve(verbose=False)

    # Check output
    assert solution.k_grid.shape[0] == 20
    assert solution.z_grid.shape[0] == 30
    assert np.all(solution.D_kz > 0)

    if verbose:
        print("   Growth solver: OK")

    # Test HMF
    if verbose:
        print("\n3. Testing halo mass function...")

    hmf = HaloMassFunction(cosmo=cosmo, growth_solution=solution)
    result = hmf.compute_hmf(z=0, M_min=1e8, M_max=1e12, n_M=20)

    assert len(result.M) == 20
    assert np.all(result.ratio <= 1.01)  # Should be suppressed or equal

    if verbose:
        print("   HMF computation: OK")

    return True


def load_galaxy_data(skip_download=False, n_galaxies=15, verbose=True):
    """Phase 1: Load galaxy data."""
    if verbose:
        print("\n" + "="*60)
        print("PHASE 1: DATA ACQUISITION")
        print("="*60)

    if skip_download:
        if verbose:
            print("\nUsing synthetic SPARC-like data...")
        galaxies = create_synthetic_sparc_data(n_galaxies=n_galaxies, seed=SEED)
        use_real_data = False
    else:
        try:
            if verbose:
                print("\nDownloading SPARC data...")
            download_sparc(verbose=verbose)
            galaxy_names = get_dwarf_lsb_sample(v_max_cut=80, min_points=8, verbose=verbose)
            galaxies = []
            for name in galaxy_names[:n_galaxies]:
                try:
                    galaxies.append(load_rotation_curve(name, verbose=False))
                except Exception:
                    pass
            use_real_data = len(galaxies) > 0
            if not use_real_data:
                raise RuntimeError("No galaxies loaded")
        except Exception as e:
            if verbose:
                print(f"\nFailed to download SPARC: {e}")
                print("Falling back to synthetic data...")
            galaxies = create_synthetic_sparc_data(n_galaxies=n_galaxies, seed=SEED)
            use_real_data = False

    if verbose:
        print(f"\nLoaded {len(galaxies)} galaxies")
        print(f"Data type: {'Real SPARC' if use_real_data else 'Synthetic'}")

    return galaxies, use_real_data


def run_fits(galaxies, verbose=True):
    """Phase 2-3: Fit models and check constraints."""
    if verbose:
        print("\n" + "="*60)
        print("PHASE 2-3: MODEL FITTING")
        print("="*60)

    cosmo = CosmologyParams()
    lensing = LensingConstraints.load_default()
    results_dir = ensure_results_dir()

    # CDM baseline
    if verbose:
        print("\n1. Fitting CDM baseline (NFW)...")
    params_cdm = MsoupParams(c_star_sq=0, Delta_M=0.5, z_t=2.0, w=0.5)
    results_cdm = fit_galaxy_sample(galaxies, params_cdm, cosmo, verbose=verbose)

    if verbose:
        print(f"\n   CDM χ²/DOF = {results_cdm['chi2_per_dof']:.3f}")

    # Parameter scan (coarse for speed)
    if verbose:
        print("\n2. Scanning Msoup parameter space...")

    c_star_sq_grid = [0, 50, 100, 200]
    Delta_M_grid = [0.3, 0.5]
    z_t_grid = [1.5, 2.5]

    best_result = None
    best_chi2 = np.inf
    best_params = None
    best_M_hm = 0

    for c_sq in c_star_sq_grid:
        for DM in Delta_M_grid:
            for zt in z_t_grid:
                params = MsoupParams(c_star_sq=c_sq, Delta_M=DM, z_t=zt, w=0.5)
                result = fit_galaxy_sample(galaxies, params, cosmo, verbose=False)

                # Compute M_hm
                if c_sq > 0:
                    solver = MsoupGrowthSolver(params, cosmo, n_k=25, n_z=40)
                    solution = solver.solve()
                    M_hm = compute_half_mode_mass(solution, z=0)
                else:
                    M_hm = 0

                # Check lensing
                is_ok, _ = lensing.is_consistent(M_hm, use_forecasts=False)

                # Track best lensing-consistent model
                if is_ok and result['chi2_per_dof'] < best_chi2:
                    best_chi2 = result['chi2_per_dof']
                    best_result = result
                    best_params = params
                    best_M_hm = M_hm

    if best_params is None:
        # Fallback to CDM
        best_params = params_cdm
        best_result = results_cdm
        best_M_hm = 0

    if verbose:
        print(f"\n   Best parameters:")
        print(f"     c_*² = {best_params.c_star_sq} (km/s)²")
        print(f"     ΔM = {best_params.Delta_M}")
        print(f"     z_t = {best_params.z_t}")
        print(f"     w = {best_params.w}")
        print(f"\n   Best χ²/DOF = {best_result['chi2_per_dof']:.3f}")
        print(f"   M_hm = {best_M_hm:.2e} M_sun" if best_M_hm > 0 else "   M_hm = 0 (CDM)")

    # Create plot
    if verbose:
        print("\n3. Creating fit comparison plot...")

    fitter_cdm = RotationCurveFitter(cosmo=cosmo, msoup_params=params_cdm)
    fitter_msoup = RotationCurveFitter(cosmo=cosmo, msoup_params=best_params)

    n_plot = min(6, len(galaxies))
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flatten()

    for i, gal in enumerate(galaxies[:n_plot]):
        ax = axes[i]
        ax.errorbar(gal.radius, gal.v_obs, yerr=gal.v_err,
                    fmt='ko', markersize=3, label='Data', capsize=2, alpha=0.7)

        fit_cdm = fitter_cdm.fit_single_galaxy(
            gal.radius, gal.v_obs, gal.v_err,
            gal.v_gas, gal.v_disk, gal.v_bulge
        )
        ax.plot(gal.radius, fit_cdm['v_model'], 'b--', label='CDM')

        fit_msoup = fitter_msoup.fit_single_galaxy(
            gal.radius, gal.v_obs, gal.v_err,
            gal.v_gas, gal.v_disk, gal.v_bulge
        )
        ax.plot(gal.radius, fit_msoup['v_model'], 'r-', linewidth=2, label='Msoup')

        ax.set_xlabel('R [kpc]')
        ax.set_ylabel('V [km/s]')
        ax.set_title(gal.name)
        if i == 0:
            ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(results_dir / 'rotation_curve_fits.png', dpi=150)
    plt.close()

    return {
        'cdm': results_cdm,
        'best': best_result,
        'best_params': best_params,
        'best_M_hm': best_M_hm,
        'lensing': lensing,
    }


def run_decision(fit_results, use_real_data, verbose=True):
    """Phase 4: Prove or kill decision."""
    if verbose:
        print("\n" + "="*60)
        print("PHASE 4: PROVE OR KILL DECISION")
        print("="*60)

    results_dir = ensure_results_dir()
    params = fit_results['best_params']
    M_hm = fit_results['best_M_hm']
    lensing = fit_results['lensing']

    cdm_chi2 = fit_results['cdm']['chi2_per_dof']
    best_chi2 = fit_results['best']['chi2_per_dof']
    Delta_chi2 = fit_results['cdm']['total_chi2'] - fit_results['best']['total_chi2']

    # Check falsifiers
    falsifier_1 = params.z_t > 5
    falsifier_2 = M_hm > 1e12 if M_hm > 0 else False
    is_ok, _ = lensing.is_consistent(M_hm, use_forecasts=False)
    falsifier_3 = Delta_chi2 > 5 and not is_ok

    verdict = 'survives' if not (falsifier_1 or falsifier_2 or falsifier_3) else 'tension'

    if verbose:
        print(f"\n1. Redshift turn-on: z_t = {params.z_t} -> {'CONCERN' if falsifier_1 else 'OK'}")
        print(f"2. Scale dependence: M_hm = {M_hm:.2e} -> {'CONCERN' if falsifier_2 else 'OK'}")
        print(f"3. Multi-probe: Δχ²={Delta_chi2:.1f}, lensing_ok={is_ok} -> {'CONCERN' if falsifier_3 else 'OK'}")
        print(f"\nVERDICT: {verdict.upper()}")

    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'model': 'Msoup closure v0.1.0',
        'data': 'SPARC' if use_real_data else 'synthetic',
        'best_parameters': {
            'c_star_sq': float(params.c_star_sq),
            'Delta_M': float(params.Delta_M),
            'z_t': float(params.z_t),
            'w': float(params.w),
        },
        'results': {
            'cdm_chi2_per_dof': float(cdm_chi2),
            'msoup_chi2_per_dof': float(best_chi2),
            'delta_chi2': float(Delta_chi2),
            'M_hm': float(M_hm) if M_hm > 0 else None,
            'lensing_consistent': is_ok,
        },
        'falsifiers': {
            'wrong_redshift_turnon': falsifier_1,
            'wrong_scale_dependence': falsifier_2,
            'multiprobe_inconsistent': falsifier_3,
        },
        'verdict': verdict,
    }

    with open(results_dir / 'fit_results.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def run_smoke_test():
    """Quick smoke test (<5 minutes)."""
    print("\n" + "="*60)
    print("SMOKE TEST: Quick validation")
    print("="*60)

    results_dir = ensure_results_dir()

    # 1. Validate model
    print("\n1. Model validation...")
    run_validation(verbose=False)
    print("   PASSED")

    # 2. Quick synthetic fit
    print("\n2. Quick synthetic data fit...")
    galaxies = create_synthetic_sparc_data(n_galaxies=5, seed=SEED)

    cosmo = CosmologyParams()
    params = MsoupParams(c_star_sq=100, Delta_M=0.5, z_t=2.0, w=0.5)

    result = fit_galaxy_sample(galaxies, params, cosmo, verbose=False)
    print(f"   χ²/DOF = {result['chi2_per_dof']:.3f}")
    print("   PASSED")

    # 3. Lensing check
    print("\n3. Lensing constraint check...")
    solver = MsoupGrowthSolver(params, cosmo, n_k=20, n_z=30)
    solution = solver.solve()
    M_hm = compute_half_mode_mass(solution, z=0)
    lensing = LensingConstraints.load_default()
    is_ok, msg = lensing.is_consistent(M_hm, use_forecasts=False)
    print(f"   M_hm = {M_hm:.2e} M_sun")
    print(f"   Lensing OK: {is_ok}")
    print("   PASSED")

    # 4. Generate simple plot
    print("\n4. Generating validation plot...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Power spectrum suppression
    ax = axes[0]
    k_grid = solution.k_grid
    for z in [0, 1, 2]:
        ratio = solution.power_ratio(k_grid, z)
        ax.semilogx(k_grid, ratio, label=f'z={z}')
    ax.axhline(1, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k)/P_CDM(k)')
    ax.set_title('Power Spectrum Suppression')
    ax.legend()
    ax.set_ylim(0, 1.2)

    # Example rotation curve
    ax = axes[1]
    gal = galaxies[0]
    fitter = RotationCurveFitter(cosmo=cosmo, msoup_params=params)
    fit = fitter.fit_single_galaxy(
        gal.radius, gal.v_obs, gal.v_err,
        gal.v_gas, gal.v_disk, gal.v_bulge
    )
    ax.errorbar(gal.radius, gal.v_obs, yerr=gal.v_err,
                fmt='ko', markersize=4, label='Data', capsize=2)
    ax.plot(gal.radius, fit['v_model'], 'r-', linewidth=2, label='Model')
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('V [km/s]')
    ax.set_title(f'Example Fit (r_c={fit["r_c"]:.1f} kpc)')
    ax.legend()

    plt.tight_layout()
    plt.savefig(results_dir / 'smoke_test_plots.png', dpi=150)
    plt.close()

    print("   Plot saved to results/smoke_test_plots.png")

    print("\n" + "="*60)
    print("SMOKE TEST PASSED")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description='Msoup Closure Model Pipeline')
    parser.add_argument('--smoke-test', action='store_true',
                        help='Run quick validation (<5 min)')
    parser.add_argument('--skip-download', action='store_true',
                        help='Use synthetic data instead of downloading SPARC')
    args = parser.parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    print("\n" + "="*60)
    print("MSOUP CLOSURE MODEL PIPELINE")
    print("="*60)
    print(f"\nStarted: {datetime.now().isoformat()}")
    print(f"Random seed: {SEED}")

    # Run phases
    run_validation(verbose=True)

    galaxies, use_real_data = load_galaxy_data(
        skip_download=args.skip_download,
        verbose=True
    )

    fit_results = run_fits(galaxies, verbose=True)

    summary = run_decision(fit_results, use_real_data, verbose=True)

    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"\nResults saved to: results/")
    print(f"  - fit_results.json")
    print(f"  - rotation_curve_fits.png")

    return summary


if __name__ == '__main__':
    main()
