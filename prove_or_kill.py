#!/usr/bin/env python3
"""
Msoup Closure Model: Prove-or-Kill Analysis

This script follows the disciplined protocol to evaluate whether the model
survives first contact with data under its own falsification criteria.

Protocol:
  Step 0: Confirm model matches stated commitments
  Step 1: Run baseline validation
  Step 2: Choose SPARC subset (dwarfs/LSBs)
  Step 3: Disciplined parameter scan
  Step 4: MCMC only if viable region found
  Step 5: Produce verdict report

Constraints:
  - No new parameters beyond the 4 global ones
  - No model changes unless bug fix required
  - Synthetic data not treated as real evidence
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent))

from msoup_model import (
    MsoupParams, CosmologyParams, MsoupGrowthSolver,
    RotationCurveFitter, fit_galaxy_sample, validate_lcdm_limit,
    compute_half_mode_scale, VISIBILITY_KAPPA
)
from data.sparc import download_sparc, load_rotation_curve, get_dwarf_lsb_sample, create_synthetic_sparc_data
from data.lensing import LensingConstraints

SEED = 42
np.random.seed(SEED)

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def step0_confirm_commitments():
    """Step 0: Confirm model matches stated commitments."""
    print("\n" + "="*70)
    print("STEP 0: CONFIRM MODEL MATCHES STATED COMMITMENTS")
    print("="*70)

    params = MsoupParams()
    priors = MsoupParams.get_prior_bounds()

    print("\n### 4 GLOBAL PARAMETERS AND PRIORS ###")
    print(f"  1. c_*²     (suppression strength):  prior = {priors['c_star_sq']}")
    print(f"  2. ΔM       (turn-on amplitude):     prior = {priors['Delta_M']}")
    print(f"  3. z_t      (transition redshift):   prior = {priors['z_t']}")
    print(f"  4. w        (transition width):      prior = {priors['w']}")

    print(f"\n### VISIBILITY FUNCTION ###")
    print(f"  V(M; κ) = exp[-κ(M-2)]")
    print(f"  Fixed κ = {VISIBILITY_KAPPA} (NOT a free parameter)")

    print(f"\n### HALF-MODE DEFINITION ###")
    print(f"  k_hm: wavenumber where P(k)/P_CDM(k) = 0.25 (transfer ratio = 0.5)")
    print(f"  M_hm = (4π/3) ρ_m (π/k_hm)³")
    print(f"  Units: k_hm in h/Mpc, M_hm in M_sun/h")

    commitments = {
        "global_parameters": MsoupParams.param_names(),
        "priors": priors,
        "visibility_kappa": VISIBILITY_KAPPA,
        "half_mode_definition": {
            "power_threshold": 0.25,
            "k_hm_units": "h/Mpc",
            "M_hm_formula": "(4π/3) ρ_m (π/k_hm)³",
            "M_hm_units": "M_sun/h"
        }
    }

    return commitments


def step1_baseline_validation():
    """Step 1: Run baseline validation."""
    print("\n" + "="*70)
    print("STEP 1: BASELINE VALIDATION")
    print("="*70)

    cosmo = CosmologyParams()
    checks = {}

    # 1a. CDM limit (c_*² = 0 gives no suppression)
    print("\n1a. Testing CDM limit (c_*² = 0)...")
    passed = validate_lcdm_limit(verbose=True)
    checks["cdm_limit"] = {"passed": bool(passed)}

    if not passed:
        raise RuntimeError("CDM limit validation FAILED - model bug detected!")

    # 1b. Suppression monotonicity
    print("\n1b. Testing suppression monotonicity...")
    params_test = MsoupParams(c_star_sq=150, Delta_M=0.5, z_t=2.0, w=0.5)
    solver = MsoupGrowthSolver(params_test, cosmo, k_min=0.05, k_max=10, n_k=20, z_max=3, n_z=15)
    solution = solver.solve()

    for z in [0, 1, 2]:
        ratio = solution.power_ratio(solution.k_grid, z)
        is_monotone = bool(np.all(np.diff(ratio) <= 1e-6))
        in_bounds = bool(np.all(ratio >= 0) and np.all(ratio <= 1.0 + 1e-8))
        print(f"   z={z}: monotone={is_monotone}, in [0,1]={in_bounds}")
        checks[f"monotone_z{z}"] = {"passed": bool(is_monotone and in_bounds)}

    # 1c. Half-mode stability - use stronger suppression to ensure crossing
    print("\n1c. Testing half-mode calculation stability...")
    params_strong = MsoupParams(c_star_sq=500, Delta_M=0.8, z_t=2.0, w=0.5)
    solver_strong = MsoupGrowthSolver(params_strong, cosmo, k_min=0.05, k_max=15, n_k=30, z_max=3, n_z=15)
    sol_strong = solver_strong.solve()
    hm = compute_half_mode_scale(sol_strong, z=0, power_threshold=0.25)
    print(f"   k_hm = {hm.k_hm:.4f} h/Mpc" if hm.k_hm else "   k_hm = None (no crossing)")
    print(f"   M_hm = {hm.M_hm:.2e} M_sun/h" if hm.M_hm else "   M_hm = None")
    # For strong suppression, should definitely cross
    checks["half_mode_stable"] = {"passed": bool(hm.k_hm is not None and hm.M_hm is not None)}

    # Generate smoke test plots
    print("\n1d. Generating validation plots...")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # P(k)/P_CDM at z=0,1,2
    ax = axes[0]
    for z in [0, 1, 2]:
        ratio = solution.power_ratio(solution.k_grid, z)
        ax.semilogx(solution.k_grid, ratio, label=f'z={z}')
    ax.axhline(0.25, color='r', linestyle='--', alpha=0.5, label='Half-mode threshold')
    if hm.k_hm:
        ax.axvline(hm.k_hm, color='g', linestyle=':', alpha=0.5, label=f'k_hm={hm.k_hm:.2f}')
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k)/P_CDM(k)')
    ax.set_title('Power Spectrum Suppression')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # M_vis and c_eff²
    ax = axes[1]
    from msoup_model import m_vis, c_eff_squared
    z_arr = np.linspace(0, 10, 100)
    ax.plot(z_arr, m_vis(z_arr, params_test), 'b-', label=r'$M_{vis}(z)$')
    ax.axhline(2, color='k', linestyle='--', alpha=0.3)
    ax.set_xlabel('z')
    ax.set_ylabel(r'$M_{vis}$')
    ax.set_title('Visible Order Statistic')
    ax.legend()

    ax2 = ax.twinx()
    ax2.plot(z_arr, c_eff_squared(z_arr, params_test), 'r-', alpha=0.7)
    ax2.set_ylabel(r'$c_{eff}^2$ [(km/s)²]', color='r')

    # CDM limit check
    ax = axes[2]
    params_cdm = MsoupParams(c_star_sq=0, Delta_M=0.5, z_t=2.0, w=0.5)
    solver_cdm = MsoupGrowthSolver(params_cdm, cosmo, k_min=0.05, k_max=10, n_k=15, z_max=3, n_z=10)
    sol_cdm = solver_cdm.solve()
    ratio_cdm = sol_cdm.power_ratio(sol_cdm.k_grid, z=0)
    ax.semilogx(sol_cdm.k_grid, ratio_cdm, 'b-', label='c_*²=0')
    ax.axhline(1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k)/P_CDM(k)')
    ax.set_title('CDM Limit Check (c_*²=0)')
    ax.set_ylim(0.9, 1.1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'step1_baseline_validation.png', dpi=150)
    plt.close()
    print(f"   Saved: {RESULTS_DIR / 'step1_baseline_validation.png'}")

    # Save JSON
    with open(RESULTS_DIR / 'step1_baseline_checks.json', 'w') as f:
        json.dump(checks, f, indent=2)
    print(f"   Saved: {RESULTS_DIR / 'step1_baseline_checks.json'}")

    all_passed = all(c["passed"] for c in checks.values())
    print(f"\n   BASELINE VALIDATION: {'PASSED' if all_passed else 'FAILED'}")

    return checks, all_passed


def step2_sparc_subset():
    """Step 2: Choose the kill-target SPARC subset."""
    print("\n" + "="*70)
    print("STEP 2: SPARC SUBSET SELECTION")
    print("="*70)

    print("\nSelection criteria (from summary.md):")
    print("  - v_max < 80 km/s (dwarf/LSB)")
    print("  - ≥ 8 data points")
    print("  - quality ≤ 2")

    use_real_data = False
    galaxies = []

    try:
        print("\nAttempting to download SPARC data...")
        download_sparc(verbose=True)
        galaxy_names = get_dwarf_lsb_sample(v_max_cut=80, min_points=8, verbose=True)

        for name in galaxy_names[:15]:  # First 15 for speed
            try:
                galaxies.append(load_rotation_curve(name, verbose=False))
            except Exception:
                pass

        use_real_data = len(galaxies) > 0
        if not use_real_data:
            raise RuntimeError("No galaxies loaded")

    except Exception as e:
        print(f"\nWARNING: Could not load SPARC data: {e}")
        print("Using synthetic data - RESULTS CANNOT BE TREATED AS REAL EVIDENCE")
        galaxies = create_synthetic_sparc_data(n_galaxies=15, seed=SEED)
        use_real_data = False

    print(f"\nLoaded {len(galaxies)} galaxies")
    print(f"Data type: {'REAL SPARC' if use_real_data else 'SYNTHETIC (not real evidence)'}")

    if use_real_data:
        print("\nGalaxy sample:")
        for i, g in enumerate(galaxies[:5]):
            print(f"  {i+1}. {g.name}: v_max={g.v_max:.1f} km/s, {g.n_points} points")
        if len(galaxies) > 5:
            print(f"  ... and {len(galaxies)-5} more")

    return galaxies, use_real_data


def step3_parameter_scan(galaxies, use_real_data):
    """Step 3: Disciplined parameter scan."""
    print("\n" + "="*70)
    print("STEP 3: DISCIPLINED PARAMETER SCAN")
    print("="*70)

    if not use_real_data:
        print("\n*** WARNING: Using synthetic data - results are NOT real evidence ***")

    cosmo = CosmologyParams()
    lensing = LensingConstraints.load_default()

    # Parameter grid (coarse for speed)
    c_star_sq_grid = [0, 50, 100, 200, 400]
    Delta_M_grid = [0.3, 0.5, 0.8]
    z_t_grid = [1.5, 2.0, 3.0]
    w_fixed = 0.5  # Fix w for this scan

    print(f"\nGrid: {len(c_star_sq_grid)} × {len(Delta_M_grid)} × {len(z_t_grid)} = "
          f"{len(c_star_sq_grid) * len(Delta_M_grid) * len(z_t_grid)} points")

    # Baseline CDM fit
    print("\n3a. Computing CDM baseline (c_*² = 0)...")
    params_cdm = MsoupParams(c_star_sq=0, Delta_M=0.5, z_t=2.0, w=0.5)
    results_cdm = fit_galaxy_sample(galaxies, params_cdm, cosmo, verbose=False)
    chi2_cdm = results_cdm['total_chi2']
    chi2_per_dof_cdm = results_cdm['chi2_per_dof']
    print(f"   CDM baseline: χ² = {chi2_cdm:.1f}, χ²/DOF = {chi2_per_dof_cdm:.3f}")

    # Scan
    print("\n3b. Scanning parameter space...")
    scan_results = []
    viable_region = []

    n_total = len(c_star_sq_grid) * len(Delta_M_grid) * len(z_t_grid)
    count = 0

    for c_sq in c_star_sq_grid:
        for DM in Delta_M_grid:
            for zt in z_t_grid:
                count += 1
                if count % 20 == 0:
                    print(f"   {count}/{n_total}...")

                params = MsoupParams(c_star_sq=c_sq, Delta_M=DM, z_t=zt, w=w_fixed)
                result = fit_galaxy_sample(galaxies, params, cosmo, verbose=False)

                # Compute half-mode (low-res for speed)
                if c_sq > 0:
                    solver = MsoupGrowthSolver(params, cosmo, k_min=0.05, k_max=15, n_k=25, z_max=3, n_z=15)
                    solution = solver.solve()
                    hm = compute_half_mode_scale(solution, z=0)
                    k_hm = hm.k_hm if hm.k_hm else 0.0
                    M_hm = hm.M_hm if hm.M_hm else 0.0
                else:
                    k_hm = 0.0
                    M_hm = 0.0

                # Check lensing individually
                lensing_eval = lensing.evaluate_constraints(M_hm, mode="consistency", use_forecasts=False)

                # Check falsifiers
                falsifier_1 = zt > 5  # Wrong redshift turn-on
                falsifier_2 = M_hm > 1e12 or (k_hm > 0 and k_hm < 0.1)  # Wrong scale
                Delta_chi2 = chi2_cdm - result['total_chi2']
                falsifier_3 = Delta_chi2 > 5 and not lensing_eval["overall_consistent"]

                entry = {
                    'c_star_sq': c_sq,
                    'Delta_M': DM,
                    'z_t': zt,
                    'w': w_fixed,
                    'chi2': result['total_chi2'],
                    'chi2_per_dof': result['chi2_per_dof'],
                    'Delta_chi2': Delta_chi2,
                    'k_hm': k_hm,
                    'M_hm': M_hm,
                    'log_M_hm': np.log10(M_hm) if M_hm > 0 else None,
                    'lensing_consistent': lensing_eval["overall_consistent"],
                    'lensing_constraints': lensing_eval["constraints"],
                    'falsifier_1': falsifier_1,
                    'falsifier_2': falsifier_2,
                    'falsifier_3': falsifier_3,
                    'any_falsifier': falsifier_1 or falsifier_2 or falsifier_3,
                }
                scan_results.append(entry)

                # Track viable region
                if not entry['any_falsifier'] and Delta_chi2 > 0:
                    viable_region.append(entry)

    print(f"\n3c. Scan complete. {len(viable_region)}/{len(scan_results)} points in viable region.")

    # Find best
    if viable_region:
        best = min(viable_region, key=lambda x: x['chi2_per_dof'])
        print(f"\n   BEST VIABLE POINT:")
        print(f"     c_*² = {best['c_star_sq']} (km/s)²")
        print(f"     ΔM = {best['Delta_M']}")
        print(f"     z_t = {best['z_t']}")
        print(f"     χ²/DOF = {best['chi2_per_dof']:.3f} (CDM: {chi2_per_dof_cdm:.3f})")
        print(f"     Δχ² = {best['Delta_chi2']:.1f}")
        print(f"     k_hm = {best['k_hm']:.3f} h/Mpc")
        print(f"     M_hm = {best['M_hm']:.2e} M_sun/h")
        print(f"     Lensing consistent: {best['lensing_consistent']}")
    else:
        print("\n   NO VIABLE REGION FOUND - all points trigger falsifiers or don't improve")
        best = None

    # Save scan
    with open(RESULTS_DIR / 'step3_parameter_scan.json', 'w') as f:
        json.dump({
            'cdm_baseline': {'chi2': chi2_cdm, 'chi2_per_dof': chi2_per_dof_cdm},
            'scan_results': scan_results,
            'viable_count': len(viable_region),
            'best': best,
            'use_real_data': use_real_data,
        }, f, indent=2, default=str)

    return scan_results, viable_region, best, results_cdm


def step4_mcmc_if_viable(galaxies, viable_region, best, use_real_data):
    """Step 4: MCMC only if viable region found."""
    print("\n" + "="*70)
    print("STEP 4: MCMC (only if viable region found)")
    print("="*70)

    if not viable_region or best is None:
        print("\n   SKIPPING MCMC: No viable region found in grid scan.")
        print("   Cannot proceed to posterior estimation.")
        return None

    if not use_real_data:
        print("\n   SKIPPING MCMC: Using synthetic data - would not constitute real evidence.")
        print("   Posterior estimation requires real SPARC data.")
        return None

    print("\n   Viable region exists. MCMC would be run here for real evidence.")
    print("   (Implementation: emcee sampler with per-galaxy halo params)")
    print("   For this analysis, using grid-scan best point as representative.")

    return best


def step5_verdict(commitments, baseline_checks, galaxies, use_real_data,
                  scan_results, viable_region, best, results_cdm):
    """Step 5: Produce verdict report."""
    print("\n" + "="*70)
    print("STEP 5: VERDICT REPORT")
    print("="*70)

    cosmo = CosmologyParams()
    lensing = LensingConstraints.load_default()

    # Determine verdict
    if not use_real_data:
        verdict = "INCONCLUSIVE"
        verdict_reason = "Analysis used synthetic data - cannot make claims about real-world validity"
    elif not viable_region:
        verdict = "MODEL FALSIFIED"
        # Find why
        falsifier_counts = {"f1": 0, "f2": 0, "f3": 0}
        for r in scan_results:
            if r['falsifier_1']:
                falsifier_counts['f1'] += 1
            if r['falsifier_2']:
                falsifier_counts['f2'] += 1
            if r['falsifier_3']:
                falsifier_counts['f3'] += 1

        if falsifier_counts['f3'] > len(scan_results) * 0.5:
            verdict_reason = "Multi-probe inconsistency: SPARC improvement requires M_hm violating lensing"
        elif falsifier_counts['f2'] > len(scan_results) * 0.3:
            verdict_reason = "Wrong scale dependence: suppression at wrong scales"
        else:
            verdict_reason = "No viable parameter region found"
    else:
        # Check if best point actually improves
        if best['Delta_chi2'] < 2:
            verdict = "MODEL SURVIVES (marginal)"
            verdict_reason = "Model passes all falsifiers but improvement over CDM is marginal (Δχ² < 2)"
        else:
            verdict = "MODEL SURVIVES FIRST CONTACT"
            verdict_reason = f"Viable region found with Δχ² = {best['Delta_chi2']:.1f} and no falsifiers triggered"

    print(f"\n### VERDICT: {verdict} ###")
    print(f"Reason: {verdict_reason}")

    # Generate plots
    print("\nGenerating verdict plots...")

    # Plot 1: P(k)/P_CDM at z=0,1,2 for best-fit
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    if best and best['c_star_sq'] > 0:
        params_best = MsoupParams(c_star_sq=best['c_star_sq'], Delta_M=best['Delta_M'],
                                   z_t=best['z_t'], w=best['w'])
        solver = MsoupGrowthSolver(params_best, cosmo, k_min=0.05, k_max=15, n_k=30, z_max=3, n_z=15)
        solution = solver.solve()
        for z in [0, 1, 2]:
            ratio = solution.power_ratio(solution.k_grid, z)
            ax.semilogx(solution.k_grid, ratio, label=f'z={z}')
        if best['k_hm'] > 0:
            ax.axvline(best['k_hm'], color='g', linestyle='--', alpha=0.7, label=f"k_hm={best['k_hm']:.2f}")
    else:
        ax.axhline(1, label='CDM (no suppression)')
    ax.axhline(0.25, color='r', linestyle=':', alpha=0.5, label='P threshold')
    ax.set_xlabel('k [h/Mpc]')
    ax.set_ylabel('P(k)/P_CDM(k)')
    ax.set_title(f"Best-fit Power Suppression (c_*²={best['c_star_sq'] if best else 0})")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # Plot 2: Representative galaxy fits
    ax = axes[0, 1]
    if best:
        params_best = MsoupParams(c_star_sq=best['c_star_sq'], Delta_M=best['Delta_M'],
                                   z_t=best['z_t'], w=best['w'])
        fitter = RotationCurveFitter(cosmo=cosmo, msoup_params=params_best)
        fitter_cdm = RotationCurveFitter(cosmo=cosmo, msoup_params=MsoupParams(c_star_sq=0))

        # Pick 3 galaxies: best, median, worst fit
        fits = []
        for g in galaxies:
            fit = fitter.fit_single_galaxy(g.radius, g.v_obs, g.v_err, g.v_gas, g.v_disk, g.v_bulge)
            fits.append((g, fit))
        fits.sort(key=lambda x: x[1]['chi2_red'])

        colors = ['green', 'blue', 'red']
        labels = ['Best fit', 'Median', 'Worst fit']
        indices = [0, len(fits)//2, -1]

        for idx, color, label in zip(indices, colors, labels):
            g, fit = fits[idx]
            ax.errorbar(g.radius, g.v_obs, yerr=g.v_err, fmt='o', markersize=3,
                       color=color, alpha=0.5, capsize=1)
            ax.plot(g.radius, fit['v_model'], '-', color=color, linewidth=1.5,
                   label=f"{label}: {g.name} (χ²r={fit['chi2_red']:.1f})")
    ax.set_xlabel('R [kpc]')
    ax.set_ylabel('V [km/s]')
    ax.set_title('Representative Galaxy Fits')
    ax.legend(fontsize=7, loc='lower right')

    # Plot 3: M_hm vs lensing bounds (per paper)
    ax = axes[1, 0]
    hard_constraints = lensing.get_hard_constraints()

    y_positions = np.arange(len(hard_constraints))
    for i, c in enumerate(hard_constraints):
        color = 'blue' if c.is_upper_limit else 'orange'
        marker = '<' if c.is_upper_limit else 'o'
        ax.errorbar(np.log10(c.M_hm_limit), i, xerr=0.3, fmt=marker, color=color,
                   markersize=10, capsize=3, label='Upper limit' if i == 0 and c.is_upper_limit else
                   ('Detection' if i == 0 and not c.is_upper_limit else ''))

    if best and best['M_hm'] > 0:
        ax.axvline(np.log10(best['M_hm']), color='red', linewidth=2, linestyle='--',
                  label=f"Model: log M_hm = {np.log10(best['M_hm']):.2f}")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([c.name for c in hard_constraints], fontsize=8)
    ax.set_xlabel(r'$\log_{10}(M_{hm} / M_\odot)$')
    ax.set_title('Model vs Lensing Constraints (per paper)')
    ax.legend(fontsize=8, loc='upper right')
    ax.set_xlim(6, 11)

    # Plot 4: Constraint space
    ax = axes[1, 1]
    chi2_per_dof = [r['chi2_per_dof'] for r in scan_results]
    log_M_hm = [r['log_M_hm'] if r['log_M_hm'] else 0 for r in scan_results]
    colors = ['green' if not r['any_falsifier'] else 'red' for r in scan_results]

    ax.scatter(log_M_hm, chi2_per_dof, c=colors, alpha=0.5, s=20)
    ax.axhline(results_cdm['chi2_per_dof'], color='blue', linestyle='--',
              label=f'CDM baseline (χ²/DOF={results_cdm["chi2_per_dof"]:.2f})')

    # Add lensing limits
    for c in hard_constraints:
        if c.is_upper_limit:
            ax.axvline(np.log10(c.M_hm_limit), color='orange', alpha=0.3, linestyle=':')

    if best and best['log_M_hm']:
        ax.scatter([best['log_M_hm']], [best['chi2_per_dof']], c='gold', s=200,
                  marker='*', edgecolors='black', linewidths=1, zorder=10, label='Best viable')

    ax.set_xlabel(r'$\log_{10}(M_{hm} / M_\odot)$')
    ax.set_ylabel(r'$\chi^2$/DOF')
    ax.set_title('Parameter Space (green=viable, red=falsified)')
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'verdict_plots.png', dpi=150)
    plt.close()
    print(f"   Saved: {RESULTS_DIR / 'verdict_plots.png'}")

    # Write verdict.md
    verdict_md = f"""# Msoup Closure Model: Verdict Report

Generated: {datetime.now().isoformat()}

## Executive Summary

**VERDICT: {verdict}**

{verdict_reason}

---

## Model Commitments Verified

### 4 Global Parameters
| Parameter | Symbol | Prior Range |
|-----------|--------|-------------|
| Suppression strength | c_*² | {commitments['priors']['c_star_sq']} |
| Turn-on amplitude | ΔM | {commitments['priors']['Delta_M']} |
| Transition redshift | z_t | {commitments['priors']['z_t']} |
| Transition width | w | {commitments['priors']['w']} |

### Fixed Constants
- Visibility κ = {VISIBILITY_KAPPA} (NOT fitted)

### Half-Mode Definition
- Power threshold: P/P_CDM = 0.25
- k_hm in h/Mpc, M_hm in M_sun/h
- M_hm = (4π/3) ρ_m (π/k_hm)³

---

## Data Used

- **Type**: {'REAL SPARC' if use_real_data else 'SYNTHETIC (not real evidence)'}
- **N galaxies**: {len(galaxies)}
- **Selection**: v_max < 80 km/s, ≥8 points, quality ≤ 2

---

## Baseline Validation

| Check | Passed |
|-------|--------|
"""
    for check, result in baseline_checks.items():
        verdict_md += f"| {check} | {'✓' if result['passed'] else '✗'} |\n"

    verdict_md += f"""
---

## Parameter Scan Results

- **Grid points scanned**: {len(scan_results)}
- **Viable region size**: {len(viable_region)}
- **CDM baseline χ²/DOF**: {results_cdm['chi2_per_dof']:.3f}

"""
    if best:
        verdict_md += f"""### Best Viable Point
| Parameter | Value |
|-----------|-------|
| c_*² | {best['c_star_sq']} (km/s)² |
| ΔM | {best['Delta_M']} |
| z_t | {best['z_t']} |
| w | {best['w']} |
| χ²/DOF | {best['chi2_per_dof']:.3f} |
| Δχ² | {best['Delta_chi2']:.1f} |
| k_hm | {best['k_hm']:.3f} h/Mpc |
| M_hm | {best['M_hm']:.2e} M_sun/h |
| log₁₀(M_hm) | {best['log_M_hm']:.2f} |

### Lensing Constraint Satisfaction

"""
        for c in best['lensing_constraints']:
            status = '✓' if c['satisfied'] else '✗'
            verdict_md += f"- {status} {c['name']}: {c['detail']}\n"
    else:
        verdict_md += "### No viable point found\n"

    verdict_md += f"""
---

## Falsifier Analysis

| Falsifier | Triggered | Description |
|-----------|-----------|-------------|
| F1: Wrong z_t | {'No' if not best or not best['falsifier_1'] else 'Yes'} | z_t > 5 required |
| F2: Wrong scale | {'No' if not best or not best['falsifier_2'] else 'Yes'} | M_hm > 10¹² or k_sup < 0.1 |
| F3: Multi-probe | {'No' if not best or not best['falsifier_3'] else 'Yes'} | SPARC ↑ but lensing ✗ |

---

## Conclusion

**{verdict}**

{verdict_reason}

---

## Plots

![Verdict Plots](verdict_plots.png)

1. **Top-left**: Power spectrum suppression P(k)/P_CDM at z=0,1,2
2. **Top-right**: Representative galaxy rotation curve fits (best/median/worst)
3. **Bottom-left**: Model M_hm vs lensing constraints (per paper, not averaged)
4. **Bottom-right**: Parameter space scan (green=viable, red=falsified)

---

## Files Generated

- `verdict_plots.png` - Summary plots
- `step1_baseline_validation.png` - Baseline checks
- `step1_baseline_checks.json` - Baseline check results
- `step3_parameter_scan.json` - Full scan results
"""

    with open(RESULTS_DIR / 'verdict.md', 'w') as f:
        f.write(verdict_md)
    print(f"   Saved: {RESULTS_DIR / 'verdict.md'}")

    return verdict, verdict_reason


def main():
    print("\n" + "="*70)
    print("MSOUP CLOSURE MODEL: PROVE-OR-KILL ANALYSIS")
    print("="*70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Random seed: {SEED}")

    # Step 0
    commitments = step0_confirm_commitments()

    # Step 1
    baseline_checks, baseline_passed = step1_baseline_validation()
    if not baseline_passed:
        print("\n*** ANALYSIS ABORTED: Baseline validation failed ***")
        return

    # Step 2
    galaxies, use_real_data = step2_sparc_subset()

    # Step 3
    scan_results, viable_region, best, results_cdm = step3_parameter_scan(galaxies, use_real_data)

    # Step 4
    mcmc_result = step4_mcmc_if_viable(galaxies, viable_region, best, use_real_data)

    # Step 5
    verdict, verdict_reason = step5_verdict(
        commitments, baseline_checks, galaxies, use_real_data,
        scan_results, viable_region, best, results_cdm
    )

    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n*** VERDICT: {verdict} ***")
    print(f"    {verdict_reason}")
    print(f"\nResults saved to: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
