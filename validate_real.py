#!/usr/bin/env python3
"""
Real SPARC validation of Msoup model.

This script:
1. Downloads and uses REAL SPARC data (no synthetic fallback)
2. Applies dwarf/LSB selection criteria
3. Fits rotation curves with CDM and Msoup models
4. Evaluates per-paper lensing constraints (no combined likelihood)
5. Produces verdict with explicit assumptions documented

Run with: python validate_real.py
"""

import numpy as np
import json
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from msoup_model import (
    MsoupParams, CosmologyParams, MsoupGrowthSolver,
    compute_half_mode_scale, VISIBILITY_KAPPA
)
from data.sparc import get_dwarf_lsb_sample, SPARCDownloadError
from data.lensing import LensingConstraints, wdm_mass_to_M_hm, M_hm_to_wdm_mass
from msoup_model.rotation_curves import fit_galaxy_sample

np.random.seed(42)


def run_validation(verbose: bool = True):
    """Run the full validation pipeline."""

    results = {
        "timestamp": datetime.now().isoformat(),
        "data_source": "REAL_SPARC",
        "model_commitments": {},
        "sparc_analysis": {},
        "lensing_analysis": {},
        "parameter_scan": {},
        "verdict": {}
    }

    # ================================================================
    # STEP 0: Model Commitments
    # ================================================================
    if verbose:
        print("=" * 70)
        print("MSOUP MODEL VALIDATION WITH REAL SPARC DATA")
        print("=" * 70)
        print("\n[STEP 0] MODEL COMMITMENTS")

    results["model_commitments"] = {
        "n_global_params": 4,
        "param_names": MsoupParams.param_names(),
        "fixed_kappa": float(VISIBILITY_KAPPA),
        "half_mode_threshold": 0.25,
        "notes": "κ is fixed at 1.0, NOT fitted"
    }

    if verbose:
        print(f"  4 global params: {MsoupParams.param_names()}")
        print(f"  Fixed κ = {VISIBILITY_KAPPA}")
        print(f"  Half-mode: P/P_CDM = 0.25 threshold")

    # ================================================================
    # STEP 1: Load Real SPARC Data
    # ================================================================
    if verbose:
        print("\n[STEP 1] LOADING REAL SPARC DATA")

    try:
        galaxies, selection_df = get_dwarf_lsb_sample(
            v_max_cut=80.0,
            min_points=8,
            max_quality=2,
            verbose=verbose
        )
    except SPARCDownloadError as e:
        print(f"\nERROR: Could not download SPARC data!")
        print(f"  {e}")
        print("\nValidation INCOMPLETE - no synthetic fallback allowed.")
        results["data_source"] = "DOWNLOAD_FAILED"
        results["verdict"]["status"] = "INCOMPLETE"
        results["verdict"]["reason"] = str(e)
        return results

    n_galaxies = len(galaxies)
    results["sparc_analysis"]["n_galaxies"] = n_galaxies
    results["sparc_analysis"]["selection_criteria"] = {
        "v_max_cut": 80.0,
        "min_points": 8,
        "max_quality": 2
    }

    if n_galaxies == 0:
        print("\nERROR: No galaxies passed selection criteria!")
        results["verdict"]["status"] = "INCOMPLETE"
        results["verdict"]["reason"] = "No galaxies selected"
        return results

    # Save selection
    selection_df.to_csv("results/sparc_real_selection.csv", index=False)

    if verbose:
        print(f"\n  Selected {n_galaxies} dwarf/LSB galaxies")
        print(f"  Saved: results/sparc_real_selection.csv")

    # ================================================================
    # STEP 2: CDM Baseline Fit
    # ================================================================
    if verbose:
        print("\n[STEP 2] CDM BASELINE FIT")

    cosmo = CosmologyParams()
    params_cdm = MsoupParams(c_star_sq=0)

    result_cdm = fit_galaxy_sample(galaxies, params_cdm, cosmo, verbose=False)

    results["sparc_analysis"]["cdm_fit"] = {
        "total_chi2": float(result_cdm["total_chi2"]),
        "total_dof": int(result_cdm["total_dof"]),
        "chi2_per_dof": float(result_cdm["chi2_per_dof"]),
        "n_success": int(result_cdm["n_success"])
    }

    if verbose:
        print(f"  CDM χ²/DOF = {result_cdm['chi2_per_dof']:.3f}")
        print(f"  Total χ² = {result_cdm['total_chi2']:.1f}")
        print(f"  Successful fits: {result_cdm['n_success']}/{n_galaxies}")

    # ================================================================
    # STEP 3: Parameter Scan
    # ================================================================
    if verbose:
        print("\n[STEP 3] PARAMETER SCAN")
        print("  Scanning c_*² to find suppression region...")

    lensing = LensingConstraints.load_default()

    scan_results = []
    c_sq_values = [0, 25, 50, 100, 200, 400, 800]

    for c_sq in c_sq_values:
        params = MsoupParams(c_star_sq=c_sq, Delta_M=0.5, z_t=2.0, w=0.5)

        # Compute suppression
        if c_sq > 0:
            solver = MsoupGrowthSolver(params, cosmo, k_min=0.1, k_max=100, n_k=50, z_max=3, n_z=20)
            sol = solver.solve()
            hm = compute_half_mode_scale(sol, z=0)
            k_hm = hm.k_hm if hm.k_hm else None
            M_hm = hm.M_hm if hm.M_hm else 0.0
            ratio_min = float(sol.power_ratio(sol.k_grid, z=0)[-1])
        else:
            k_hm = None
            M_hm = 0.0
            ratio_min = 1.0

        # Fit rotation curves
        result_msoup = fit_galaxy_sample(galaxies, params, cosmo, verbose=False)
        delta_chi2 = result_cdm["total_chi2"] - result_msoup["total_chi2"]

        # Lensing check
        lensing_result = lensing.evaluate_all(M_hm) if M_hm > 0 else {
            "overall_consistent": True,
            "n_violated": 0
        }

        scan_entry = {
            "c_star_sq": c_sq,
            "ratio_min": ratio_min,
            "k_hm": k_hm,
            "M_hm": M_hm,
            "m_WDM_equiv": M_hm_to_wdm_mass(M_hm) if M_hm > 0 else None,
            "chi2": float(result_msoup["total_chi2"]),
            "delta_chi2": float(delta_chi2),
            "lensing_ok": lensing_result["overall_consistent"],
            "n_lensing_violated": lensing_result.get("n_violated", 0)
        }
        scan_results.append(scan_entry)

        if verbose:
            k_str = f"{k_hm:.1f}" if k_hm else "none"
            M_str = f"{M_hm:.1e}" if M_hm > 0 else "none"
            lens_str = "✓" if scan_entry["lensing_ok"] else "✗"
            print(f"    c_*²={c_sq:4d}: Δχ²={delta_chi2:+5.1f}, M_hm={M_str:>10}, lens={lens_str}")

    results["parameter_scan"]["values"] = scan_results

    # ================================================================
    # STEP 4: Lensing Constraints Summary
    # ================================================================
    if verbose:
        print("\n[STEP 4] LENSING CONSTRAINTS (per-paper)")

    constraint_table = lensing.get_constraint_table()
    results["lensing_analysis"]["constraints"] = constraint_table

    # Save constraints table
    pd.DataFrame(constraint_table).to_csv("results/lensing_constraints_table.csv", index=False)

    if verbose:
        print(f"\n  Constraints used:")
        for c in constraint_table:
            m_wdm = c.get("m_WDM_equivalent_keV")
            M_hm = c.get("M_hm_equivalent_Msun")
            if m_wdm:
                print(f"    - {c['paper']}: m_WDM > {c['reported_value']:.1f} keV → M_hm < {M_hm:.1e} Msun")
            else:
                print(f"    - {c['paper']}: M_sub detection at {M_hm:.1e} Msun")
        print(f"\n  Mapping: M_hm = 10^10 * (m_WDM/keV)^(-3.33) Msun")
        print(f"  Saved: results/lensing_constraints_table.csv")

    # ================================================================
    # STEP 5: Falsifier Analysis
    # ================================================================
    if verbose:
        print("\n[STEP 5] FALSIFIER ANALYSIS")

    # Find best SPARC improvement
    best_improvement = max(scan_results, key=lambda x: x["delta_chi2"])
    sparc_improvement_significant = best_improvement["delta_chi2"] >= 5.0

    # Check if any suppression value passes lensing
    any_lensing_ok = any(r["lensing_ok"] and r["c_star_sq"] > 0 for r in scan_results)

    # Check suppression at large scales (k=0.2 h/Mpc should be unsuppressed)
    large_scale_ok = True
    for r in scan_results:
        if r["c_star_sq"] > 0 and r["ratio_min"] < 0.99:
            # Check if suppression extends to large scales
            if r["k_hm"] and r["k_hm"] < 1.0:
                large_scale_ok = False

    # Multi-probe tension
    multi_probe_tension = False
    if best_improvement["delta_chi2"] > 2.0:  # Some improvement wanted
        # Does the improving region violate lensing?
        improving_regions = [r for r in scan_results if r["delta_chi2"] > 2.0 and r["c_star_sq"] > 0]
        if improving_regions and not any(r["lensing_ok"] for r in improving_regions):
            multi_probe_tension = True

    results["verdict"]["falsifiers"] = {
        "sparc_improvement_significant": sparc_improvement_significant,
        "best_delta_chi2": float(best_improvement["delta_chi2"]),
        "best_c_star_sq": int(best_improvement["c_star_sq"]),
        "lensing_compatible_region_exists": any_lensing_ok,
        "large_scale_suppression_ok": large_scale_ok,
        "multi_probe_tension": multi_probe_tension
    }

    if verbose:
        print(f"\n  Falsifier Status:")
        print(f"    F1 (SPARC Δχ² >= 5): {'✓ PASSED' if sparc_improvement_significant else '✗ NOT TRIGGERED'} (Δχ² = {best_improvement['delta_chi2']:.1f})")
        print(f"    F2 (Lensing-compatible region): {'✓ EXISTS' if any_lensing_ok else '✗ NONE FOUND'}")
        print(f"    F3 (Multi-probe tension): {'✗ TRIGGERED' if multi_probe_tension else '✓ NOT TRIGGERED'}")
        print(f"    F4 (Large-scale OK): {'✓ PASSED' if large_scale_ok else '✗ FAILED'}")

    # ================================================================
    # STEP 6: Final Verdict
    # ================================================================
    if verbose:
        print("\n" + "=" * 70)
        print("VERDICT")
        print("=" * 70)

    # Determine overall status
    if multi_probe_tension:
        verdict_status = "MULTI_PROBE_TENSION"
        verdict_msg = "Model shows tension: SPARC-improving region violates lensing constraints"
    elif not sparc_improvement_significant and not any_lensing_ok:
        verdict_status = "NO_VIABLE_REGION"
        verdict_msg = "No parameter region provides significant SPARC improvement while satisfying lensing"
    elif sparc_improvement_significant and any_lensing_ok:
        verdict_status = "VIABLE"
        verdict_msg = "Model has viable parameter region (needs further MCMC investigation)"
    else:
        verdict_status = "INCONCLUSIVE"
        verdict_msg = "Neither clear tension nor viable region identified"

    results["verdict"]["status"] = verdict_status
    results["verdict"]["message"] = verdict_msg

    if verbose:
        print(f"\n  Status: {verdict_status}")
        print(f"  {verdict_msg}")
        print()

    # ================================================================
    # Save Results
    # ================================================================
    Path("results").mkdir(exist_ok=True)

    with open("results/sparc_fit_summary.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Per-paper lensing check for best SPARC point
    if best_improvement["M_hm"] > 0:
        lensing_check = lensing.evaluate_all(best_improvement["M_hm"])
        with open("results/lensing_check.json", "w") as f:
            json.dump(lensing_check, f, indent=2, default=str)

    if verbose:
        print("Saved:")
        print("  - results/sparc_real_selection.csv")
        print("  - results/sparc_fit_summary.json")
        print("  - results/lensing_constraints_table.csv")
        print("  - results/lensing_check.json")

    return results


def update_verdict_md(results: dict):
    """Update verdict.md with real SPARC results."""

    verdict_content = f"""# Msoup Closure Model: Verdict Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Data Sources

- **Rotation Curves**: REAL SPARC data (Lelli et al. 2016)
  - Source: Zenodo archive (https://zenodo.org/records/16284118)
  - Selection: v_max < 80 km/s, n_points >= 8, quality <= 2
  - Galaxies used: {results['sparc_analysis'].get('n_galaxies', 'N/A')}

- **Lensing Constraints**: Per-paper checks (no combined likelihood)
  - See results/lensing_constraints_table.csv for details

## Model Commitments

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Suppression strength | c_*² | (km/s)² |
| Turn-on amplitude | ΔM | dimensionless |
| Transition redshift | z_t | - |
| Transition width | w | - |

**Fixed**: κ = {results['model_commitments'].get('fixed_kappa', 1.0)} (NOT fitted)

## CDM Baseline

| Metric | Value |
|--------|-------|
| Total χ² | {results['sparc_analysis'].get('cdm_fit', {}).get('total_chi2', 'N/A'):.1f} |
| χ²/DOF | {results['sparc_analysis'].get('cdm_fit', {}).get('chi2_per_dof', 'N/A'):.3f} |

## Parameter Scan Results

| c_*² | P_min/P_CDM | M_hm (M_sun) | Δχ² | Lensing |
|------|-------------|--------------|-----|---------|
"""

    for r in results.get("parameter_scan", {}).get("values", []):
        M_hm_str = f"{r['M_hm']:.1e}" if r['M_hm'] > 0 else "none"
        lens_str = "✓" if r['lensing_ok'] else "✗"
        verdict_content += f"| {r['c_star_sq']} | {r['ratio_min']:.3f} | {M_hm_str} | {r['delta_chi2']:+.1f} | {lens_str} |\n"

    verdict_content += f"""
## Lensing Constraints Applied

| Paper | Reported | M_hm Limit | Mapping |
|-------|----------|------------|---------|
"""

    for c in results.get("lensing_analysis", {}).get("constraints", []):
        verdict_content += f"| {c['paper']} | {c['reported_quantity']}={c['reported_value']} {c['reported_units']} | {c['M_hm_equivalent_Msun']:.1e} M_sun | {c['mapping_used'][:40]} |\n"

    verdict_content += f"""
**Mapping formula**: M_hm = 10^10 × (m_WDM/keV)^(-3.33) M_sun (Schneider et al. 2012)

## Falsifier Status

| Falsifier | Status | Notes |
|-----------|--------|-------|
| F1: SPARC improvement (Δχ² >= 5) | {'TRIGGERED' if results['verdict']['falsifiers'].get('sparc_improvement_significant') else 'NOT MET'} | Best Δχ² = {results['verdict']['falsifiers'].get('best_delta_chi2', 0):.1f} |
| F2: Lensing-compatible region | {'EXISTS' if results['verdict']['falsifiers'].get('lensing_compatible_region_exists') else 'NONE'} | |
| F3: Multi-probe tension | {'TRIGGERED' if results['verdict']['falsifiers'].get('multi_probe_tension') else 'OK'} | |

## Conclusion

**VERDICT: {results['verdict'].get('status', 'UNKNOWN')}**

{results['verdict'].get('message', '')}

---

## Reproducibility

All results generated from:
- Real SPARC data from Zenodo
- Per-paper lensing constraints with explicit WDM↔M_hm mapping
- Code: validate_real.py

To reproduce: `python validate_real.py`
"""

    Path("results/verdict.md").write_text(verdict_content)


if __name__ == "__main__":
    results = run_validation(verbose=True)
    update_verdict_md(results)
    print("\nUpdated: results/verdict.md")
