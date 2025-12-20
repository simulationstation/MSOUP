#!/usr/bin/env python3
"""Quick validation of Msoup model - runs in <1 minute."""

import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from msoup_model import (
    MsoupParams, CosmologyParams, MsoupGrowthSolver,
    compute_half_mode_scale, VISIBILITY_KAPPA, validate_lcdm_limit
)
from data.sparc import create_synthetic_sparc_data
from data.lensing import LensingConstraints
from msoup_model.rotation_curves import fit_galaxy_sample

np.random.seed(42)

print("="*60)
print("MSOUP QUICK VALIDATION")
print("="*60)

# Step 0: Model commitments
print("\n[STEP 0] MODEL COMMITMENTS")
print(f"  4 params: {MsoupParams.param_names()}")
print(f"  Fixed κ = {VISIBILITY_KAPPA}")
print(f"  Half-mode: P/P_CDM = 0.25 threshold")

# Step 1: CDM limit
print("\n[STEP 1] CDM LIMIT CHECK")
params_cdm = MsoupParams(c_star_sq=0)
cosmo = CosmologyParams()
solver = MsoupGrowthSolver(params_cdm, cosmo, k_min=0.1, k_max=5, n_k=10, z_max=2, n_z=5)
sol = solver.solve()
ratio = sol.power_ratio(sol.k_grid, z=0)
cdm_ok = np.allclose(ratio, 1.0, atol=0.01)
print(f"  c_*²=0 gives ratio~1: {cdm_ok} (max dev: {np.max(np.abs(ratio-1)):.4f})")

# Step 2: Suppression with c_*² > 0
print("\n[STEP 2] SUPPRESSION CHECK")
params_sup = MsoupParams(c_star_sq=200, Delta_M=0.5, z_t=2.0, w=0.5)
solver2 = MsoupGrowthSolver(params_sup, cosmo, k_min=0.1, k_max=10, n_k=15, z_max=2, n_z=5)
sol2 = solver2.solve()
ratio2 = sol2.power_ratio(sol2.k_grid, z=0)
suppressed = ratio2[-1] < ratio2[0]
print(f"  High-k suppressed: {suppressed} (ratio[0]={ratio2[0]:.3f}, ratio[-1]={ratio2[-1]:.3f})")

hm = compute_half_mode_scale(sol2, z=0)
print(f"  k_hm = {hm.k_hm:.2f} h/Mpc" if hm.k_hm else "  k_hm = None")
print(f"  M_hm = {hm.M_hm:.2e} M_sun" if hm.M_hm else "  M_hm = None")

# Step 3: Lensing check
print("\n[STEP 3] LENSING CONSTRAINTS")
lensing = LensingConstraints.load_default()
M_hm = hm.M_hm if hm.M_hm else 0
eval_result = lensing.evaluate_constraints(M_hm, mode="consistency")
print(f"  M_hm = {M_hm:.2e}")
print(f"  Lensing consistent: {eval_result['overall_consistent']}")
for c in eval_result['constraints'][:3]:
    print(f"    - {c['name']}: {'✓' if c['satisfied'] else '✗'}")

# Step 4: Quick SPARC test (synthetic)
print("\n[STEP 4] ROTATION CURVE FIT (synthetic)")
galaxies = create_synthetic_sparc_data(n_galaxies=5, seed=42)
result_cdm = fit_galaxy_sample(galaxies, params_cdm, cosmo, verbose=False)
result_sup = fit_galaxy_sample(galaxies, params_sup, cosmo, verbose=False)
print(f"  CDM χ²/DOF = {result_cdm['chi2_per_dof']:.3f}")
print(f"  Msoup χ²/DOF = {result_sup['chi2_per_dof']:.3f}")
print(f"  Δχ² = {result_cdm['total_chi2'] - result_sup['total_chi2']:.1f}")

# Verdict
print("\n" + "="*60)
print("QUICK VERDICT")
print("="*60)

falsifier_1 = params_sup.z_t > 5
falsifier_2 = M_hm > 1e12 if M_hm > 0 else False
falsifier_3 = not eval_result['overall_consistent']

print(f"  F1 (z_t > 5): {falsifier_1}")
print(f"  F2 (M_hm > 10^12): {falsifier_2}")
print(f"  F3 (lensing violation): {falsifier_3}")

if falsifier_1 or falsifier_2 or falsifier_3:
    print("\n  *** MODEL SHOWS TENSION ***")
else:
    print("\n  *** MODEL SURVIVES BASIC CHECKS ***")

# Save
results = {
    "cdm_limit_ok": bool(cdm_ok),
    "suppression_works": bool(suppressed),
    "k_hm": float(hm.k_hm) if hm.k_hm else None,
    "M_hm": float(hm.M_hm) if hm.M_hm else None,
    "lensing_ok": eval_result['overall_consistent'],
    "falsifiers": {"f1": falsifier_1, "f2": falsifier_2, "f3": falsifier_3}
}
Path("results").mkdir(exist_ok=True)
with open("results/quick_validate.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved: results/quick_validate.json")
