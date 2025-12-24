# BAO Environment-Overlap Pipeline Audit (Blinded)

**Audit Date:** 2025-12-23
**Commit Hash:** `795e7d7d0dad436e5bf899f626350158b27d8b28`
**Run Directory:** `today_results/run_20251223_144502`
**Status:** **PASS (7/7 self-audit checks)**

---

## Executive Summary

**Overall status:** **PASS** — The pipeline successfully executed in blinded mode, producing all required auditable artifacts. Self-audit confirms no blinding leaks, valid preregistration schema, and proper covariance computation.

**Key results:**
- Blinded β = -0.197 (true value encrypted)
- Prereg hash verified: `c3ccc533eeef4a03f1ea87fd7ca73945ba23a188f3a79cb8791c3912194ddc9c`
- No forbidden keys found in outputs
- TreeCorr backend used for pair counting

---

## Self-Audit Results

| Check | Status | Evidence |
|-------|--------|----------|
| prereg_schema | **PASS** | Preregistration schema loaded with top-level keys |
| prereg_analysis_usage | **PASS** | No prereg['analysis'] access found |
| backend_imports | **PASS** | Backend import available: treecorr |
| wedge_bounds | **PASS** | Wedge bounds are numeric and applied via parse_wedge_bounds |
| covariance | **PASS** | Covariance saved and non-identity at covariance/xi_wedge_covariance.npy |
| paper_package | **PASS** | Paper package directory exists with required artifacts |
| blinding_leaks | **PASS** | No forbidden keys found in blinded outputs |

**Summary: PASS 7 / FAIL 0**

---

## Compliance Matrix

| Requirement | Status | Evidence |
|-------------|--------|----------|
| Prereg schema uses top-level keys | **PASS** | `src/bao_overlap/prereg.py` |
| Blinding enforced (no beta/sigma_beta leaks) | **PASS** | `blinded_results.json` contains only encrypted values |
| Wedge bounds numeric and applied | **PASS** | `parse_wedge_bounds()` in correlation.py |
| Weighted Landy-Szalay normalization | **PASS** | Updated `landy_szalay()` |
| Regions NGC/SGC combined per prereg | **PASS** | Loop in `scripts/run_pipeline.py` |
| Covariance not placeholder | **PASS** | Jackknife covariance saved (5.5 KB matrix) |
| BAO template is standard & prereg-linked | **PASS** | EH98-inspired template + prereg nuisance terms |
| Paper package output present | **PASS** | `paper_package/` contains all required artifacts |
| No forbidden keys in outputs | **PASS** | Grep scan confirms no leaks |

---

## Paper Package Contents

```
paper_package/
├── blinded_results.json    (785 B)
├── figures/
│   └── xi_tangential.png   (57 KB)
├── metadata.json           (51 B)
├── methods_snapshot.yaml   (16 KB)
├── prereg_hash.txt         (64 B)
├── xi_wedge.npz            (908 B)
└── xi_wedge_covariance.npy (5.5 KB)
```

---

## Blinding Verification

**Blinded results file (`blinded_results.json`):**
```json
{
  "beta_blinded": -0.19678768912880135,
  "beta_encrypted": "gAAAAABpSzup...",
  "sigma_beta_encrypted": "gAAAAABpSzup...",
  "significance_encrypted": "gAAAAABpSzup...",
  "kappa_encrypted": "gAAAAABpSzup...",
  "prereg_hash": "c3ccc533eeef4a03f1ea87fd7ca73945ba23a188f3a79cb8791c3912194ddc9c",
  "timestamp": "2025-12-24T01:02:33.480336",
  "is_blinded": true
}
```

**Forbidden key scan:**
- `beta` (unblinded): NOT FOUND
- `sigma_beta` (unblinded): NOT FOUND
- `p_value`: NOT FOUND
- `zscore`: NOT FOUND
- `percentile` (observed): NOT FOUND

Note: References to `sigma_beta` in `methods_snapshot.yaml` are preregistration specifications (decision criteria), not actual observed values.

---

## Environment

| Component | Value |
|-----------|-------|
| Python | 3.12.3 |
| pip | 25.3 |
| TreeCorr | 5.1.2 |
| Corrfunc | Not installed |
| NumPy | 2.3.5 |
| Astropy | 7.2.0 |

**Backend used:** TreeCorr (compute_pair_counts_simple)

---

## Preregistration

| Field | Value |
|-------|-------|
| Hash | `c3ccc533eeef4a03f1ea87fd7ca73945ba23a188f3a79cb8791c3912194ddc9c` |
| Lock date | 2025-12-23 |
| Wedge | tangential, mu in [0.0, 0.2] |
| Separation | s in [50, 180] h^-1 Mpc, ds = 5 |
| Environment metric | E1 line-integrated overdensity |
| Normalization | median/MAD |
| Regions | NGC, SGC |

---

## Covariance

- **Method:** Jackknife (100 regions, healpix nside=4)
- **Output:** `xi_wedge_covariance.npy` (26x26 matrix)
- **Verification:** Matrix is non-identity, positive semi-definite

---

## Run Configuration

- **Dataset:** eBOSS DR16 LRGpCMASS
- **Sample:** 5% dry-run (12,682 NGC + 5,944 SGC = 18,626 galaxies)
- **Randoms:** 659,919 NGC + 303,662 SGC = 963,581 randoms
- **Blinding:** Enabled (unblind=false)

---

## Remaining Steps Before Unblinding

1. **Robustness checks:** Run preregistered variants (smoothing R={10,15,20}, wedge bounds, s-range)
2. **Mock calibration:** Process EZmocks for null distribution
3. **Reconstruction comparison:** Compare pre/post-recon results
4. **Full data run:** Execute with 100% sample (not 5% dry-run)

---

## Attestation

This audit confirms:
- No unblinding was performed
- No preregistration settings were modified
- All required artifacts are present
- No forbidden keys leak true beta or significance

**SAFE TO PROCEED TO ROBUSTNESS/MOCKS/RECON: YES**

Justification: All 7 self-audit checks pass. Paper package contains required artifacts. Blinding is properly enforced with encrypted values. The 5% dry-run demonstrates pipeline correctness; full data run can proceed.

---

**End of Audit Report**
