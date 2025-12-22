# Data Readiness Summary

All cosmological probe datasets are validated and ready for f0 inference testing.

## Directory Structure

```
data_ignored/probes/
├── fb_reference.csv                  # Planck 2018 baryon fraction
├── pantheonplus/
│   ├── pantheonplus.csv              # → symlink to pantheonplus_mu.csv
│   ├── pantheonplus_mu.csv           # Distance moduli (N=1701)
│   ├── covariance.npz                # → symlink to pantheonplus_cov_stat_sys.npz
│   ├── pantheonplus_cov_stat_sys.npz # Full STAT+SYS covariance (1701×1701)
│   └── pantheonplus_splits.json      # Index splits by redshift
├── bao_dr16/
│   ├── bao.csv                       # BAO measurements (N=15)
│   └── bao_cov.npz                   # Approximate covariance (15×15)
└── tdcosmo/
    └── td.csv                        # Time-delay distances (N=7)
```

## Dataset Summaries

### A) Pantheon+ Supernovae

| Property | Value |
|----------|-------|
| File | `pantheonplus/pantheonplus.csv` |
| N | 1701 SNe Ia |
| z range | [0.0012, 2.2613] |
| z median | 0.1636 |
| Columns | `z`, `mu_obs`, `mu_err_diag` |
| Units | mu_obs = **distance modulus (magnitudes)** |
| Covariance | 1701×1701, symmetric, positive-definite (min λ = 8.0×10⁻⁴) |
| NaN | 0 |

**Column Schema:**
- `z`: CMB-frame redshift (zCMB from Pantheon+SH0ES)
- `mu_obs`: Distance modulus in magnitudes (MU_SH0ES column from raw data)
- `mu_err_diag`: Diagonal uncertainty √(C_ii) in magnitudes

**Conversion note:** No conversion needed. Pipeline should compute μ_theory = 5 log₁₀(D_L/10 pc) and compare directly to mu_obs.

### B) BAO DR16

| Property | Value |
|----------|-------|
| File | `bao_dr16/bao.csv` |
| N | 15 measurements |
| z range | [0.15, 2.33] |
| Columns | `z`, `observable`, `value`, `sigma`, `tracer`, `paper_tag` |
| Covariance | 15×15, symmetric, positive-definite (min λ = 2.4×10⁻²) |
| NaN | 0 |

**Observable counts:**
| Type | Count |
|------|-------|
| DV/rd | 1 |
| DM/rd | 7 |
| DH/rd | 7 |

**Tracer breakdown:**
- MGS (z=0.15): DV/rd
- BOSS Galaxy (z=0.38, 0.51): DM/rd + DH/rd
- eBOSS LRG (z=0.70): DM/rd + DH/rd
- eBOSS ELG (z=0.85): DM/rd + DH/rd
- eBOSS QSO (z=1.48): DM/rd + DH/rd
- Lya-Lya, Lya-QSO (z=2.33): DM/rd + DH/rd

### C) TDCOSMO Time-Delay Lenses

| Property | Value |
|----------|-------|
| File | `tdcosmo/td.csv` |
| N | 7 lenses |
| z_lens range | [0.30, 0.70] |
| z_source range | [0.70, 2.40] |
| Columns | `lens_id`, `z_lens`, `z_source`, `D_dt`, `sigma_D_dt`, `D_d`, `sigma_D_d`, `paper_tag` |
| Units | D_dt, D_d in **Mpc** |

**Individual lenses:**
| Lens | z_lens | z_source | D_dt (Mpc) | D_d (Mpc) |
|------|--------|----------|------------|-----------|
| B1608+656 | 0.60 | 1.40 | 5156 ± 236 | 1228 ± 193 |
| RXJ1131-1231 | 0.30 | 0.70 | 1740 ± 155 | 804 ± 226 |
| HE0435-1223 | 0.50 | 1.70 | 2707 ± 183 | N/A |
| SDSS1206+4332 | 0.70 | 1.80 | 5769 ± 589 | 1805 ± 564 |
| WFI2033-4723 | 0.70 | 1.70 | 4784 ± 399 | N/A |
| PG1115+080 | 0.30 | 1.70 | 1470 ± 137 | N/A |
| DES0408-5354 | 0.60 | 2.40 | 3382 ± 146 | 1711 ± 328 |

**Note:** 3 lenses lack D_d measurements (only D_dt available).

### D) Baryon Fraction Reference

| Property | Value |
|----------|-------|
| File | `fb_reference.csv` |
| Columns | `fb_mean`, `fb_sigma`, `source_tag` |
| Value | f_b = 0.1571 ± 0.0017 |
| Source | Planck 2018 TT,TE,EE+lowE+lensing (arXiv:1807.06209) |

Derived from: f_b = Ω_b h² / (Ω_b h² + Ω_c h²) = 0.02237 / 0.14237

## Validation Status

| Dataset | N | z range | Units OK | NaN | Cov PD |
|---------|---|---------|----------|-----|--------|
| Pantheon+ | 1701 | 0.001–2.26 | ✓ (mag) | 0 | ✓ |
| BAO DR16 | 15 | 0.15–2.33 | ✓ (rd⁻¹) | 0 | ✓ |
| TDCOSMO | 7 | 0.30–0.70 | ✓ (Mpc) | 0* | N/A |
| f_b ref | 1 | — | ✓ | 0 | — |

*3 lenses have NaN for D_d/sigma_D_d (expected, not all lenses have D_d constraints)

## Caveats

1. **Pantheon+ covariance** is the full STAT+SYS matrix. For fits, invert once and cache.
2. **BAO covariance** uses approximate correlations (r_DM_DH ≈ -0.4 within same tracer). Cross-tracer correlations set to 0.
3. **TDCOSMO uncertainties** are symmetric approximations of asymmetric posteriors.
4. **f_b reference** is for comparison only; not used as a prior in fitting.

## Git Status

- `data_ignored/` is in `.gitignore` (line 80)
- Data files are NOT tracked in git
- Symlinks created for backward-compatible paths

---
*Generated: 2024-12-22*
