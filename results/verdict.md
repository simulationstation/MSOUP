# Msoup Closure Model: Verdict Report

Generated: 2025-12-19 22:06

## Data Sources

- **Rotation Curves**: REAL SPARC data (Lelli et al. 2016)
  - Source: Zenodo archive (https://zenodo.org/records/16284118)
  - Selection: v_max < 80 km/s, n_points >= 8, quality <= 2
  - Galaxies used: 33

- **Lensing Constraints**: Per-paper checks (no combined likelihood)
  - See results/lensing_constraints_table.csv for details

## Model Commitments

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| Suppression strength | c_*² | (km/s)² |
| Turn-on amplitude | ΔM | dimensionless |
| Transition redshift | z_t | - |
| Transition width | w | - |

**Fixed**: κ = 1.0 (NOT fitted)

## CDM Baseline

| Metric | Value |
|--------|-------|
| Total χ² | 841.1 |
| χ²/DOF | 2.304 |

## Parameter Scan Results

| c_*² | P_min/P_CDM | M_hm (M_sun) | Δχ² | Lensing |
|------|-------------|--------------|-----|---------|
| 0 | 1.000 | none | +0.0 | ✓ |
| 25 | 0.039 | 2.6e+08 | +600.4 | ✗ |
| 50 | 0.020 | 7.5e+08 | +598.5 | ✗ |
| 100 | 0.010 | 2.1e+09 | +596.9 | ✗ |
| 200 | 0.005 | 6.0e+09 | +595.2 | ✗ |
| 400 | 0.003 | 1.7e+10 | +583.9 | ✗ |
| 800 | 0.001 | 4.8e+10 | +536.1 | ✗ |

## Lensing Constraints Applied

| Paper | Reported | M_hm Limit | Mapping |
|-------|----------|------------|---------|
| Gilman2020_FluxRatios | m_WDM=5.2 keV | 4.1e+07 M_sun | Schneider et al. 2012 thermal relic rela |
| Hsueh2020_SHARP | m_WDM=4.0 keV | 9.9e+07 M_sun | Schneider et al. 2012 thermal relic rela |
| Vegetti2018_Detection | M_sub=1000000000.0 M_sun | 1.0e+09 M_sun | Direct mass detection |
| Enzi2021_Combined | m_WDM=6.3 keV | 2.2e+07 M_sun | Schneider et al. 2012 thermal relic rela |

**Mapping formula**: M_hm = 10^10 × (m_WDM/keV)^(-3.33) M_sun (Schneider et al. 2012)

## Falsifier Status

| Falsifier | Status | Notes |
|-----------|--------|-------|
| F1: SPARC improvement (Δχ² >= 5) | TRIGGERED | Best Δχ² = 600.4 |
| F2: Lensing-compatible region | NONE | |
| F3: Multi-probe tension | TRIGGERED | |

## Conclusion

**VERDICT: MULTI_PROBE_TENSION**

Model shows tension: SPARC-improving region violates lensing constraints

---

## Reproducibility

All results generated from:
- Real SPARC data from Zenodo
- Per-paper lensing constraints with explicit WDM↔M_hm mapping
- Code: validate_real.py

To reproduce: `python validate_real.py`
