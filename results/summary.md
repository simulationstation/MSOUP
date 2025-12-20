# Msoup Closure Model: Summary Report

## Executive Summary

This pipeline tests whether a **minimal Msoup closure model** with exactly **4 global parameters** can simultaneously:

1. **Improve fits** to dwarf/LSB galaxy rotation curves (addressing the cusp-core problem)
2. **Remain consistent** with strong lensing substructure constraints (not over-suppressing small halos)

The model implements a "visibility horizon" that suppresses small-scale structure through an effective sound-speed-like term in the growth equation.

---

## Model Specification

### Closure Constraints

| Constraint | Implementation |
|------------|----------------|
| **(C1)** Visibility function | V(M; κ) = exp[-κ(M-2)] with fixed κ=1.0 (reported, not fitted) |
| **(C2)** Visible order statistic | M_vis(z) = 2 + ΔM / (1 + exp[(z - z_t)/w]) |
| **(C3)** Single response channel | c_eff²(z) derived from M_vis via sigmoid mapping |

### Global Parameters (the ONLY free parameters)

| Parameter | Symbol | Description | Prior Range |
|-----------|--------|-------------|-------------|
| **Suppression strength** | c_*² | Effective sound speed squared | 0–1000 (km/s)² |
| **Turn-on amplitude** | ΔM | How much M_vis rises above 2 | 0–2 |
| **Transition redshift** | z_t | When suppression becomes active | 0.5–10 |
| **Transition width** | w | How rapidly suppression activates | 0.05–2 |
| **Visibility decay (fixed)** | κ | Set to 1.0 for all runs; reported for transparency | — |

### Modified Growth Equation

```
δ̈_k + 2H δ̇_k = 4πG ρ_m δ_k − c_eff²(z) k² δ_k
```

The last term causes scale-dependent suppression at high k (small scales).

**Unit conventions**
- k in h/Mpc (internally converted to 1/Mpc using h)
- H(z) in km/s/Mpc
- c_eff² in (km/s)²
- Dimensionless factor in the ODE: (c_eff · k · h / H)²

---

## Data Sources

### SPARC Rotation Curves

- **Source**: http://astroweb.case.edu/SPARC/
- **Archived mirror (preferred)**: Zenodo record 4274930 (`Rotmod_LTG.zip`)
- **Reference**: Lelli et al. (2016), AJ 152, 157
- **DOI**: [10.3847/0004-6256/152/6/157](https://doi.org/10.3847/0004-6256/152/6/157)
- **Selection**: Dwarf/LSB galaxies with v_max < 80 km/s, ≥8 data points, quality ≤ 2
- **Synthetic fallback**: If downloads are skipped (`--skip-download`) or fail, logs and outputs state “using synthetic data” and store that flag in `fit_results.json`.

### Strong Lensing Constraints

Compiled from published analyses:

| Constraint | Type | M_hm limit | Reference |
|------------|------|------------|-----------|
| Gilman+2020 | HARD | < 10^7.6 M⊙ | MNRAS 491, 6077 |
| Hsueh+2020 (SHARP) | HARD | < 10^8.0 M⊙ | MNRAS 492, 3047 |
| Vegetti+2018 | HARD | Detection at 10^9 M⊙ | MNRAS 481, 3661 |
| Enzi+2021 | HARD | < 10^7.8 M⊙ | MNRAS 506, 5848 |
| Roman forecast | FORECAST | < 10^6.5 M⊙ | Projected |
| Rubin forecast | FORECAST | < 10^7.0 M⊙ | Projected |

**Lensing application modes**
- **Conservative consistency check (default)**: each constraint is treated as a bound; per-paper satisfaction is reported in outputs.
- **Hard likelihood**: Gaussian/upper-limit likelihoods aggregated only when explicitly requested and assumptions are documented.

---

## Methodology

### Rotation Curve Fitting

For each galaxy:
- **Baryonic component**: V_bar² = V_gas² + (M/L_disk)·V_disk² + (M/L_bulge)·V_bulge²
- **Dark matter halo**: Cored NFW profile
- **Core radius**: Derived from Msoup suppression scale at halo formation epoch
- **Per-galaxy parameters**: M_200, concentration c, M/L_disk (with priors)
- **Global parameters**: (c_*², ΔM, z_t, w) shared across all galaxies

### Half-Mode Mass and Wavenumber

The primary diagnostic connecting rotation curves to lensing:

- k_hm: wavenumber where P(k)/P_CDM(k) = 0.25
- M_hm: mass scale derived from k_hm via (4π/3) ρ_m (π/k_hm)³

This must satisfy:
- **Lensing upper limits**: M_hm < 10^7.8 M⊙ (approximately)
- **SPARC improvement**: Requires M_hm > 0 (some suppression)

Both k_hm (h/Mpc) and M_hm (M⊙/h) are written to outputs.

---

## Falsification Criteria

The model is considered **falsified** if any of these trigger:

### Falsifier 1: Wrong Redshift Turn-on
- **Trigger**: z_t > 5 required
- **Meaning**: Effect must be active at cosmic noon or earlier, conflicting with early universe observations

### Falsifier 2: Wrong Scale Dependence
- **Trigger**: M_hm > 10^12 M⊙ or k_suppression < 0.1 h/Mpc
- **Meaning**: Suppression at wrong scales, affecting cluster/galaxy formation

### Falsifier 3: Multi-Probe Inconsistency
- **Trigger**: SPARC requires Δχ² > 5 improvement but M_hm violates lensing constraints
- **Meaning**: Cannot satisfy both probes with same parameters

---

## Results Structure

```
results/
├── fit_results.json       # Best-fit parameters and diagnostics
├── parameter_scan.csv     # Full parameter grid results
├── rotation_curve_fits.png # Visual comparison of fits
├── constraint_space.png   # χ²/DOF vs M_hm with lensing limits
└── subhalo_mf.png         # Predicted subhalo MF suppression
```

`fit_results.json` records the fixed visibility κ, k_hm, M_hm, and a per-paper lensing constraint table (not just a single boolean).

---

## Key Equations

### Visibility (C1)
```
V(M; κ) = exp[-κ · max(M-2, 0)]
```

### Visible Order (C2)
```
M_vis(z) = 2 + ΔM / (1 + exp[(z - z_t)/w])
```

### Effective Sound Speed (C3)
```
c_eff²(z) = c_*² · σ((M_vis(z) - 2 - Δ_threshold) / σ_map)
```

where σ(x) = 1/(1+e^(-x)) is the sigmoid function.

### Half-Mode Mass
```
M_hm ≈ (4π/3) ρ_m (π/k_hm)³
```

where k_hm is the wavenumber where P(k)/P_CDM(k) = 0.25.

---

## Running the Pipeline

### Quick Validation (<5 minutes)
```bash
python run_all.py --smoke-test
```

### Full Analysis
```bash
python run_all.py
```

### With Synthetic Data (if SPARC unavailable)
```bash
python run_all.py --skip-download
```

### Run Notebooks
```bash
jupyter notebook analysis/baseline_validation.ipynb
jupyter notebook analysis/sparc_fit_and_lensing_check.ipynb
```

---

## References

1. Lelli, F., McGaugh, S. S., & Schombert, J. M. (2016). "SPARC: Mass Models for 175 Disk Galaxies with Spitzer Photometry and Accurate Rotation Curves". AJ, 152, 157.

2. Gilman, D., et al. (2020). "Warm dark matter chills out: constraints on the halo mass function and the free-streaming length of dark matter with eight quadruple-image strong gravitational lenses". MNRAS, 491, 6077.

3. Vegetti, S., et al. (2018). "Constraining the warm dark matter particle mass with gravitational lensing". MNRAS, 481, 3661.

4. Hsueh, J.-W., et al. (2020). "SHARP - VII. New constraints on the dark matter free-streaming properties and substructure abundance from gravitationally lensed quasars". MNRAS, 492, 3047.

5. Enzi, W., et al. (2021). "Joint constraints on thermal relic dark matter from strong gravitational lensing, the Ly α forest, and Milky Way satellites". MNRAS, 506, 5848.

6. Read, J. I., et al. (2016). "The case for a cold dark matter cusp in Draco". MNRAS, 459, 2573.

---

## Version History

- **v0.1.0** (2025-01-XX): Initial implementation with minimal closure model

---

## Contact

This pipeline implements the theoretical framework described in the Msoup observability horizon hypothesis. For questions about the implementation, see the code documentation.
