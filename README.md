# MSOUP: Multiscale Structure and Observability Universality Project

## Two-Gate Interface Model Simulation (Primary)

Monte Carlo simulation testing the CSF/MVerse "two-gate" interface model for gate independence/multiplicative suppression.

### Model Overview

Simulates coupled dynamics on a 2D periodic square lattice:
- **Ising spins** s_i in {+1, -1} - represents matter configuration
- **Edge phases** theta_ij in (-pi, pi] - U(1) gauge field (holonomy)
- **Credit field** c_i in R - explicit eta surrogate (observability)

The simulation tests whether two gate observables are statistically independent:
- **Geometric gate** G_geo = 1 iff max|Phi_C| <= delta (holonomy closure)
- **Neutrality gate** G_neu = 1 iff |m| <= m_tol (magnetization near zero)

Key metric: **rho = p_both / (p_geo * p_neu)**
- rho ~ 1 indicates gate independence (multiplicativity)
- rho != 1 indicates gate correlation

### Quick Start

```bash
# Requirements: Python 3, numpy, matplotlib (no external physics libs)

# Run full simulation with parameter sweeps
python two_gate_simulation.py

# Outputs saved to results/two_gate_sim/:
#   - 1.png: rho vs kappa (holonomy strength)
#   - 2.png: p_both vs (p_geo * p_neu) scatter
#   - 3.png: rho heatmap over (kappa, g) plane
#   - 4.png: |Phi| and m distributions
#   - results.csv: all parameter combinations and gate probabilities
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| L | 16 | Lattice size (L x L) |
| T | 4.0 | Temperature |
| J | 0.3 | Ising coupling |
| kappa | 50.0 | Holonomy closure strength |
| g | 0.5 | Spin-phase coupling |
| delta | 1.0 | Geometric gate threshold |
| m_tol | 0.10 | Neutrality gate threshold |

---

## Msoup Closure Model Pipeline (Secondary)

A reproducible Python pipeline to test the constrained "Msoup + observability horizon" effective model for small-scale matter-structure anomalies.

### Overview

This pipeline tests whether a minimal 4-parameter model can simultaneously:
- **Improve fits** to dwarf/LSB galaxy rotation curves (addressing cusp-core problem)
- **Not violate** strong-lensing substructure constraints

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test (<5 minutes)
python run_all.py --smoke-test

# Run full analysis
python run_all.py

# Run the ejection + decompression simulation (fast smoke)
python -m msoup_ejection_sim.run --smoke
```

## Model Parameters

| Parameter | Symbol | Description |
|-----------|--------|-------------|
| `c_star_sq` | c_*² | Suppression strength (km/s)² |
| `Delta_M` | ΔM | Turn-on amplitude |
| `z_t` | z_t | Transition redshift |
| `w` | w | Transition width |

## Project Structure

```
MSOUP/
├── msoup_model/           # Core model implementation
│   ├── visibility.py      # V(M; κ), M_vis(z), c_eff²(z)
│   ├── growth.py          # Growth ODE solver
│   ├── cosmology.py       # ΛCDM background
│   ├── halo.py            # Halo mass function
│   └── rotation_curves.py # Rotation curve fitting
├── data/                  # Data acquisition
│   ├── sparc.py           # SPARC rotation curves
│   └── lensing.py         # Lensing constraints
├── analysis/              # Jupyter notebooks
│   ├── baseline_validation.ipynb
│   └── sparc_fit_and_lensing_check.ipynb
├── results/               # Output files
│   └── summary.md         # Analysis report
├── tests/                 # Unit tests
├── run_all.py             # Main entry point
└── requirements.txt       # Dependencies
```

## Key Results

After running the pipeline, check `results/` for:
- `fit_results.json` - Best-fit parameters and diagnostics
- `rotation_curve_fits.png` - Visual comparison of fits
- `constraint_space.png` - χ²/DOF vs M_hm with lensing limits

## Msoup ejection + decompression module

An auxiliary toy module lives in `msoup_ejection_sim/` and is calibrated to reproduce the required dual expansion readouts (H0≈67 early anchor, H0≈73 late ladder) via ejection + decompression dynamics.

### CLI usage

```bash
# Smoke run (small grid)
python -m msoup_ejection_sim.run --smoke

# Full calibration (adjust --max-evals, grid, steps as needed)
python -m msoup_ejection_sim.run --calibrate --max-evals 4000 --grid 256 --steps 400

# Run with saved parameters
python -m msoup_ejection_sim.run --params results/msoup_ejection_sim/best_params.json --grid 256 --steps 400
```

Outputs are saved to `results/msoup_ejection_sim/`, including `FINAL_REPORT.md`, best parameter JSON/CSV, and diagnostic figures.

## Data Sources

- **SPARC**: http://astroweb.case.edu/SPARC/ (Lelli et al. 2016)
- **Lensing**: Compiled from Gilman+2020, Vegetti+2018, Hsueh+2020, Enzi+2021

## Testing

```bash
pytest tests/ -v
```

## License

Research code - see individual file headers.
