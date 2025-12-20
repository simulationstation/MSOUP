# Msoup Closure Model Pipeline

A reproducible Python pipeline to test the constrained "Msoup + observability horizon" effective model for small-scale matter-structure anomalies.

## Overview

This pipeline tests whether a minimal 4-parameter model can simultaneously:
- **Improve fits** to dwarf/LSB galaxy rotation curves (addressing cusp-core problem)
- **Not violate** strong-lensing substructure constraints

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run smoke test (<5 minutes)
python run_all.py --smoke-test

# Run full analysis
python run_all.py
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

## Data Sources

- **SPARC**: http://astroweb.case.edu/SPARC/ (Lelli et al. 2016)
- **Lensing**: Compiled from Gilman+2020, Vegetti+2018, Hsueh+2020, Enzi+2021

## Testing

```bash
pytest tests/ -v
```

## License

Research code - see individual file headers.
