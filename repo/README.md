# BAO Overlap Pipeline

Production-grade, reproducible pipeline for preregistered environment-overlap BAO analysis. The pipeline is fully configurable, deterministic, auditable, and supports blinding and mock-calibrated inference.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .

python scripts/run_pipeline.py --config configs/runs/eboss_lrgpcmass_default.yaml
```

## Determinism and Auditability
- All random seeds are defined in `configs/preregistration.yaml`.
- Every stage saves intermediate products in `outputs/` with metadata JSONs.
- All thresholds, binning, and modeling parameters are loaded from preregistration.

## Blinding
The pipeline never computes or prints the observed β significance unless `blinding.unblind: true` is set in the run config. Outputs before unblinding include only blinded placeholders.

## HPC Usage
- Use `--n-workers` to set multiprocessing workers.
- Stage-level execution is supported with `scripts/run_stage.py`.

## Repository Layout
```
repo/
  configs/
    preregistration.yaml
    datasets.yaml
    runs/
  src/bao_overlap/
  scripts/
  tests/
  outputs/
```

## Example Dry Run
```bash
python scripts/run_pipeline.py --config configs/runs/eboss_lrgpcmass_default.yaml --dry-run
```

## Notes
- Correlation backend defaults to a lightweight numpy implementation, but can be configured to use pycorr/Corrfunc when available.
- This repository is a scaffold with conservative defaults and explicit checkpoints for scientific reproducibility.

## MSoup/Mverse Neutrality Research Suite
This repo also includes a self-contained research codebase under `src/msoup` for testing the MSoup/Mverse neutrality target with XOR-SAT constraints.

### Quickstart
```bash
pip install -e .
python scripts/run_all.py
```

The run writes `outputs/REPORT.md`, `outputs/REPORT.json`, and optional figures to `outputs/figures/`.
