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

## Mverse Channel Pipeline
This repo also includes the `mverse_channel` package, which implements a falsifiable simulation + analysis pipeline for a coherence/topology/boundary-condition gated extra correlation channel hypothesis in superconducting/cavity EM systems.

### Scientific Goal (Operationally Defined)
- Simulate a baseline open quantum system (two-mode Lindblad model + measurement chain).
- Simulate an extended model with a gated hidden-sector correlation channel.
- Fit and compare baseline vs extended models on synthetic data.
- Quantify detectability with power analysis and pre-registered anomaly scoring.

### Install
```bash
pip install -e .
```
The pipeline uses QuTiP for Lindblad simulations. Ensure `qutip` is available in your environment.

### Demo End-to-End Run
```bash
python -m mverse_channel.cli end-to-end --preset demo
```

The demo writes `outputs/mverse_demo/data.npz`, `outputs/mverse_demo/metrics.json`, `outputs/mverse_demo/figures/`, and `outputs/mverse_demo/report.md`.

### CLI Examples
```bash
python -m mverse_channel.cli simulate --config configs/mverse_channel_example.json --out outputs/mverse_run
python -m mverse_channel.cli sweep --config configs/mverse_channel_sweep.json --out outputs/mverse_sweep
python -m mverse_channel.cli report --data-dir outputs/mverse_run --out outputs/mverse_run/report.md
```

### Detection Criteria
Detection in sweeps requires both a positive anomaly score and coherence peak above fixed thresholds, and model comparison outputs AIC/BIC deltas for baseline vs extended models.

### Extending Phenomenologies
Add new hidden-channel phenomenologies in `src/mverse_channel/physics/extended_model.py` by extending the `HiddenChannelConfig` and switching on the `phenomenology` field.
