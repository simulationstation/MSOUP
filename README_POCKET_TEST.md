# MV-O1B Pocket/Intermittency Test

This package implements a statistically rigorous test for transient pockets and intermittency in global GNSS clock products (IGS CLK/SP3) and ground magnetometers (INTERMAGNET-like). It combines over-dispersion (Fano), window-debiased clustering, and optional propagation geometry to search for domain-wall/clump traversals while respecting per-sensor sensitivity windows.

## What is being tested?

- **Fano (over-dispersion)** — Do event counts across sensors/time exceed Poisson expectations?
- **Window-debiased clustering (`C_excess`)** — Are events clustered in time/geometry beyond what window-conditioned nulls allow?
- **Propagation geometry** — Are arrival times consistent with a coherent front versus shuffled sensors?
- **Cross-modality coherence** — Do GNSS and magnetometers agree?

Kill conditions (**printed in the report**):

- **K1:** `C_excess` consistent with 0 under robustness sweeps (change SNR thresholds, masks, candidate rules).
- **K2:** No propagation geometry consistency vs shuffled arrivals.
- **K3:** No GNSS–magnetometer coherence when both are available.

## Running the pipeline

Prepare data under `data/` (ignored by git). Then run:

```bash
python -m msoup_pocket_test.run --config configs/pocket_default.yaml
```

Flags:

- `--fast` — fast sanity run (fewer nulls/block length) without dropping sensors or time.
- `--unsafe` — opt out of WSL2 safety overrides (not recommended on 12 GB RAM).

Outputs:

- `results/msoup_pocket_test/REPORT.md`
- Cached intermediates under `results/cache/`

## Configuration

See `configs/pocket_default.yaml` for tunable parameters (paths, preprocessing, candidate thresholds, null counts, geometry toggles). The default is scientifically meaningful; the fast mode is explicit.

## Design choices

- **Streaming/batching:** CLK and magnetometer products are processed in time chunks to handle year-scale data on modest hardware.
- **Caching:** Parsed SP3 positions and clock batches are cached under `results/cache/`.
- **Deterministic parallelism:** Random seeds are fixed; multiprocessing is used only in null generation.
- **No silent subsampling:** All sensors/windows are kept; accelerations rely on block bootstrap/FFT-friendly detrending instead of dropping data.

## Hardware safety (12 GB WSL2)

- Default `resources` in `configs/pocket_default.yaml` target a 12 GB WSL2 host: `max_workers=1`, `max_rss_gb=9`, `pair_mode=binned`, `chunk_days=7`.
- Sanity mode uses `resamples_sanity_default=128`; full mode uses `resamples_full_default=512`, both capped by the RSS guard.
- WSL2 is auto-detected and forces the safe defaults unless `--unsafe` is passed.
- If you hit a `RuntimeError` about RSS, lower `resources.resamples_full_default`, increase `chunk_days`, and keep `pair_mode=binned`.

## Tests

Unit tests cover window debiasing, synthetic cluster injections, null behavior, and guard against single-sensor dominance.
