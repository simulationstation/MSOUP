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

- `--mode sanity` — quick validation with minimal resampling
- `--mode full` — full empirical p-values (default)
- `--mode dry-run` — audit file counts and memory estimates without loading data
- `--unsafe` — opt out of WSL2 safety overrides (not recommended on 12 GB RAM)

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

## Running on 12 GB WSL2 Safely

The pipeline implements multiple safety layers to prevent system freezes on memory-constrained WSL2 environments:

### Memory Guardrails

1. **Hard RSS limit** (`max_rss_gb: 9.0`): Pipeline aborts cleanly if memory exceeds this threshold, writing a partial report.

2. **Streaming I/O**: Files are processed one at a time via iterator-based loading. No bulk loading of all CLK/SP3/mag files.

3. **Online null resampling**: Statistics are computed using Welford's algorithm with O(1) memory per resample (no storing arrays of null values).

4. **Binned clustering** (`pair_mode: binned`): Default O(n) memory scaling instead of O(n²) pairwise comparisons.

### Before Running Full Coverage

1. **Use dry-run mode first** to audit file counts and memory estimates without loading data:
   ```bash
   python -m msoup_pocket_test.run --mode dry-run
   ```

2. **Start with sanity mode** for quick validation:
   ```bash
   python -m msoup_pocket_test.run --mode sanity
   ```

3. **Only then run full mode** if dry-run shows safe estimates:
   ```bash
   python -m msoup_pocket_test.run --mode full
   ```

### Safe Configuration Defaults

The default `configs/pocket_default.yaml` targets 12 GB WSL2:

```yaml
resources:
  max_workers: 1        # Serial processing to avoid memory duplication
  max_rss_gb: 9.0       # Hard ceiling (leave 3 GB for OS/WSL overhead)
  pair_mode: binned     # O(n) clustering (not O(n²))
  chunk_days: 7         # Process data in 1-week chunks
  rss_check_interval_steps: 5  # Check memory every 5 resamples
```

### If Memory Guard Triggers

If you see `MemoryLimitExceeded`, try these mitigations in order:

1. Reduce `null_realizations` (try 64 or 128 instead of 512)
2. Increase `chunk_days` (smaller concurrent data)
3. Ensure `pair_mode = binned` (not `exact`)
4. Use `max_clk_files` and `max_mag_files` limits temporarily
5. Run with `--mode sanity` for validation

### Full Coverage Mode

To run with full dataset coverage (no file caps), first verify streaming is working:

```bash
# 1. Dry-run to check estimates
python -m msoup_pocket_test.run --mode dry-run

# 2. Sanity mode with n_resamples 128
python -m msoup_pocket_test.run --mode sanity

# 3. Full mode (only if above succeeds)
python -m msoup_pocket_test.run --mode full
```

### Abort Handling

If the pipeline aborts due to memory limits, it writes:
- `REPORT.md` with abort reason and last completed chunk
- `summary.json` with partial results and resource snapshot

This allows you to resume from the last successful chunk by adjusting config.

## Tests

Unit tests cover:
- Memory guard triggers (monkeypatched RSS)
- Online null accumulation (no array storage)
- Binned vs exact clustering equivalence
- Window debiasing and synthetic cluster injection
- Single-sensor dominance prevention
- No NaN/Inf in outputs
