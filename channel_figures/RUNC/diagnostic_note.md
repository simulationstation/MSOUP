# Diagnostic Improvements - Summary

**Date**: 2025-12-28

## Issues Addressed

### 1. Coherence Plot Saturation (FIXED)

**Problem**: Coherence plot appeared empty because values saturated at y=1.0 and were hidden by the axis boundary.

**Solution**: Updated `figures.py`:
- Changed `plt.ylim(0, 1)` to `plt.ylim(0, 1.05)` for visibility at y=1.0
- Added markers (`marker="o"`) to ensure points are visible at boundaries
- Added grid for improved readability

### 2. Short Time Series (FIXED)

**Problem**: Demo preset used duration=3.0s and dt=0.1, yielding only 31 samples. This caused:
- Spectral saturation (coherence stuck at ~1.0)
- Only 16 frequency bins
- Poor spectral resolution

**Solution**: Updated `cli.py` demo preset:
- Changed `duration` from 3.0s to 10.0s
- Changed `dt` from 0.1s to 0.05s
- Now yields 201 samples and 65 frequency bins

### 3. Sweep Epsilon Range (FIXED)

**Problem**: Original sweep only covered epsilon [0.0, 0.05, 0.1, 0.2], missing the range where detection power increases.

**Solution**: Extended sweep script with epsilon values [0.0, 0.1, 0.2, 0.3, 0.4, 0.5].

### 4. Recalibration (COMPLETED)

**Problem**: Thresholds calibrated with short time series were not appropriate for longer series.

**Solution**: Re-ran null calibration with new parameters:
- 50 null replicates
- duration=10.0s, dt=0.05s
- New thresholds:
  - `coherence_mag_threshold`: 0.9994 (was ~1.0)
  - `anomaly_score_threshold`: 1.2085

## Results

### Updated Demo Metrics
```
coherence_mag: 0.9983  (vs 0.9999 previously)
freqs: 65              (vs 16 previously)
samples: 201           (vs 31 previously)
```

### Extended Power Sweep
```
ε=0.00: detection=0.00, coh_mag=0.9938, anomaly=1.10
ε=0.10: detection=0.00, coh_mag=0.9939, anomaly=1.10
ε=0.20: detection=0.00, coh_mag=0.9940, anomaly=1.10
ε=0.30: detection=0.00, coh_mag=0.9941, anomaly=1.10
ε=0.40: detection=0.00, coh_mag=0.9942, anomaly=1.10
ε=0.50: detection=0.00, coh_mag=0.9943, anomaly=1.10
```

**Observation**: Detection power remains at 0% across all epsilon values. This indicates:
1. FPR is controlled (good - no false positives at ε=0)
2. The hidden channel effect is too subtle to detect with current metrics
3. coherence_mag shows minimal variation with epsilon (~0.994 across all levels)

## Files Modified

| File | Changes |
|------|---------|
| `src/mverse_channel/reporting/figures.py` | ylim 0-1.05, markers, grid |
| `src/mverse_channel/cli.py` | Demo duration=10s, dt=0.05 |

## New Artifacts

```
/home/primary/MSOUP/channel_figures/
├── mverse_demo_updated/
│   ├── config.json
│   ├── data.npz
│   ├── metrics.json        # 201 samples, 65 freqs
│   ├── report.md
│   └── figures/
│       ├── coherence.png   # Now visible with ylim 1.05
│       └── anomaly.png
├── extended_sweep/
│   ├── power_sweep.json
│   ├── thresholds.json     # Recalibrated
│   └── detection_curve.png
└── diagnostic_note.md      # This file
```

## Remaining Work (Future Iterations)

1. **Increase hidden channel coupling**: Current epsilon_max values may be too subtle. Consider:
   - Higher epsilon_max (>0.5)
   - Lower gating thresholds
   - Alternative phenomenology ("parametric" vs "additive")

2. **Alternative detection metrics**: Current coherence_mag + anomaly_score combination may not be optimal. Consider:
   - Mutual information
   - Transfer entropy
   - Non-linear correlation measures

3. **Metric combination strategy**: Current detection requires BOTH metrics to exceed thresholds. Consider:
   - OR logic instead of AND
   - Weighted combination
   - Machine learning classifier

## Verification

```bash
# All mverse_channel tests pass
python3 -m pytest tests/test_baseline_shapes.py tests/test_cli_smoke.py \
  tests/test_extended_toggle.py tests/test_measurement_chain.py \
  tests/test_metrics.py tests/test_rng_reproducibility.py -v
# Result: 12 passed
```
