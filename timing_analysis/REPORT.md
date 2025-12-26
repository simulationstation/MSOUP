# Timing Analysis Report: Rare-Event Detection in Distributed Clock Networks

**Analysis Date:** 2024-12-26
**Data Source:** IGS Final 30-second Clock Products
**Data URL:** https://igs.bkg.bund.de/root_ftp/IGS/products/

## Executive Summary

| Metric | Value |
|--------|-------|
| **Conclusion** | DETECTED |
| **Observed ε** | 1.01e-02 |
| **Target ε** | 5.33e-05 |
| **Ratio** | 190x above target |
| **Coincidence p-value** | <1e-15 |

**Interpretation:** The analysis detected significantly more timing events than expected under the null hypothesis. The observed event rate is approximately 190x higher than the target rare-event rate of ε ≈ 5.33e-5. Additionally, the number of multi-stream coincidences (1,334) far exceeds the expected count under independence (~2.2), suggesting either:
1. A genuine signal of coordinated timing anomalies
2. Common-mode noise sources (ionosphere, satellite clocks) affecting multiple stations
3. Detection threshold may need adjustment for this application

## Data Summary

### Time Period
- **Start:** 2024-11-22 00:00:00 UTC
- **End:** 2024-11-28 23:59:30 UTC
- **Duration:** 7 days

### Clock Streams
- **Total streams analyzed:** 286 receiver clocks
- **Streams with detected events:** 181
- **Total samples:** 1,067,123

### Downloaded Files
| File | DOY | Size |
|------|-----|------|
| IGS0OPSFIN_20243270000_01D_30S_CLK.CLK | 327 | 12 MB |
| IGS0OPSFIN_20243280000_01D_30S_CLK.CLK | 328 | 12 MB |
| IGS0OPSFIN_20243290000_01D_30S_CLK.CLK | 329 | 12 MB |
| IGS0OPSFIN_20243300000_01D_30S_CLK.CLK | 330 | 12 MB |
| IGS0OPSFIN_20243310000_01D_30S_CLK.CLK | 331 | 12 MB |
| IGS0OPSFIN_20243320000_01D_30S_CLK.CLK | 332 | 12 MB |
| IGS0OPSFIN_20243330000_01D_30S_CLK.CLK | 333 | 12 MB |

## Event Detection

### Method
- **Algorithm:** PELT (Pruned Exact Linear Time) changepoint detection
- **Model:** Piecewise constant (L2 cost)
- **Fallback:** CUSUM for robustness
- **Threshold:** 5σ for both phase and frequency steps

### Results
| Event Type | Count |
|------------|-------|
| Phase steps | 3,098 |
| Frequency steps | 7,731 |
| **Total** | **10,829** |

### Event Rate (ε)
- **Observed ε:** 1.0148e-02 (10,829 events / 1,067,123 samples)
- **95% CI:** [9.96e-03, 1.03e-02]
- **Target ε:** 5.33e-05
- **Ratio:** 190.4x

## Coincidence Analysis

### Method
Multi-stream coincidence detection with Monte Carlo significance testing:
1. Find events occurring within ±2 seconds across ≥3 independent streams
2. Exclude same-site clocks (first 4 characters of station name match)
3. Compare observed count to null expectation (independent Poisson processes)

### Results
| Metric | Value |
|--------|-------|
| Coincidence window | 2.0 seconds |
| Minimum streams | 3 |
| **Observed coincidences** | 1,334 |
| **Expected (null)** | 2.23 |
| **Excess** | 1,331.77 |
| **Significance (σ)** | 891.77 |
| **P-value** | <1e-15 |

## Interpretation

### High Event Rate
The observed event rate (ε ≈ 1%) is much higher than the target rare-event rate (ε ≈ 5.33e-5). This indicates either:
1. The detection threshold is too sensitive for this data type
2. The IGS clock products contain many genuine timing discontinuities (equipment changes, maintenance, etc.)
3. The methodology detects common-mode signals (satellite clock adjustments, ionospheric effects)

### Highly Significant Coincidences
The extreme excess of coincidences (600x expected) and p-value < 1e-15 indicates:
1. **Not independent:** Clock streams are NOT independent - they share common-mode errors
2. **Satellite clock jumps:** All receivers see the same satellite clock adjustments simultaneously
3. **Processing artifacts:** The IGS combined solution may introduce correlated discontinuities

### Caveats
1. **Derived products:** These are processed clock solutions, not raw observations
2. **Common reference:** All clocks are referenced to the same GPS time scale
3. **Correlated noise:** Ionospheric, tropospheric, and satellite errors affect multiple stations

## Conclusion

**DETECTED** - The analysis detects statistically significant coordinated timing events across the global IGS network. However, interpretation requires caution:

- The high event rate likely reflects the data processing method rather than genuine rare events
- Multi-stream coincidences are almost certainly driven by common-mode sources (satellite clocks, reference time scale adjustments)
- This methodology would benefit from:
  1. Raw RINEX observations with independent PPP processing
  2. Filtering of known satellite clock adjustments
  3. Higher detection thresholds appropriate for this noise environment

## Files Generated

- `events/all_events.csv` - All 10,829 detected events
- `events/coincidences.csv` - 1,334 coincident event groups
- `data_ingest/stream_info.csv` - Statistics for 286 clock streams
- `data_ingest/residuals.csv` - Full time series data
- `REPORT.json` - Machine-readable results

## Data Provenance

| Item | Value |
|------|-------|
| Source | IGS (International GNSS Service) |
| Mirror | BKG (Bundesamt für Kartographie und Geodäsie) |
| Product | Final 30-second clock solutions |
| Format | RINEX CLK 3.04 |
| Access Date | 2024-12-26 |
| Integrity | Files downloaded and decompressed successfully |

## References

1. IGS Analysis Center Coordinator: https://igs.org/
2. IGS Clock Products: https://igs.bkg.bund.de/root_ftp/IGS/products/
3. RINEX Clock Format: https://files.igs.org/pub/data/format/rinex_clock304.txt
