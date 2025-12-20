# COWLS Field-Level Residual Study: Final Summary

## Executive Summary

This study applies the MSOUP arc-domain clustering methodology to JWST COWLS gravitational lens residual images. We analyzed **34 lenses** across score bins M25, S10, S11, and S12 using model-subtracted residuals (source_light + lens_light subtraction).

### Key Finding

**Global Z_corr = 3.67** (>3σ significance) indicates statistically significant angular correlation structure in the residuals. This signal is robust across threshold variations and strongest in the highest-quality M25 bin.

---

## 1. Main Results

### Full Study (M25 + S10 + S11 + S12)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Lenses processed | 34 | All with model residuals |
| Global Z_corr | **3.671** | **Significant correlation signal** |
| Global Z_pow | -0.700 | No power excess |
| Mean Z_corr | 0.629 | Consistently positive across lenses |
| Mean Z_pow | -0.120 | Neutral power distribution |

### Interpretation

- **Z_corr > 3**: The residuals show angular clustering beyond what random noise would produce. This could indicate:
  - Systematic model subtraction artifacts
  - Unmodeled source structure (e.g., clumpy host galaxies)
  - Real physical signal (e.g., substructure effects)

- **Z_pow ~ 0**: No evidence for excess high-frequency power, suggesting the signal is coherent rather than noise-like.

---

## 2. Robustness Checks

### 2.1 By Score Bin

| Score Bin | N Lenses | Global Z_corr | Global Z_pow | Mean Z_corr |
|-----------|----------|---------------|--------------|-------------|
| **M25** | 17 | **3.623** | -1.092 | 0.879 |
| S10 | 12 | 1.657 | -0.460 | 0.478 |
| S11 | 3 | -0.156 | -0.888 | -0.090 |
| S12 | 2 | 0.264 | 2.251 | 0.186 |

**Key observations:**
- The signal is **driven primarily by M25** (highest-quality lenses with best models)
- S10 shows a weaker but still positive trend
- S11/S12 have too few lenses for meaningful statistics

### 2.2 Threshold Sensitivity (SNR threshold)

| SNR Threshold | Global Z_corr | Global Z_pow |
|---------------|---------------|--------------|
| 1.0 (loose) | 3.714 | -0.827 |
| 1.5 (default) | 3.671 | -0.700 |
| 2.0 (strict) | 3.602 | -0.788 |

**Conclusion:** The correlation signal is **stable across threshold choices** (Z_corr ≈ 3.6-3.7), demonstrating robustness.

### 2.3 Residual Type

All 34 lenses used model residuals (source_light + lens_light subtraction). No approximate residuals were needed, ensuring consistent methodology.

---

## 3. Notable Individual Lenses

### Highest Correlation (Z_corr ≈ 1.0)

| Lens ID | T_corr | Z_corr | Notes |
|---------|--------|--------|-------|
| COSJ100013+023424 | 0.721 | 1.00 | Exceptional correlation, high T_pow |
| COSJ100047+015023 | 0.417 | 0.99 | Strong arc structure |
| COSJ100025+015245 | 0.273 | 0.98 | Consistent positive signal |

### Outliers Worth Investigating

| Lens ID | Z_corr | Z_pow | Notes |
|---------|--------|-------|-------|
| COSJ100121+022740 | 0.94 | 1.90 | High both metrics - possible strong residual |
| COSJ095908+021559 | -0.35 | 1.84 | Negative corr but high power |
| COSJ100027+020051 | -0.71 | -0.44 | Only negative outlier in M25 |

---

## 4. Methodology Notes

### Pipeline Configuration

```
Band: F277W (auto-selected as reddest available)
Null draws: 300 (shift + resample combined)
SNR threshold: 1.5 (arc mask construction)
Annulus width: 0.5 (Einstein radius fraction)
Lag max: 6 (correlation lags)
HF fraction: 0.35 (high-frequency power cutoff)
```

### Residual Construction

Residuals computed as: `data - source_light - lens_light`

Both model components are image-plane products (209×209 pixels), ensuring proper alignment with science data.

---

## 5. Conclusions

1. **Significant angular correlation detected**: Global Z_corr = 3.67 exceeds the 3σ threshold for statistical significance.

2. **Signal quality-dependent**: The effect is strongest in M25 (best-modeled lenses), suggesting it relates to residual source structure rather than pure noise.

3. **Robust to analysis choices**: The signal persists across SNR threshold variations (1.0-2.0).

4. **No power excess**: Z_pow ≈ 0 rules out simple noise contamination; the signal has coherent angular structure.

5. **Next steps**:
   - Visual inspection of high-Z_corr residuals
   - Cross-correlation with known source morphology
   - Comparison with mock lenses (pure noise baseline)
   - Investigation of model systematics

---

## 6. File Locations

```
results/cowls_field_study/
├── pilot_M25/report.md              # Initial pilot run
├── full_M25_S10-S12/report.md       # Main combined result
├── robustness/
│   ├── by_score_bin_M25/report.md   # M25 only
│   ├── by_score_bin_S10/report.md   # S10 only
│   ├── by_score_bin_S11/report.md   # S11 only
│   ├── by_score_bin_S12/report.md   # S12 only
│   ├── snr_thresh_1.0/report.md     # Loose threshold
│   └── snr_thresh_2.0/report.md     # Strict threshold
└── FINAL_SUMMARY.md                 # This file
```

---

*Generated by COWLS field-level study pipeline*
*Date: 2025-12-20*
