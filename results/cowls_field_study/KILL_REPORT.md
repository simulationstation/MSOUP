# COWLS Field-Level Detection: Kill Analysis Report

## Executive Summary

**VERDICT: SIGNAL KILLED - Dominated by Low-m (Macro-Model) Systematics**

| Metric | Baseline | High-Pass (m_cut=3) | High-Pass (m_cut=5) |
|--------|----------|---------------------|---------------------|
| Global Z_corr | 3.634 | 2.274 | **-0.503** |
| p_emp (resample) | 0.003 | 0.014 | N/A |
| Significance | 3.6σ | 2.3σ | **<0** |

**Critical Finding**: The signal is entirely concentrated in modes m ≤ 3. After removing these low-order (large-scale) modes:
- m_cut=3: Signal drops from 3.6σ to 2.3σ (p > 0.01, not significant at 99% level)
- m_cut=5: Signal becomes **negative** (-0.5σ)

---

## Key Questions Answered

### a) Does the signal survive after removing low-m?

**NO.** The signal does not survive high-pass filtering:

| m_cut | Global Z_corr_hp | Status |
|-------|------------------|--------|
| 0 (baseline) | 3.634 | Significant |
| 3 | 2.274 | p > 0.01, not significant |
| 5 | -0.503 | Negative (no signal) |

The low-m dominance correlation ρ(Z_corr, low-m ratio) = **0.919** confirms that lenses with high Z_corr have their signal concentrated in m ≤ 3 modes. After high-pass filtering, this correlation drops to 0.332.

### b) Does band-consistency improve under matched masks and/or after high-pass filtering?

**NO.** Band consistency remains poor regardless of filtering or masking:

| Condition | Sign Consistency | F150W vs F277W Correlation |
|-----------|-----------------|---------------------------|
| Baseline (unfiltered) | 41% (14/34) | r = 0.046 |
| High-pass (m_cut=3) | 44% (15/34) | r = 0.085 |
| Matched mask (all_bands) | Applied | No improvement |

- Expected for gravitational (achromatic) signal: ~100% sign consistency
- Observed: ~41-44% (barely above chance at 25%)
- Cross-band correlation is essentially zero (r ≈ 0.05-0.09)

This confirms the signal is wavelength-dependent, consistent with source morphology or PSF systematics rather than gravitational lensing.

### c) Conclusion

**The phenomenon is dominated by macro-model/morphology systematics.**

The evidence is definitive:
1. **Low-m dominance**: Signal is in m ≤ 3 modes (large-scale angular structure), not the high-m modes expected from subhalo perturbations
2. **High-pass failure**: Signal vanishes or becomes negative after removing low-order modes
3. **Band inconsistency**: Only 41% sign consistency, r ≈ 0 between bands
4. **Wavelength dependence**: The signal varies with band, ruling out achromatic gravitational effects

---

## Sample Overview

| Score Bin | N Lenses | Mean Z_corr | Mean Z_corr_hp |
|-----------|----------|-------------|----------------|
| M25 | 17 | 0.875 | 0.430 |
| S10 | 12 | 0.497 | 0.350 |
| S11 | 3 | 0.000 | -0.270 |
| S12 | 2 | 0.180 | -0.225 |
| **Total** | **34** | **0.623** | **0.390** |

- All 34 lenses used (no exclusions)
- All 34 have 4 bands: F115W, F150W, F277W, F444W

---

## Kill Condition Checklist

| Condition | Status | Value | Threshold |
|-----------|--------|-------|-----------|
| Null p-value (baseline) | ✓ PASS | p = 0.003 | < 0.01 |
| Null p-value (hp) | ❌ FAIL | p = 0.014 | < 0.01 |
| Single lens dominates | ✓ PASS | max drop = 0.12σ | < 0.5σ |
| LOO min | ✓ PASS | 3.52σ | > 3.0σ |
| Proxy correlation | ✓ PASS | max |ρ| = 0.42 | < 0.5 |
| Low-m dominance | ❌ FAIL | ρ = 0.92 | < 0.6 |
| Band sign consistency | ❌ FAIL | 41% | > 50% |
| Signal survives high-pass | ❌ FAIL | Z_hp = 2.27 | p < 0.01 |

---

## Section A: Headline Numbers & Dominance

### Reproduced Statistics

| Metric | Value |
|--------|-------|
| n_lenses | 34 |
| mean(Z_corr) | 0.623 |
| mean(Z_corr_hp) | 0.390 |
| Global Z_corr | 3.634 |
| Global Z_corr_hp | 2.274 |
| Z_corr min/median/max | -0.70 / 0.92 / 1.00 |

### Dominance Analysis

| Metric | Value |
|--------|-------|
| Top 1 lens contribution | 4.7% |
| Top 3 lens contribution | 14.1% |
| Top 5 lens contribution | 23.4% |
| Global Z without top 1 | 3.515 |
| Global Z without top 3 | 3.271 |
| Global Z without top 5 | 3.016 |

![Z_corr Distribution](kill_plots/a_zcorr_distribution.png)

---

## Section B: Leave-One-Out & Jackknife

| Metric | Value |
|--------|-------|
| LOO min | 3.515 |
| LOO median | 3.529 |
| LOO max | 3.811 |
| Max drop lens | COSJ100013+023424 |
| Max drop value | 0.119 |
| Dominance alarm | NO |
| Jackknife mean | 0.623 |
| Jackknife SE | 0.088 |

![Leave-One-Out](kill_plots/b_leave_one_out.png)

---

## Section C: Null Adequacy Tests

| Null Method | G | Std Synth Z | p_emp | p_emp (hp) |
|-------------|---|-------------|-------|------------|
| Resample | 1000 | 1.00 | 0.003 | 0.014 |
| Shift | 500 | 0.90 | 0.002 | 0.012 |

**WARNING: p_hp > 0.01 - high-pass signal is NOT significant at 99% level**

- Observed Z_corr: 3.634
- Observed Z_corr_hp: 2.274
- Lenses used: 34/34

![Null Adequacy](kill_plots/c_null_adequacy.png)

---

## Section D: Artifact Proxy Correlations

| Proxy | Spearman ρ | p-value | N |
|-------|------------|---------|---|
| residual_rms | 0.418 | 0.014 | 34 |
| coverage | -0.074 | 0.679 | 34 |
| texture | 0.025 | 0.888 | 34 |
| psf_fwhm | 0.205 | 0.245 | 34 |

**Strongest proxy: residual_rms (ρ = 0.42)**

![Artifact Proxies](kill_plots/d_artifact_proxies.png)

---

## Section E: Frequency Structure (High-Pass Analysis)

### Sensitivity to m_cut

| m_cut | Global Z_corr_hp | Interpretation |
|-------|------------------|----------------|
| 0 (baseline) | 3.634 | Full signal |
| 3 | 2.274 | 63% of signal in m ≤ 3 |
| 5 | -0.503 | 114% of signal in m ≤ 5 |

### Low-m Dominance

| Metric | Value |
|--------|-------|
| ρ(Z_corr, low-m ratio) | 0.919 (p < 10⁻¹⁵) |
| ρ(Z_corr_hp, low-m ratio) | 0.332 (p = 0.055) |
| Pearson r(Z_corr, low-m ratio) | 0.571 (p = 0.0004) |
| Mean low-m power ratio | 0.163 |
| Lenses with Z_corr > 0.9 & Z_pow < 0 | 12/34 (35%) |

**The extremely strong correlation ρ = 0.92 is definitive**: lenses with high Z_corr have signal in m ≤ 3 modes, not the higher frequencies expected from subhalo perturbations.

![Frequency Structure](kill_plots/e_frequency_structure.png)

---

## Section F: Band Consistency (Matched Masks)

### All-Bands Matched Mask

| Metric | Baseline | High-Pass (m_cut=3) |
|--------|----------|---------------------|
| Sign consistency | 41% (14/34) | 44% (15/34) |
| Mean per-lens variance | 0.231 | 0.222 |
| F150W vs F277W correlation | r = 0.046 | r = 0.085 |

**Interpretation**:
- For gravitational (achromatic) signals, we expect ~100% sign consistency
- Observed 41-44% is barely above chance (25%)
- Cross-band correlation ≈ 0 confirms wavelength-dependent systematics

---

## Top Mundane Explanations (Ranked by Evidence)

| Rank | Explanation | Key Evidence | Confidence |
|------|-------------|--------------|------------|
| 1 | **Low-mode macro-model mismatch** | ρ(Z_corr, low-m) = 0.92, signal vanishes at m_cut=5 | Very High |
| 2 | **Wavelength-dependent source morphology** | Band consistency = 41%, r ≈ 0 between bands | Very High |
| 3 | **Model quality variation** | ρ(Z_corr, residual_RMS) = 0.42 | Moderate |
| 4 | **Score-bin quality gradient** | M25: Z=0.88 vs S11: Z=0.00 | Moderate |

---

## What Would Still Make It Pockets?

For the signal to be attributed to dark substructure perturbations, we would need:

| Requirement | Current Status | Needed |
|-------------|----------------|--------|
| Signal survives high-pass | ❌ Z_hp = -0.5 at m_cut=5 | Z_hp > 3σ at m > 10 |
| Band independence | ❌ 41% consistent | ~100% |
| High-m power peak | ❌ Low-m dominated | Peak at subhalo scales |
| Cross-band correlation | ❌ r ≈ 0 | r ≈ 1 |

**None of these requirements are met. The signal is conclusively dominated by macro-model/morphology systematics.**

---

## Data Inventory

- All 34 lenses used, no exclusions
- Bands per lens: F115W, F150W, F277W, F444W (all 34 have all 4 bands)
- Configuration:
  - G_resample = 1000
  - G_shift = 500
  - m_cut = 3 (sensitivity tested at m_cut = 5)
  - band_common_mask = all_bands

---

*Generated by kill_analysis.py with high-pass filtering and matched-mask band consistency*
