# Mverse Shadow Simulation: K1/K2 Kill Condition Calibration

Generated: 2025-12-19 22:34:01

## Executive Summary

This simulation tests whether an M2 observer would falsely conclude that
the gravity-channel projection is non-universal (alpha varies) even when
the underlying truth is perfectly universal (alpha = 1.0 constant).

**Key Finding:** Even with universal alpha, known systematic biases
(hydrostatic bias, projection effects, anisotropy) cause substantial
scatter in the observed ratio R = M_exc^lens / M_exc^dyn.

- **Median R_X** (lens/X-ray): 1.328
- **Median R_V** (lens/velocity): 0.928
- **Scatter (MAD) R_X**: 0.315
- **Scatter (MAD) R_V**: 0.289

## Recommended Pre-Registration Thresholds

Based on this simulation under the null hypothesis (universal alpha):

| Kill Condition | Metric | Recommended Tolerance |
|----------------|--------|----------------------|
| K1 (scatter) | 95th pct of \|R-1\| | **1.367** |
| K2 (mass trend) | 3-sigma null slope | **0.0568** dex^-1 |
| K2 (z trend) | 3-sigma null slope | **0.1126** |

**Interpretation:**
- K1: If observed |R-1| exceeds this tolerance for a significant fraction
  of clusters (>5%), the model may have a problem.
- K2: If the slope of R vs log(M) or z exceeds these tolerances,
  there may be real environmental dependence of alpha.

## Simulation Settings

| Parameter | Value |
|-----------|-------|
| N clusters | 10000 |
| Random seed | 1 |
| Mode | Fast (vectorized) |
| z range | [0.1, 0.8] |
| log(M200) range | [13.5, 15.5] |
| Alpha (truth) | 1.0 (universal) |
| Runtime | 5.9 s |

### Bias Parameters

| Bias | Mean | Scatter |
|------|------|---------|
| Hydrostatic (b_hse) | 0.20 | 0.08 |
| Projection boost | 0.00 | 0.12 dex |
| Anisotropy (b_aniso) | 0.00 | 0.07 |
| Lensing calibration | 0.000 | 0.020 |

### Noise Levels

| Observable | Noise (fractional) |
|------------|-------------------|
| X-ray mass | 8% |
| Velocity dispersion | 7% |
| Visible mass (gas) | 10% |
| Visible mass (stars) | 20% |

## Metrics at r500 (N = 10000)

### K1: Scatter about R = 1

| Statistic | R_X (Lens/X-ray) | R_V (Lens/Velocity) |
|-----------|------------------|---------------------|
| Mean | 1.412 | 1.053 |
| Median | 1.328 | 0.928 |
| Std | 0.515 | 0.553 |
| MAD | 0.315 | 0.289 |
| 16-84% | [0.929, 1.882] | [0.583, 1.505] |
| 5-95% | [0.732, 2.367] | [0.432, 2.105] |

**Violation fractions P(|R-1| > eps):**

| eps | R_X | R_V |
|-----|-----|-----|
| 0.05 | 91.9% | 92.1% |
| 0.10 | 84.0% | 83.6% |
| 0.20 | 69.8% | 66.8% |
| 0.30 | 56.4% | 51.5% |
| 0.50 | 37.3% | 25.2% |

**Recommended K1 tolerance (95th pct of |R-1|):** X=1.367, V=1.105

### K2: Trends with Environment

| Trend | Slope | Std Err | p-value |
|-------|-------|---------|---------|
| R_X vs log(M) | 0.1189 | 0.0131 | 0.000 |
| R_V vs log(M) | -0.0755 | 0.0141 | 0.000 |
| R_X vs z | 0.1872 | 0.0253 | 0.000 |
| R_V vs z | -0.1164 | 0.0273 | 0.000 |

**Recommended K2 tolerances (3-sigma of null):** mass=0.0568, z=0.1126

**False positive rates under null:** mass=4.0%, z=3.0%


## Metrics at half_r500 (N = 10000)

### K1: Scatter about R = 1

| Statistic | R_X (Lens/X-ray) | R_V (Lens/Velocity) |
|-----------|------------------|---------------------|
| Mean | 1.410 | 0.947 |
| Median | 1.329 | 0.817 |
| Std | 0.522 | 0.535 |
| MAD | 0.318 | 0.270 |
| 16-84% | [0.921, 1.896] | [0.505, 1.376] |
| 5-95% | [0.723, 2.388] | [0.361, 1.960] |

**Violation fractions P(|R-1| > eps):**

| eps | R_X | R_V |
|-----|-----|-----|
| 0.05 | 92.1% | 92.8% |
| 0.10 | 84.4% | 85.7% |
| 0.20 | 69.7% | 71.3% |
| 0.30 | 56.3% | 56.3% |
| 0.50 | 37.0% | 27.9% |

**Recommended K1 tolerance (95th pct of |R-1|):** X=1.388, V=0.960

### K2: Trends with Environment

| Trend | Slope | Std Err | p-value |
|-------|-------|---------|---------|
| R_X vs log(M) | 0.1165 | 0.0133 | 0.000 |
| R_V vs log(M) | -0.1262 | 0.0136 | 0.000 |
| R_X vs z | 0.1870 | 0.0257 | 0.000 |
| R_V vs z | -0.1929 | 0.0263 | 0.000 |

**Recommended K2 tolerances (3-sigma of null):** mass=0.0542, z=0.1032

**False positive rates under null:** mass=5.0%, z=4.0%


## Physical Interpretation

### Why R_X > 1 (lensing > X-ray hydrostatic)

The X-ray hydrostatic mass is biased low because:
- Non-thermal pressure support not accounted for (b_hse ~ 20%)
- Analyst assumes hydrostatic equilibrium

This causes M_exc^dynX to be underestimated, pushing R_X above 1.
Expected: R_X ~ 1/(1-b_hse) ~ 1.25
Observed median: 1.328

### Scatter Sources

1. **Projection effects**: Triaxial halos projected along different axes
   cause scatter in lensing masses (not dynamics)

2. **Intrinsic scatter in biases**: b_hse and b_aniso vary cluster-to-cluster

3. **Measurement noise**: Shape noise (lensing), temperature errors (X-ray),
   velocity errors (dynamics)

4. **Visible mass uncertainty**: Gas and stellar fraction estimation errors

## How to Use These Results

1. **Pre-register K1/K2 tolerances** before analyzing real data

2. **Do NOT interpret |R-1| < tolerance as evidence for universal gravity**
   - This is the expected null result under standard systematics

3. **DO interpret |R-1| >> tolerance as potential tension** that warrants
   investigation of either:
   - Underestimated systematics
   - Real physics (non-universal gravity channel)

4. **K2 trends are more diagnostic** than K1 scatter:
   - Mass/z trends in R would be harder to explain with systematics
   - But beware: b_hse has known mass/z dependence that creates false trends

## Caveats

- This simulation uses simplified NFW profiles and analytic scalings
- Real cluster physics is more complex (baryonic effects, substructure)
- Systematic biases in real data may differ from assumed values
- Selection effects not modeled

---

*Report generated by mverse_shadow_sim*