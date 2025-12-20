# Mverse Shadow Simulation: K1/K2 Kill Condition Calibration

Generated: 2025-12-19 22:33:47

## Executive Summary

This simulation tests whether an M2 observer would falsely conclude that
the gravity-channel projection is non-universal (alpha varies) even when
the underlying truth is perfectly universal (alpha = 1.0 constant).

**Key Finding:** Even with universal alpha, known systematic biases
(hydrostatic bias, projection effects, anisotropy) cause substantial
scatter in the observed ratio R = M_exc^lens / M_exc^dyn.

- **Median R_X** (lens/X-ray): 1.729
- **Median R_V** (lens/velocity): 1.313
- **Scatter (MAD) R_X**: 0.859
- **Scatter (MAD) R_V**: 0.704

## Recommended Pre-Registration Thresholds

Based on this simulation under the null hypothesis (universal alpha):

| Kill Condition | Metric | Recommended Tolerance |
|----------------|--------|----------------------|
| K1 (scatter) | 95th pct of \|R-1\| | **5.802** |
| K2 (mass trend) | 3-sigma null slope | **0.5125** dex^-1 |
| K2 (z trend) | 3-sigma null slope | **0.9877** |

**Interpretation:**
- K1: If observed |R-1| exceeds this tolerance for a significant fraction
  of clusters (>5%), the model may have a problem.
- K2: If the slope of R vs log(M) or z exceeds these tolerances,
  there may be real environmental dependence of alpha.

## Simulation Settings

| Parameter | Value |
|-----------|-------|
| N clusters | 2000 |
| Random seed | 1 |
| Mode | Full (per-object fitting) |
| z range | [0.1, 0.8] |
| log(M200) range | [13.5, 15.5] |
| Alpha (truth) | 1.0 (universal) |
| Runtime | 8.9 s |

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

## Metrics at r500 (N = 2000)

### K1: Scatter about R = 1

| Statistic | R_X (Lens/X-ray) | R_V (Lens/Velocity) |
|-----------|------------------|---------------------|
| Mean | 2.314 | 1.849 |
| Median | 1.729 | 1.313 |
| Std | 2.028 | 1.829 |
| MAD | 0.859 | 0.704 |
| 16-84% | [0.720, 3.777] | [0.511, 2.978] |
| 5-95% | [0.296, 6.802] | [0.190, 5.344] |

**Violation fractions P(|R-1| > eps):**

| eps | R_X | R_V |
|-----|-----|-----|
| 0.05 | 96.8% | 95.7% |
| 0.10 | 93.3% | 91.8% |
| 0.20 | 85.9% | 83.4% |
| 0.30 | 79.5% | 74.1% |
| 0.50 | 67.3% | 59.6% |

**Recommended K1 tolerance (95th pct of |R-1|):** X=5.802, V=4.344

### K2: Trends with Environment

| Trend | Slope | Std Err | p-value |
|-------|-------|---------|---------|
| R_X vs log(M) | -1.6381 | 0.1103 | 0.000 |
| R_V vs log(M) | -1.4840 | 0.0994 | 0.000 |
| R_X vs z | 0.2563 | 0.2258 | 0.256 |
| R_V vs z | -0.1422 | 0.2037 | 0.485 |

**Recommended K2 tolerances (3-sigma of null):** mass=0.5125, z=0.9877

**False positive rates under null:** mass=3.0%, z=7.0%


## Metrics at half_r500 (N = 2000)

### K1: Scatter about R = 1

| Statistic | R_X (Lens/X-ray) | R_V (Lens/Velocity) |
|-----------|------------------|---------------------|
| Mean | 2.266 | 1.835 |
| Median | 1.728 | 1.316 |
| Std | 1.923 | 1.773 |
| MAD | 0.856 | 0.706 |
| 16-84% | [0.716, 3.699] | [0.512, 3.017] |
| 5-95% | [0.296, 6.211] | [0.203, 5.205] |

**Violation fractions P(|R-1| > eps):**

| eps | R_X | R_V |
|-----|-----|-----|
| 0.05 | 96.5% | 96.2% |
| 0.10 | 92.8% | 91.5% |
| 0.20 | 85.8% | 82.8% |
| 0.30 | 79.5% | 74.1% |
| 0.50 | 66.6% | 59.8% |

**Recommended K1 tolerance (95th pct of |R-1|):** X=5.211, V=4.205

### K2: Trends with Environment

| Trend | Slope | Std Err | p-value |
|-------|-------|---------|---------|
| R_X vs log(M) | -1.4818 | 0.1051 | 0.000 |
| R_V vs log(M) | -1.3654 | 0.0969 | 0.000 |
| R_X vs z | 0.1695 | 0.2141 | 0.429 |
| R_V vs z | -0.1127 | 0.1974 | 0.568 |

**Recommended K2 tolerances (3-sigma of null):** mass=0.4851, z=0.8885

**False positive rates under null:** mass=4.0%, z=6.0%


## Physical Interpretation

### Why R_X > 1 (lensing > X-ray hydrostatic)

The X-ray hydrostatic mass is biased low because:
- Non-thermal pressure support not accounted for (b_hse ~ 20%)
- Analyst assumes hydrostatic equilibrium

This causes M_exc^dynX to be underestimated, pushing R_X above 1.
Expected: R_X ~ 1/(1-b_hse) ~ 1.25
Observed median: 1.729

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