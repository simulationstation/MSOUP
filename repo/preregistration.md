# Preregistration: BAO Environment-Overlap Analysis

**Version**: 1.0.0
**Date**: 2025-12-23
**Status**: LOCKED
**SHA256 of preregistration.yaml**: `c3ccc533eeef4a03f1ea87fd7ca73945ba23a188f3a79cb8791c3912194ddc9c`

---

## 1. Research Question

**Primary Question**: Does the BAO tangential dilation parameter α⊥ exhibit systematic variation with environment-overlap strength E in post-DR7 spectroscopic galaxy surveys, beyond what is expected from ΛCDM mock catalogs and beyond what BAO reconstruction removes?

**Motivation**: Standard BAO analyses assume the acoustic scale is environment-independent. If large-scale structure introduces mode-coupling or nonlinear effects that correlate with local environment, this could manifest as an apparent E-dependence of α⊥. Such a signal would be cosmologically significant if it persists after reconstruction and exceeds mock expectations.

---

## 2. Primary Endpoint

**Parameter**: β in the linear model α⊥(E) = α₀ + β·E

**Specifications**:
- Wedge definition: μ ∈ [0.0, 0.2] (tangential/transverse)
- Separation range: s ∈ [50, 180] h⁻¹ Mpc
- Bin width: Δs = 5 h⁻¹ Mpc
- Number of s-bins: 26

**Null hypothesis**: β = 0 (no environment dependence)

---

## 3. Primary Dataset

**Survey**: eBOSS DR16
**Sample**: LRGpCMASS (combined BOSS CMASS + eBOSS LRG)
**Redshift range**: 0.6 < z < 1.0
**Effective redshift**: z_eff = 0.698

**File paths** (relative to data root):
```
DATA_NGC: data/eboss/dr16/LRGpCMASS/v1/eBOSS_LRGpCMASS_clustering_data-NGC-vDR16.fits
DATA_SGC: data/eboss/dr16/LRGpCMASS/v1/eBOSS_LRGpCMASS_clustering_data-SGC-vDR16.fits
RAND_NGC: data/eboss/dr16/LRGpCMASS/v1/eBOSS_LRGpCMASS_clustering_random-NGC-vDR16.fits
RAND_SGC: data/eboss/dr16/LRGpCMASS/v1/eBOSS_LRGpCMASS_clustering_random-SGC-vDR16.fits
DATA_NGC_REC: data/eboss/dr16/LRGpCMASS/v1/eBOSS_LRGpCMASS_clustering_data_rec-NGC-vDR16.fits
DATA_SGC_REC: data/eboss/dr16/LRGpCMASS/v1/eBOSS_LRGpCMASS_clustering_data_rec-SGC-vDR16.fits
RAND_NGC_REC: data/eboss/dr16/LRGpCMASS/v1/eBOSS_LRGpCMASS_clustering_random_rec-NGC-vDR16.fits
RAND_SGC_REC: data/eboss/dr16/LRGpCMASS/v1/eBOSS_LRGpCMASS_clustering_random_rec-SGC-vDR16.fits
```

**Provenance**: https://data.sdss.org/sas/dr16/eboss/lss/catalogs/DR16/

---

## 4. Primary Environment Metric (E1)

**Definition**: Line-integrated smoothed overdensity along galaxy-pair paths

**Algorithm**:
1. Construct 3D density field δ(x) from galaxy catalog using CIC assignment
2. Apply Gaussian smoothing with radius R = 15 h⁻¹ Mpc
3. For each galaxy pair (i,j), compute:
   ```
   E1_ij = ∫₀¹ δ_smooth(x_i + t·(x_j - x_i)) dt
   ```
   evaluated via trapezoidal rule with step = 2 h⁻¹ Mpc along the path
4. Assign per-galaxy E1 as mean of E1_ij over all pairs containing galaxy i
5. Normalize E1 globally: E1_norm = (E1 - median(E1)) / MAD(E1)

**Parameters** (LOCKED):
- Smoothing radius R: 15 h⁻¹ Mpc
- Line integral step: 2 h⁻¹ Mpc
- Pair subsample fraction: 0.1 (for computational efficiency)
- Normalization: Global median/MAD

---

## 5. Primary Analysis Choices (LOCKED)

### 5.1 Correlation Estimator
- **Estimator**: Landy-Szalay (1993)
  ```
  ξ(s,μ) = (DD - 2DR + RR) / RR
  ```
- **Normalization**: Standard (DD, DR, RR normalized by pair counts)

### 5.2 Binning
- s-bins: 50 to 180 h⁻¹ Mpc, Δs = 5 h⁻¹ Mpc (26 bins)
- μ-bins: 0 to 1, Δμ = 0.01 (100 bins, then integrated for wedges)
- Environment bins: 4 quartile bins by per-galaxy E1

### 5.3 Weights Expression
```
w_total = WEIGHT_SYSTOT × WEIGHT_CP × WEIGHT_NOZ × WEIGHT_FKP
```
where:
- WEIGHT_SYSTOT: imaging systematics
- WEIGHT_CP: close-pair fiber collision
- WEIGHT_NOZ: redshift failure
- WEIGHT_FKP: FKP optimal weight (P₀ = 10000 h⁻³ Mpc³)

### 5.4 Random Catalog Usage
- Use full random catalogs (no downsampling)
- Random:data ratio as provided (~50:1)

### 5.5 Covariance Estimation
- **Primary**: Mock-based covariance from N ≥ 200 EZmocks
- **Fallback**: If mocks unavailable, use jackknife with 100 angular regions
- Jackknife region scheme: HEALPix Nside=4, combined NGC+SGC

### 5.6 BAO Template Fit

**Model**:
```
ξ_model(s) = B² × ξ_template(α·s) + A(s)
```
where:
- ξ_template: Eisenstein & Hu (1998) linear template at fiducial cosmology
- α: isotropic dilation parameter (for wedge-specific fits, α⊥ for tangential)
- B: amplitude nuisance parameter
- A(s): polynomial nuisance terms

**Nuisance polynomial**:
```
A(s) = a₀ + a₁/s + a₂/s²
```

**Fit range**: s ∈ [60, 160] h⁻¹ Mpc (subset of measurement range)

**Fitting method**: χ² minimization via scipy.optimize.minimize (L-BFGS-B)

### 5.7 Reconstruction Settings

**Pre-reconstruction**: Use clustering_data files
**Post-reconstruction**: Use clustering_data_rec files

Reconstruction parameters (as applied by SDSS pipeline):
- Bias: b = 2.0
- Smoothing: 15 h⁻¹ Mpc
- Fiducial cosmology: Ωₘ = 0.31, h = 0.676

---

## 6. Blinding Protocol

### 6.1 Method: β-Significance Hiding

The blinding prevents any examination of the statistical significance of β until the formal unblinding step.

**Implementation**:
1. During blinded analysis, the pipeline computes β and σ_β but stores only:
   - `beta_blinded = β + κ` where κ ~ U(-0.5, 0.5) is a random offset
   - `significance_hidden = True`
2. The true β, σ_β, and |β|/σ_β ratio are encrypted using Fernet symmetric encryption
3. The encryption key is stored in a separate file with restricted permissions

### 6.2 Key Storage
- **Key file**: `~/.bao_blind_key` (mode 0600, read-only by analyst)
- **Key generation**: `secrets.token_bytes(32)` at analysis initialization
- **Key destruction**: After unblinding, key is securely deleted

### 6.3 Access Control
- Only the designated PI may execute the unblind script
- Unblinding requires explicit `--confirm-unblind` flag
- All unblinding actions are logged with timestamp and user

### 6.4 What Remains Visible During Blinding
- ξ(s) wedge measurements (shapes, but not α interpretation)
- Environment bin assignments
- Mock pipeline outputs
- Covariance matrices
- All QA diagnostics except β significance

---

## 7. Decision Criteria

### 7.1 Primary Detection Threshold

Declare **"anomalous environment scaling"** if ALL of the following are satisfied:

1. **Statistical significance**: |β| / σ_β ≥ 3.0 (3σ detection)

2. **Mock calibration**: β_obs exceeds the 99.7th percentile of the mock β distribution
   - Equivalently: fewer than 1 in 370 mocks have |β_mock| ≥ |β_obs|

3. **Reconstruction persistence**: The effect persists post-reconstruction at a level inconsistent with mocks:
   - |β_recon| / σ_β,recon ≥ 2.0 AND
   - β_recon is beyond the 95th percentile of mock β_recon distribution

### 7.2 Null Result Criteria

Accept null hypothesis (β = 0) if:
- |β| / σ_β < 2.0 AND
- β is within the central 90% of the mock distribution

### 7.3 Inconclusive Result

If criteria for neither detection nor null are met:
- Report β ± σ_β and mock percentile
- Proceed to DESI for higher statistical power
- Do not claim detection or null

---

## 8. Secondary Endpoints

### 8.1 Alternative Environment Metric (E2)
- **Definition**: Supercluster voxel overlap
- **Algorithm**: Binary indicator of whether pair path intersects high-density voxels (δ > 1)
- **Purpose**: Cross-check E1 results with simpler metric

### 8.2 Radial Wedge (Control)
- **Definition**: μ ∈ [0.8, 1.0]
- **Purpose**: BAO in radial direction should show different systematics pattern
- **Expectation**: If E-dependence is due to AP-like effects, radial and tangential should differ; if spurious, they may be correlated

### 8.3 Tracer Consistency
- **eBOSS QSO**: z ∈ [0.8, 2.2], repeat analysis
- **DESI ELG**: If proceeding to DESI, include ELG sample
- **Requirement**: Consistent β sign and magnitude across tracers (within 2σ)

### 8.4 Regional Consistency
- **NGC vs SGC**: Analyze separately
- **Requirement**: NGC and SGC β values consistent within 2σ
- **Combined**: Primary result uses NGC+SGC combined

---

## 9. Robustness Checks (Pre-declared)

All robustness checks are pre-declared and will be run BEFORE unblinding.

### 9.1 Smoothing Radius Variants
| Variant | R (h⁻¹ Mpc) | Status |
|---------|-------------|--------|
| R10 | 10 | Secondary |
| R15 | 15 | PRIMARY |
| R20 | 20 | Secondary |

### 9.2 Wedge Bound Variants
| Variant | μ range | Status |
|---------|---------|--------|
| W02 | [0.0, 0.2] | PRIMARY |
| W025 | [0.0, 0.25] | Secondary |
| W03 | [0.0, 0.3] | Secondary |

### 9.3 Separation Range Variants
| Variant | s range (h⁻¹ Mpc) | Status |
|---------|-------------------|--------|
| S50_180 | [50, 180] | PRIMARY |
| S60_160 | [60, 160] | Secondary |
| S40_200 | [40, 200] | Secondary |

### 9.4 Weight Robustness
| Variant | Description |
|---------|-------------|
| W_FULL | All weights (PRIMARY) |
| W_NOTOP | Remove top 0.5% by total weight |
| W_NOCP | Exclude WEIGHT_CP |

### 9.5 Fit Model Variants
| Variant | Nuisance polynomial |
|---------|---------------------|
| POLY2 | a₀ + a₁/s + a₂/s² (PRIMARY) |
| POLY3 | a₀ + a₁/s + a₂/s² + a₃/s³ |
| POLY1 | a₀ + a₁/s |

### 9.6 Environment Binning Variants
| Variant | N bins | Method |
|---------|--------|--------|
| Q4 | 4 | Quartiles (PRIMARY) |
| Q5 | 5 | Quintiles |
| Q3 | 3 | Terciles |

### 9.7 Pass/Fail Criteria for Robustness
A robustness check **passes** if:
- β changes by < 1σ from primary analysis
- Sign of β is preserved
- Statistical significance changes by < 1 unit (e.g., 3.2σ → 2.5σ is a fail)

---

## 10. Stopping Rules

### 10.1 Proceed to DESI if:
- eBOSS result is inconclusive (2σ < |β|/σ_β < 3σ)
- OR eBOSS shows detection and DESI needed for confirmation
- OR null result but DESI provides >3× statistical power for science value

### 10.2 Terminate Analysis if:
- Critical pipeline bug discovered that invalidates results
- Mock distribution shows pathological behavior (e.g., non-Gaussian, bimodal)
- >50% of robustness checks fail

### 10.3 Accept Null if:
- eBOSS |β|/σ_β < 2.0
- β within central 90% of mocks
- All robustness checks pass
- No anomalies in QA diagnostics

---

## 11. Fiducial Cosmology

| Parameter | Value |
|-----------|-------|
| Ωₘ | 0.31 |
| h | 0.676 |
| Ωb | 0.049 |
| σ₈ | 0.81 |
| nₛ | 0.97 |
| r_d | 147.09 Mpc |

---

## 12. Software and Reproducibility

### 12.1 Required Software
- Python ≥ 3.10
- numpy, scipy, astropy, fitsio
- emcee (for Bayesian inference)
- corrfunc or treecorr (pair counting)
- healpy (jackknife regions)

### 12.2 Version Pinning
All package versions will be recorded in `environment.yaml` at analysis start.

### 12.3 Random Seeds
- Master seed: 12345
- All RNG operations derive from this seed deterministically

---

## 13. Audit Trail Requirements

The following artifacts must be generated and preserved:

1. **Config hash**: SHA256 of preregistration.yaml, recorded in all outputs
2. **Command log**: Every shell command with timestamp
3. **Artifact manifest**: List of all generated files with checksums
4. **Stage checksums**: Hash of inputs and outputs for each pipeline stage
5. **Git commit**: Analysis code committed before execution

---

## 14. Signatures

This preregistration is locked as of the date above. Any modifications require:
1. Documented justification
2. New version number
3. Explicit notation of changes from previous version

**Preregistration hash will be computed and stored after this document is finalized.**

---

## Appendix A: Mock Catalog Specification

**eBOSS EZmocks**:
- Location: SDSS SAS or local mirror
- N_mocks: 1000 (use ≥200 for covariance, full set for null distribution)
- Format: Same as data catalogs
- Reconstruction: Pre-computed reconstructed versions available

---

## Appendix B: Glossary

- **α⊥**: Transverse/tangential BAO dilation parameter
- **α∥**: Radial/line-of-sight BAO dilation parameter
- **E**: Environment-overlap metric
- **β**: Slope of α⊥ vs E linear relation
- **μ**: Cosine of angle to line of sight
- **s**: Comoving separation (h⁻¹ Mpc)
