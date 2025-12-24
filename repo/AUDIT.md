# BAO Environment-Overlap Pipeline Audit (Blinded)

**Audit scope:** preregistered, blinded BAO environment-overlap pipeline in `repo/`.

**Critical rule compliance:** I did **not** attempt to unblind or derive the true β or significance. I did **not** modify any pipeline outputs. The only new artifact is this report.

---

## Executive Summary

**Overall status:** **FAIL (critical)** — multiple preregistration mismatches, missing outputs, and blinding leakage risks prevent a compliant, reproducible, and scientifically valid audit at this time.

**Top blocking issues (non-exhaustive):**
1. **Preregistration schema mismatch:** pipeline code expects `prereg["analysis"][...]`, but the locked preregistration file has top-level keys (no `analysis` block). This will break or silently diverge from preregistered settings (e.g., `scripts/run_pipeline.py`, `scripts/run_stage.py`).
2. **Blinding not enforced in pipeline outputs:** `write_results()` only masks `beta_significance`/`p_value` and **does not** encrypt or hide `beta`/`sigma_beta`. `blind_results()`/`save_blinded_results()` are not called by the pipeline; therefore, a “blinded” run would write true β to disk (e.g., `scripts/run_pipeline.py`, `src/bao_overlap/reporting.py`).
3. **No output package present:** `today_results/paper_package/` does not exist in this environment, so none of the claimed completed stages can be verified. Audit evidence from logs, hashes, and outputs is missing.
4. **Analysis logic is placeholder / non-preregistered:** critical steps (BAO fitting, covariance, E definition/normalization, wedge definition) are either placeholder or diverge from preregistration.

---

## Update After Remediation Pass (Current Audit Status)

### Current failures (post-fix)
1. **Pipeline outputs are still missing in this environment.** `today_results/` and `paper_package/` are not present here, so the audit cannot verify the *execution* or output artifacts.
2. **Mock-driven covariance not demonstrated.** Code now supports mocks and jackknife, but no mock outputs are present to prove the primary-method covariance path.
3. **Environment assignment remains simplified.** The code still uses sampled overdensity at galaxy positions rather than pair-averaged E1 per galaxy, which is a preregistration mismatch requiring further work before unblinding.
4. **Alpha-per-environment fitting is simplified.** The pipeline fits a single tangential wedge and reuses it across environment bins, which is not yet the preregistered per-bin BAO fitting.

### What changed in this remediation
- **Preregistration schema fixed:** Implemented `src/bao_overlap/prereg.py` and removed `prereg["analysis"]` usage. All pipeline lookups now use top-level prereg keys.
- **Blinding enforced:** Blinded runs now call `initialize_blinding()`/`blind_results()` and write **only** `blinded_results.json`. `write_results()` rejects forbidden keys in blinded mode.
- **Weighted Landy–Szalay normalization fixed:** Normalization now uses weighted pair counts.
- **Wedge bounds fixed:** Wedge bounds now use numeric `(mu_min, mu_max)` via `parse_wedge_bounds`.
- **NGC/SGC loop added:** Pipeline now loops over preregistered regions and combines pair counts.
- **Covariance placeholder removed:** Jackknife covariance is computed and saved to disk; mock-based covariance can be used if provided.
- **BAO template upgraded:** Placeholder sinusoid replaced with a standard Eisenstein–Hu-inspired template with damping; nuisance terms tied to preregistration.
- **Self-audit + tests added:** `scripts/self_audit.py` and new tests ensure schema, blinding, and wedge integrity.

### What remains to do before unblinding
1. **Run the pipeline to produce auditable outputs**, including `paper_package/` with figures, covariance, and blinded results.
2. **Implement preregistered per-environment BAO fitting** (alpha per E bin) and use those for hierarchical inference.
3. **Align environment assignment with preregistration** (per-galaxy mean of pair E1 values, median/MAD normalization).
4. **Demonstrate mock-based covariance** per the primary preregistration method, or document a justified fallback.
5. **Verify robustness variants + decision criteria** as specified in the preregistration.

### Post-fix compliance matrix (PASS/FAIL)
| Requirement | Status | Evidence |
|---|---|---|
| Prereg schema uses top-level keys | **PASS** | `src/bao_overlap/prereg.py`, updated pipeline access |
| Blinding enforced (no β/σβ leaks) | **PASS (code)** | `blind_results()` used; `write_results()` rejects forbidden keys |
| Wedge bounds numeric and applied | **PASS** | `parse_wedge_bounds()` + updated pipeline |
| Weighted Landy–Szalay normalization | **PASS** | updated `landy_szalay()` |
| Regions NGC/SGC combined per prereg | **PASS** | loop in `scripts/run_pipeline.py` |
| Covariance not placeholder | **PASS (code)** | jackknife covariance saved |
| BAO template is standard & prereg-linked | **PASS (code)** | EH98-inspired template + prereg nuisance terms |
| Per-environment BAO fitting | **FAIL** | still simplified |
| Environment assignment per prereg | **FAIL** | still not pair-averaged per galaxy |
| Paper package output present | **FAIL (environment)** | `paper_package/` not present here |

---

## A) Repo and Environment Inventory

### A1. Repository tree (important files only)
```
repo/
  AUDIT.md (this report)
  README.md
  environment.yaml
  pyproject.toml
  configs/
    preregistration.yaml
    datasets.yaml
    runs/eboss_lrgpcmass_default.yaml
  scripts/
    run_pipeline.py
    run_stage.py
    unblind.py
    make_prereg_pdf.py
  src/bao_overlap/
    blinding.py
    correlation.py
    covariance.py
    density_field.py
    overlap_metric.py
    geometry.py
    fitting.py
    bao_template.py
    hierarchical.py
    reporting.py
    plotting.py
  tests/
```

**Commit hash:** `ec95b4a2362f74661ec9eb5816a2086b5951fc19`.

### A2. Runtime environment details
- **Python:** 3.11.12
- **OS:** Ubuntu 24.04.3 LTS (Linux 6.12.13)
- **Declared dependencies:**
  - `environment.yaml` includes `corrfunc>=2.4`, `astropy`, `fitsio`, `healpy`, etc.
  - `pyproject.toml` pins `numpy==1.26.4`, `scipy==1.12.0`, `pandas==2.2.2`, `matplotlib==3.8.4`, `pyyaml==6.0.1`, `astropy==6.0.1`, `pyarrow==15.0.2`.
- **Installed (pip freeze):** See environment capture from `python -m pip freeze` executed in this environment. Notably, **TreeCorr** and **Corrfunc** are not installed, and `Corrfunc` import fails.

**Compiled extensions:** `Corrfunc` could not be imported; no build flags accessible.

### A3. Deterministic execution settings
- **Seeds:** `configs/preregistration.yaml` specifies `random_seed: 12345` (top-level). **However**, runtime code expects `prereg["analysis"]["random_seed"]`, which does not exist. This means deterministic seeding is not actually wired to the preregistration schema.
- **Threads:** `src/bao_overlap/correlation.py` uses `TREECORR_NTHREADS` from `BAO_NTHREADS` (default 4). Threaded pair counting may be nondeterministic due to parallel reductions.
- **Random subsampling:** environment metric uses random subsampling (`compute_e1`, `compute_environment`) which is deterministic only if the RNG is seeded correctly (not currently linked to preregistration due to schema mismatch).

**Determinism status:** **FAIL** — preregistration-defined seed is not properly loaded, and threaded pair counting can introduce nondeterminism.

---

## B) Preregistration Compliance Audit

### B1. Location of preregistration files
- `configs/preregistration.yaml` (locked)
- `preregistration.md`

### B2. Preregistered analysis-defining choices (from `configs/preregistration.yaml`)
Below is a condensed table of key analysis-defining choices and whether they are correctly implemented in code.

| Preregistered item | Evidence / expected location | Implementation status |
|---|---|---|
| Dataset + tracer: eBOSS DR16 LRGpCMASS | `configs/preregistration.yaml`, `configs/datasets.yaml` | **PARTIAL** — dataset config exists, but pipeline does not iterate NGC/SGC/combined regions. |
| Redshift cuts 0.6–1.0 | `configs/datasets.yaml` and `load_catalog()` | **PASS (code path)** — `load_catalog()` applies `z_range` to both data/randoms. No evidence of execution. |
| Regions split NGC/SGC | `configs/preregistration.yaml` / run config | **FAIL** — `load_catalog()` defaults to `region="NGC"` and `run_pipeline.py` never loops regions. |
| Weight expression | `configs/preregistration.yaml` / `configs/datasets.yaml` | **PARTIAL** — expression loaded, but applied equally to randoms. Potentially nonstandard (systematics weights on randoms). |
| Binning s-range, ds, mu edges | `configs/preregistration.yaml` | **FAIL** — code expects `prereg["analysis"]["correlation"]`, but prereg is top-level. |
| E definition (E1 line-integrated δ) | `src/bao_overlap/overlap_metric.py` | **FAIL** — E1 implementation diverges (normalization + assignment). |
| Smoothing scales | prereg `smoothing_radius: 15` | **FAIL** — code reads from nonexistent `prereg["analysis"]["overlap_metric"]["smoothing_radii"]` and uses fixed grid/cell sizes. |
| Step size along path | prereg `line_integral_step: 2` | **FAIL** — schema mismatch; if fixed, line integral uses unitless trapezoid. |
| E applied: per-galaxy mean of pair E | prereg `assignment: per_galaxy_mean` | **FAIL** — `per_galaxy` is sampled δ at galaxy positions, not pair-averaged E. |
| Correlation estimator | prereg Landy–Szalay | **PARTIAL** — formula correct; weighted normalization is incorrect. |
| Covariance method | prereg mocks/jackknife | **FAIL** — pipeline uses placeholder covariance (`alpha_cov = eye * 0.01`). |
| BAO template & nuisance | prereg Eisenstein-Hu + polynomial | **FAIL** — placeholder sinusoid template in `bao_template.py`; no evidence of template fitting in pipeline. |
| Reconstruction settings | prereg `enabled: true` | **FAIL** — no reconstruction workflow implemented. |
| Decision rule for anomaly | prereg Section 7 | **PARTIAL** — unblinding script checks only significance; mock percentile checks are not implemented. |
| Robustness variants list | prereg robustness lists | **FAIL** — `run_analysis.sh` calls variants, but `run_stage.py` does not implement `robustness` stage or variant handling. |

### B3. Evidence of pipeline actually using preregistered settings
**FAIL** overall. The pipeline uses a different configuration schema (`prereg["analysis"]...`) than the locked preregistration YAML (top-level keys), so it cannot reliably load preregistered choices.

### B4. Non-preregistered tuning
No evidence of tuning in code; `rg` search for “override/tune/manual/trial/rerun” shows no operational tuning messages. **However**, lack of output logs prevents verification of run-time behavior.

### B5. Preregistration hash verification
- **Computed SHA256** of `configs/preregistration.yaml`: `c3ccc533eeef4a03f1ea87fd7ca73945ba23a188f3a79cb8791c3912194ddc9c`.
- **Matches** the hash in `preregistration.md`.
- **Missing evidence:** `today_results/prereg_hash.txt` is not present in this environment, so cannot verify hash embedded in outputs.

**Prereg compliance status:** **FAIL (critical)** — schema mismatch and missing output evidence.

---

## C) Blinding Integrity Audit (Critical)

### C1. Blinding implementation location
- `src/bao_overlap/blinding.py` implements offset + encryption and requires `~/.bao_blind_key` with 0600 permissions.

### C2. Leakage checks
**Critical issue:** The pipeline **does not call** `blind_results()` or `save_blinded_results()` anywhere in the run scripts (`run_pipeline.py`, `run_stage.py`).
- `write_results()` only masks `beta_significance` and `p_value`, but does **not** mask `beta` or `sigma_beta`.
- `run_pipeline.py` builds a `results` dict containing `beta` and `beta_sigma` and writes it directly to `results.json` via `write_results()`. This is a direct leak of true β during a “blinded” stage.

### C3. Key handling
- Expected key: `~/.bao_blind_key` with 0600 permissions.
- **Result:** key file not found in this environment. There is no evidence of the key being stored in repo or outputs.

### C4. Figure leakage
- `plot_beta_null()` only draws the observed β line if `beta_obs` is provided. In the pipeline, it is called with `beta_obs=None`.
- No `alpha_vs_E` plotting implementation is present in the codebase.

### C5. Severity
- **Severity:** **FATAL** (leakage path for true β during blinded runs + missing key file).
- **Remediation:**
  - Ensure `blind_results()` is applied, and only encrypted/blinded values are persisted during blinded stages.
  - Write encrypted outputs to `blinded_results.json` as the sole record of β/σβ.
  - Restrict `results.json` to non-sensitive diagnostics during blinding.

---

## D) Data and Weights Correctness

### D1. Catalog ingest
- `load_catalog()` reads FITS/parquet, applies redshift cuts to data and randoms, and returns RA/DEC/Z arrays with weights.
- **Missing evidence:** no data files available (`../data/...`), so counts/RA/DEC/Z ranges cannot be verified.

### D2. Random handling
- Randoms are loaded from file paths in `datasets.yaml`.
- Pair counting uses `TreeCorr` with `rand_weights` if provided.
- **Risk:** weight expression applied to randoms uses the same `WEIGHT_SYSTOT * WEIGHT_CP * WEIGHT_NOZ * WEIGHT_FKP` as for data. This can be incorrect depending on catalog conventions (systematic weights should often be 1 for randoms). Cannot verify without data.

### D3. Weight expression
- Expression comes from dataset config; there is no separate randoms weight policy.
- **Potential failure:** weighted Landy–Szalay normalization in `landy_szalay()` ignores weight sums and uses raw counts of objects.

### D4. Selection functions
- Redshift cut applied symmetrically to data and randoms.
- Region handling: code defaults to `NGC` and does not loop over `SGC` or combined regions.

**Data integrity status:** **FAIL** — missing evidence and likely incorrect weighted normalization.

---

## E) Environment Metric (E) Audit

**Preregistered E1:** line-integrated smoothed overdensity along pair paths, normalized by median/MAD, per-galaxy mean over pairs.

**Code observations:**
- `compute_e1()` performs line integrals using `line_integral()` with subsampled pairs.
- `line_integral()` integrates over *parameter t* in [0,1] with `np.trapz(values, ts)` and does **not** multiply by physical path length — units are not correct.
- `compute_environment()` sets `per_galaxy` to **trilinear sampled δ at galaxy positions**, not the mean E1 over pairs.
- Normalization uses **mean/std**, not **median/MAD**.
- Smoothing radius and other parameters are read from `prereg["analysis"]["overlap_metric"]`, which does not exist in the locked preregistration file.
- Pair sampling is limited to ~200 pairs in `run_pipeline.py`/`run_stage.py`, not preregistered.

**Diagnostics requested (E distribution, systematics correlations):** cannot be produced due to missing outputs and systematics maps.

**Risk rating:** **HIGH** — E definition and normalization are inconsistent with preregistration, and line-integral implementation appears physically incorrect.

---

## F) Correlation Function and Covariance Audit

### F1. Landy–Szalay implementation
- Formula is correct, but normalization uses raw object counts, ignoring weights.
- Weighted pair counts from TreeCorr are not normalized by total weights.
- **Status:** **FAIL**

### F2. Wedge construction
- `wedge_xi()` uses mu-bin centers correctly.
- **Bug:** `run_pipeline.py` passes `tuple(wedges["tangential"])` where `wedges["tangential"]` is a dict (`mu_min`, `mu_max`, `label`), so the tuple becomes dict keys, not numeric bounds. This will cause incorrect wedge selection.

### F3. Covariance method
- Preregistration requires mocks/jackknife. Pipeline uses a **placeholder identity covariance**.
- **Status:** **FAIL**

### F4. Noise fitting risks
- Covariance insufficient (placeholder), no regularization or Hartlap correction implemented.

**Correlation/covariance status:** **FAIL**

---

## G) BAO Template Fitting Audit

### G1. Template model
- `bao_template.py` defines a **placeholder** damped sinusoid, not Eisenstein–Hu.

### G2. Nuisance model
- `fitting.py` supports only polynomial terms `[a0, a1/s, a2/s^2]`, but is not connected to the pipeline.

### G3. Fit range and binning
- Preregistered 60–160 h⁻¹ Mpc range is not used in the pipeline; no integration with the correlation output.

### G4. Optimizer/inference
- `fit_wedge()` uses grid search; no L-BFGS-B as preregistered. Not invoked anywhere.

### G5. Sensitivity check (mocks)
- Not possible: mock infrastructure not invoked, outputs missing.

**BAO fitting status:** **FAIL**

---

## H) Hierarchical β Inference Audit (Blinded)

### H1. Method
- Implemented `two_step_beta()` in `src/bao_overlap/hierarchical.py` (weighted linear regression).
- `run_pipeline.py` uses `two_step_beta()` **with placeholder `alpha_bins = 1` and `alpha_cov = I*0.01`**, not actual BAO fits.

### H2. β_blinded=0.2053 provenance
- **No evidence found** in code or outputs of a recorded `beta_blinded=0.2053` value. No `today_results` outputs exist for verification.

### H3. Blinding application
- No call to `blind_results()` in pipeline; therefore, β is not blinded prior to persistence.

**Hierarchical inference status:** **FAIL**

---

## Missing Evidence and Required Artifacts

To complete a high-stakes audit, the following **must** be provided in the environment:
- `today_results/paper_package/` (as specified) including:
  - `prereg_hash.txt`, audit logs, stage logs
  - environment snapshot (`environment.txt`)
  - `blinded_results.json`
  - correlation outputs, covariance matrices
  - figures with blinded annotations
- `today_results/execution.log` and stage-by-stage logs
- Mock output summaries and covariance mock counts
- QA metrics (counts, RA/DEC/Z ranges, weight distributions, NaN checks)

Without these, scientific validity and reproducibility cannot be verified.

---

## Required Remediations (before any unblinding)

1. **Fix preregistration schema linkage.** Ensure the code reads the locked `configs/preregistration.yaml` structure **as-is**. Eliminate `prereg["analysis"]` references or update the prereg file (not allowed post-lock).
2. **Enforce blinding in outputs.** Only store encrypted/offset values during blinded stages; forbid true β/σβ from being written prior to unblinding.
3. **Implement full BAO fitting with preregistered template + nuisance model.** Placeholder models must be replaced with the preregistered Eisenstein–Hu + polynomial model and L-BFGS-B optimization.
4. **Implement and log covariance method.** Mocks/jackknife must be wired into pipeline with sample counts and any corrections.
5. **Fix wedge parameter passing and region loops.** Use actual numeric wedge bounds and iterate NGC/SGC/combined as preregistered.
6. **Provide full run outputs.** Ensure `today_results/paper_package/` is included for audit verification.

---

## Appendix: Key Evidence Locations

- **Preregistration file:** `configs/preregistration.yaml`
- **Prereg hash in docs:** `preregistration.md` (hash matches computed)
- **Run config:** `configs/runs/eboss_lrgpcmass_default.yaml`
- **Pipeline:** `scripts/run_pipeline.py`, `scripts/run_stage.py`, `run_analysis.sh`
- **Blinding:** `src/bao_overlap/blinding.py`, `src/bao_overlap/reporting.py`
- **Correlation:** `src/bao_overlap/correlation.py`
- **Environment metric:** `src/bao_overlap/overlap_metric.py`, `src/bao_overlap/density_field.py`
- **BAO fitting:** `src/bao_overlap/bao_template.py`, `src/bao_overlap/fitting.py`
- **Inference:** `src/bao_overlap/hierarchical.py`

---

**End of report.**
