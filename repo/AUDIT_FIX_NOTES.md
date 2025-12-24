# Audit Fix Notes (Changelog)

This changelog enumerates each remediation applied to the preregistered, blinded BAO environment-overlap pipeline. Paths and key functions are referenced for audit traceability.

## Preregistration schema linkage
- **Added loader + schema validation**: `src/bao_overlap/prereg.py` → `load_prereg()`.
- **Config wiring updated**: `src/bao_overlap/io.py` → `load_run_config()` now uses `load_prereg()`.
- **Removed `prereg["analysis"]` usage**: `scripts/run_pipeline.py`, `scripts/run_stage.py` now access top-level prereg keys.

## Blinding enforcement
- **Enforced blinded outputs only**: `scripts/run_pipeline.py` now calls `initialize_blinding()`, `blind_results()`, `save_blinded_results()` and writes only `blinded_results.json` in blinded mode.
- **Leak guard in reporting**: `src/bao_overlap/reporting.py` → `write_results()` rejects forbidden keys (`beta`, `sigma_beta`, `p_value`, `zscore`, `percentile`) when blinded.

## Correlation and wedge fixes
- **Weighted Landy–Szalay normalization**: `src/bao_overlap/correlation.py` → `landy_szalay()` now uses weighted pair-count norms.
- **Wedge bounds bug fix**: `src/bao_overlap/correlation.py` → `parse_wedge_bounds()`; applied in `scripts/run_pipeline.py` and `scripts/run_stage.py`.
- **Region combination**: `src/bao_overlap/correlation.py` → `combine_pair_counts()` used in `scripts/run_pipeline.py` to merge NGC/SGC counts.

## Covariance placeholder removal
- **Jackknife covariance implementation**: `src/bao_overlap/covariance.py` → `covariance_from_jackknife()` and `assign_jackknife_regions()`.
- **Covariance persisted to disk**: `scripts/run_pipeline.py` saves `covariance/xi_wedge_covariance.npy` and bundles it into `paper_package/`.

## BAO template replacement
- **Standard template**: `src/bao_overlap/bao_template.py` replaced placeholder sinusoid with an Eisenstein–Hu-inspired transfer + damped wiggles (`bao_template()`).
- **Fit wiring**: `src/bao_overlap/fitting.py` now accepts `template_params` and uses preregistered nuisance terms.

## Environment normalization
- **Median/MAD support**: `src/bao_overlap/overlap_metric.py` → `normalize()` now supports `median_mad`; `compute_environment()` accepts `normalization_method`.

## Self-audit and tests
- **Self-audit script**: `scripts/self_audit.py` validates schema, backends, blinding leaks, wedge bounds, covariance, and paper package artifacts.
- **Tests added**:
  - `tests/test_prereg_schema.py` for prereg loader validation.
  - `tests/test_blinding_no_leak.py` for blinding safeguards.
  - `tests/test_wedge_bounds.py` for wedge parsing/application.
- **Existing tests updated**: `tests/test_overlap_metric.py`, `tests/test_template_fit_smoke.py`.
