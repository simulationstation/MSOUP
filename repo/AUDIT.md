# Density Grid Sizing Bug Audit

## Executive Summary
The density grid was built with a fixed grid size and scalar cell size that did not reflect the actual catalog extents. This created grids that were far too small, producing out-of-range indices during trilinear sampling and invalidating E1 line integrals. The fix introduces a `GridSpec` with per-axis cell sizes derived from the catalog extent plus padding, updates trilinear sampling and smoothing to use per-axis spacing, and removes hard-coded grid sizing from pipeline scripts.

## Before Evidence (from latest failing run)
- Data XYZ span was approximately **[1998, 3869, 2071] h^-1 Mpc**.
- Grid shape **(64, 64, 64)** with **cell_size = 5.0** implied coverage of **320 h^-1 Mpc per axis**, far smaller than the data extent.
- `density_field.py` built edges implying ~66 h^-1 Mpc bins but stored `cell_size=5.0`; `trilinear_sample` used 5.0 and computed absurd indices (e.g., 459).
- Result: **All E1 integrals invalid** due to sampling outside the grid (0/100 valid in diagnostics).

## After Evidence (Grid Diagnostic Output)
- The diagnostic script ran but **could not load catalogs in this environment**, so it appended a Grid Diagnostic section with a `FileNotFoundError`. Because the data files are missing, the script could not report valid E1 fractions. This is documented in the Grid Diagnostic section below.

## Code Locations Updated
- `repo/src/bao_overlap/density_field.py`
  - Added `GridSpec` dataclass and `build_grid_spec`.
  - Updated `build_density_field` to accept `GridSpec`.
  - Updated `gaussian_smooth`, `trilinear_sample`, and `sample_with_mask` to use per-axis `cell_sizes` and correct origin.
- `repo/src/bao_overlap/overlap_metric.py`
  - Updated `compute_e2` sampling and step sizing to use per-axis `cell_sizes`.
- `repo/scripts/run_pipeline.py`
  - Removed hard-coded `grid_size=64` and `cell_size=5.0`.
  - Compute grid parameters from catalog extents + padding.
- `repo/scripts/run_stage.py`
  - Uses `build_grid_spec` and new grid parameters.
- `repo/scripts/grid_diagnostic.py`
  - New diagnostic script that appends a “Grid Diagnostic” section to `AUDIT.md`.
- Tests:
  - `repo/tests/test_grid_covers_span.py`
  - `repo/tests/test_trilinear_sample_consistency.py`
  - `repo/tests/test_e1_not_all_invalid_toy.py`

## How to Reproduce (Blinded)
- `PYTHONPATH=src python scripts/grid_diagnostic.py --config configs/runs/eboss_lrgpcmass_default.yaml`
- `PYTHONPATH=src python scripts/run_pipeline.py --config configs/runs/eboss_lrgpcmass_default.yaml --dry-run`

## Test Results Summary
- `PYTHONPATH=repo/src pytest repo/tests/test_grid_covers_span.py repo/tests/test_trilinear_sample_consistency.py repo/tests/test_e1_not_all_invalid_toy.py` **PASS**

## Exact Commands Run
1. `ls`
2. `find .. -name AGENTS.md -print`
3. `sed -n '1,220p' repo/src/bao_overlap/density_field.py`
4. `sed -n '1,200p' repo/scripts/run_pipeline.py`
5. `sed -n '200,400p' repo/scripts/run_pipeline.py`
6. `rg -n "trilinear" -n repo/src/bao_overlap`
7. `sed -n '1,200p' repo/src/bao_overlap/overlap_metric.py`
8. `sed -n '200,320p' repo/src/bao_overlap/overlap_metric.py`
9. `rg -n "cell_size|grid" repo/configs repo/src -g"*.yaml"` (exit code 1; no matches)
10. `rg -n "cell_size|grid" repo/configs` (exit code 1; no matches)
11. `ls repo/configs/runs`
12. `sed -n '1,200p' repo/configs/runs/eboss_lrgpcmass_default.yaml`
13. `rg -n "cell|grid|padding|density" configs/preregistration.yaml` (exit code 2; file not found in cwd)
14. `ls /workspace/MSOUP/configs`
15. `ls /workspace/MSOUP/repo/configs`
16. `rg -n "cell|grid|padding|density" /workspace/MSOUP/repo/configs/preregistration.yaml`
17. `sed -n '1,200p' /workspace/MSOUP/repo/configs/preregistration.yaml`
18. `rg -n "build_density_field" -n`
19. `sed -n '1,120p' repo/scripts/run_stage.py`
20. `rg -n "cell_size" repo/src/bao_overlap`
21. `sed -n '480,560p' repo/scripts/run_pipeline.py`
22. `rg -n "grid_size|cell_size" repo/scripts/run_pipeline.py`
23. `rg -n "cell_size" repo/src repo/scripts`
24. `cat <<'EOF' > /workspace/MSOUP/repo/scripts/grid_diagnostic.py ... EOF`
25. `cat <<'EOF' > /workspace/MSOUP/repo/tests/test_grid_covers_span.py ... EOF`
26. `cat <<'EOF' > /workspace/MSOUP/repo/tests/test_trilinear_sample_consistency.py ... EOF`
27. `cat <<'EOF' > /workspace/MSOUP/repo/tests/test_e1_not_all_invalid_toy.py ... EOF`
28. `sed -n '70,170p' /workspace/MSOUP/repo/scripts/grid_diagnostic.py`
29. `python repo/scripts/grid_diagnostic.py --config repo/configs/runs/eboss_lrgpcmass_default.yaml` (failed: ModuleNotFoundError: bao_overlap)
30. `PYTHONPATH=repo/src python repo/scripts/grid_diagnostic.py --config repo/configs/runs/eboss_lrgpcmass_default.yaml` (failed: FileNotFoundError: configs/preregistration.yaml due to cwd)
31. `PYTHONPATH=src python scripts/grid_diagnostic.py --config configs/runs/eboss_lrgpcmass_default.yaml` (appended Grid Diagnostic; data files missing)
32. `PYTHONPATH=repo/src pytest repo/tests/test_grid_covers_span.py repo/tests/test_trilinear_sample_consistency.py repo/tests/test_e1_not_all_invalid_toy.py`
33. `cat <<'EOF' > /workspace/MSOUP/repo/AUDIT.md ... EOF`
34. `PYTHONPATH=src python scripts/grid_diagnostic.py --config configs/runs/eboss_lrgpcmass_default.yaml` (appended Grid Diagnostic; data files missing)
35. `tail -n 20 /workspace/MSOUP/repo/AUDIT.md`

## Grid Diagnostic
Timestamp: 2025-12-24T02:06:25.064751Z
Config: configs/runs/eboss_lrgpcmass_default.yaml
Grid params: target_cell_size=10.0, padding=50.0, max_n_per_axis=512

Grid diagnostic failed with exception:
FileNotFoundError: [Errno 2] No such file or directory: '../data/eboss/dr16/LRGpCMASS/v1/eBOSS_LRGpCMASS_clustering_data-NGC-vDR16.fits'

## Grid Diagnostic
Timestamp: 2025-12-24T02:13:03.849699Z
Config: configs/runs/eboss_lrgpcmass_default.yaml
Grid params: target_cell_size=10.0, padding=50.0, max_n_per_axis=512

Grid diagnostic failed with exception:
TypeError: radec_to_cartesian() got an unexpected keyword argument 'omega_b'

## Grid Diagnostic
Timestamp: 2025-12-24T02:13:49.184040Z
Config: configs/runs/eboss_lrgpcmass_default.yaml
Grid params: target_cell_size=10.0, padding=50.0, max_n_per_axis=512

Region: NGC
Span (h^-1 Mpc): [2086.041278121041, 3995.094296663516, 2244.6043940772925]
Grid shape: (219, 410, 235)
Cell sizes (h^-1 Mpc): [9.98192364438819, 9.98803486991119, 9.97703997479698]
Coverage (h^-1 Mpc): [2186.0412781210134, 4095.0942966635876, 2344.6043940772906]
Estimated grid memory (GB): 0.079
Random sample invalid fraction: 0.000
E1 diagnostic: mean attempted pairs per galaxy=0.52, mean valid pairs=0.52, invalid fraction=0.000, finite fraction=0.390

Region: SGC
Span (h^-1 Mpc): [1209.8933674539048, 3086.8095891189932, 1700.5550095038734]
Grid shape: (131, 319, 181)
Cell sizes (h^-1 Mpc): [9.999186011098573, 9.989998711971793, 9.947817732065573]
Coverage (h^-1 Mpc): [1309.893367453913, 3186.8095891190023, 1800.5550095038689]
Estimated grid memory (GB): 0.028
Random sample invalid fraction: 0.000
E1 diagnostic: mean attempted pairs per galaxy=0.98, mean valid pairs=0.98, invalid fraction=0.000, finite fraction=0.610
