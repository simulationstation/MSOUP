"""
CLI entrypoint for the MV-O1B pocket/intermittency test.

Modes:
  sanity: Full coverage with minimal resampling (quick validation)
  full: Full analysis with empirical p-values
  robustness: Full + robustness sweeps across thresholds/masking/nulls
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .candidates import candidates_to_frame, extract_candidates
from .config import PocketConfig, load_config
from .geometry import GeometryResult, evaluate_geometry
from .io_igs import (
    group_by_sensor,
    load_clk_from_cache,
    load_positions_from_cache,
    resolve_paths,
)
from .io_mag import group_by_station, iter_magnetometer
from .guardrails import assert_finite_metrics
from .nulls import NullType
from .plots import generate_all_plots
from .preprocess import drop_bad_windows, standardize_magnetometer
from .report import compute_counts_coverage, generate_report, generate_robustness_report
from .stats import GlobalStats, compute_debiased_stats
from .windows import Window, derive_windows, windows_to_frame


def _process_clk_batch(batch: pd.DataFrame, cadence_seconds: float) -> Dict[str, pd.DataFrame]:
    df = batch.copy()
    df["value"] = df["bias_s"]
    df["sensor"] = df["satellite"]
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "sensor", "value"]]
    return group_by_sensor(df, "sensor")


def load_all_data(cfg: PocketConfig) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, List[Path]]:
    """Load CLK, SP3, and magnetometer data. Returns per_sensor_series, positions, mag_paths."""
    cache_dir = Path(cfg.output.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    clk_paths = resolve_paths(cfg.data.clk_glob, cfg.data.max_clk_files)
    sp3_paths = resolve_paths(cfg.data.sp3_glob, cfg.data.max_clk_files)  # SP3 follows CLK limit
    mag_paths = resolve_paths(cfg.data.magnetometer_glob, cfg.data.max_mag_files)

    print(f"Found {len(clk_paths)} CLK files, {len(sp3_paths)} SP3 files, {len(mag_paths)} MAG files")
    if cfg.data.max_clk_files > 0:
        print(f"  (limited to {cfg.data.max_clk_files} CLK/SP3 files for memory safety)")
    if cfg.data.max_mag_files > 0:
        print(f"  (limited to {cfg.data.max_mag_files} MAG files for memory safety)")

    if not clk_paths:
        raise RuntimeError(f"No CLK files found matching: {cfg.data.clk_glob}")

    clk_batches = load_clk_from_cache(cache_dir, clk_paths, cfg.data.chunk_hours)
    positions = load_positions_from_cache(cache_dir, sp3_paths)

    per_sensor_series: Dict[str, pd.DataFrame] = {}

    # Process CLK batches
    for batch in tqdm(clk_batches, desc="Processing CLK"):
        per_sensor_series.update(_process_clk_batch(batch, cfg.preprocess.cadence_seconds))

    # Process magnetometer data
    for batch in tqdm(list(iter_magnetometer(mag_paths, cfg.data.chunk_hours)), desc="Processing MAG"):
        mag = standardize_magnetometer(batch, cfg.preprocess)
        mag = drop_bad_windows(mag)
        per_station = group_by_station(mag)
        for station, df in per_station.items():
            df = df.rename(columns={"btot": "value", "station": "sensor"})
            if station in per_sensor_series:
                per_sensor_series[station] = pd.concat([per_sensor_series[station], df[["time", "sensor", "value"]]])
            else:
                per_sensor_series[station] = df[["time", "sensor", "value"]]

    print(f"Loaded {len(per_sensor_series)} sensors total")
    return per_sensor_series, positions, mag_paths


def run_single_analysis(
    per_sensor_series: Dict[str, pd.DataFrame],
    positions: pd.DataFrame,
    cfg: PocketConfig,
    label: str = "default",
    mode: str = "full",
) -> Tuple[GlobalStats, GeometryResult, pd.DataFrame, pd.DataFrame, List[Window]]:
    """Run a single analysis pass with given config."""
    windows = derive_windows(per_sensor_series, cfg.preprocess, cfg.windows)
    window_frame = windows_to_frame(windows)

    candidates = extract_candidates(per_sensor_series, windows, cfg.candidates)
    candidate_frame = candidates_to_frame(candidates)

    print(f"[{label}] Extracted {len(candidate_frame)} candidates from {len(windows)} windows")

    stats = compute_debiased_stats(candidate_frame, window_frame, cfg.stats, cfg.candidates)

    if cfg.geometry.enable_geometry_test and not positions.empty:
        geom = evaluate_geometry(candidate_frame, positions, cfg.geometry, run_mode=mode)
    else:
        geom = GeometryResult(chi2_obs=None, null_chi2s=None, p_value=None, n_events=0, speed_kmps=None, normal=np.zeros(3), status="NOT_TESTABLE", reason="geometry disabled or positions unavailable")

    return stats, geom, candidate_frame, window_frame, windows


def run_robustness_sweeps(
    per_sensor_series: Dict[str, pd.DataFrame],
    positions: pd.DataFrame,
    base_cfg: PocketConfig,
) -> List[Dict[str, Any]]:
    """Run robustness sweeps across thresholds, masking, and null types."""
    results = []

    thresholds = base_cfg.robustness.thresholds or [3.0, 4.0, 5.0]

    masking_variants = base_cfg.robustness.mask_variants or [
        ("full_coverage", 0.6),
        ("relaxed_coverage", 0.4),
    ]

    null_types = base_cfg.robustness.null_variants or [
        ("conditioned_resample", NullType.CONDITIONED_RESAMPLE),
        ("block_bootstrap", NullType.BLOCK_BOOTSTRAP),
    ]

    total_runs = len(thresholds) * len(masking_variants) * len(null_types)
    print(f"\nRunning {total_runs} robustness configurations...")

    run_idx = 0
    for thresh in thresholds:
        for mask_name, coverage in masking_variants:
            for null_name, null_type in null_types:
                run_idx += 1
                label = f"thresh={thresh}_mask={mask_name}_null={null_name}"
                print(f"\n[{run_idx}/{total_runs}] {label}")

                # Clone config and modify
                cfg = base_cfg.clone()
                cfg.candidates.threshold_sigma = thresh
                cfg.windows.require_fractional_coverage = coverage
                cfg.stats.null_type = null_type
                if base_cfg.robustness.n_resamples:
                    cfg.stats.null_realizations = base_cfg.robustness.n_resamples

                try:
                    stats, geom, cand_df, win_df, _ = run_single_analysis(
                        per_sensor_series, positions, cfg, label, mode="robustness"
                    )

                    results.append({
                        "label": label,
                        "threshold_sigma": thresh,
                        "masking": mask_name,
                        "coverage_threshold": coverage,
                        "null_type": null_name,
                        "n_candidates": len(cand_df),
                        "n_windows": len(win_df),
                        "fano_obs": stats.fano_obs,
                        "fano_null_mean": stats.fano_null_mean,
                        "fano_null_std": stats.fano_null_std,
                        "fano_z": stats.fano_z,
                        "c_obs": stats.c_obs,
                        "c_excess": stats.c_excess,
                        "c_excess_std": stats.c_excess_std,
                        "c_z": stats.c_z,
                        "p_fano": stats.p_fano,
                        "p_clustering": stats.p_clustering,
                        "geom_chi2": geom.chi2_obs,
                        "geom_p": geom.p_value,
                        "K1_killed": stats.p_clustering > 0.05 or stats.c_excess <= 0,
                        "K2_killed": geom.status == "COMPLETED" and geom.p_value is not None and geom.p_value > 0.05,
                        "geom_status": geom.status,
                        "geom_reason": geom.reason,
                    })
                except Exception as e:
                    print(f"  Error: {e}")
                    results.append({
                        "label": label,
                        "threshold_sigma": thresh,
                        "masking": mask_name,
                        "null_type": null_name,
                        "error": str(e),
                    })

    return results


def create_output_dir(base_dir: str) -> Path:
    """Create timestamped output directory and update 'latest' symlink."""
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    # Update 'latest' symlink
    latest = base / "latest"
    if latest.is_symlink():
        latest.unlink()
    elif latest.exists():
        import shutil
        shutil.rmtree(latest)
    latest.symlink_to(timestamp, target_is_directory=True)

    return run_dir


def generate_summary_json(
    output_dir: Path,
    stats: GlobalStats,
    geom: GeometryResult,
    counts_coverage: Dict[str, Any],
    run_mode: str,
    robustness_results: Optional[List[Dict]] = None,
) -> Path:
    """Generate summary.json with all key metrics."""
    summary = {
        "timestamp": dt.datetime.now().isoformat(),
        "kill_conditions": {
            "K1_pockets_beyond_selection": {
                "description": "C_excess consistent with 0 and Fano explainable by Cox-rate variability",
                "killed": stats.p_clustering > 0.05 or stats.c_excess <= 0,
                "c_excess": stats.c_excess,
                "c_excess_std": stats.c_excess_std,
                "p_clustering": stats.p_clustering,
            },
            "K2_no_propagation_geometry": {
                "description": "Fitted propagation parameters indistinguishable from shuffled nulls",
                "killed": geom.status == "COMPLETED" and geom.p_value is not None and geom.p_value > 0.05,
                "status": geom.status,
                "reason": geom.reason,
                "chi2_obs": geom.chi2_obs if geom.chi2_obs is not None else None,
                "p_value": geom.p_value if geom.p_value is not None else None,
                "n_events_tested": geom.n_events,
            },
            "K3_no_cross_channel_coherence": {
                "description": "Cross-band/modality consistency required if claimed gravitational",
                "killed": True,  # Default to killed unless proven otherwise
                "notes": "Requires explicit cross-modality analysis",
            },
        },
        "statistics": {
            "fano_obs": stats.fano_obs,
            "fano_null_mean": stats.fano_null_mean,
            "fano_null_std": stats.fano_null_std,
            "fano_z": stats.fano_z,
            "p_fano": stats.p_fano,
            "c_obs": stats.c_obs,
            "c_null_mean": stats.c_null_mean,
            "c_null_std": stats.c_null_std,
            "c_excess": stats.c_excess,
            "c_excess_std": stats.c_excess_std,
            "c_z": stats.c_z,
            "p_clustering": stats.p_clustering,
            "p_clustering_two_sided": stats.p_clustering_two_sided,
            "n_resamples": stats.n_resamples,
        },
        "geometry": {
            "status": geom.status,
            "reason": geom.reason,
            "chi2_obs": geom.chi2_obs if geom.chi2_obs is not None else None,
            "p_value": geom.p_value if geom.p_value is not None else None,
            "n_events": geom.n_events,
        },
        "counts_coverage": counts_coverage,
    }

    if robustness_results:
        summary["robustness"] = robustness_results

    allow_null = {"geometry.chi2_obs", "geometry.p_value", "kill_conditions.K2_no_propagation_geometry.chi2_obs", "kill_conditions.K2_no_propagation_geometry.p_value"}
    assert_finite_metrics(summary, allow_null, mode=run_mode)

    path = output_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MV-O1B pocket/intermittency test.")
    parser.add_argument("--config", type=str, default="configs/pocket_default.yaml",
                        help="Path to YAML config.")
    parser.add_argument("--mode", type=str, choices=["sanity", "full", "robustness"],
                        default="full", help="Run mode: sanity (quick), full, or robustness")
    parser.add_argument("--robustness-thresholds", type=str, help="Comma-separated thresholds for robustness sweeps")
    parser.add_argument("--robustness-masks", type=str, help="Comma-separated mask variants name:coverage (e.g., strict:0.6,lenient:0.4)")
    parser.add_argument("--robustness-nulls", type=str, help="Comma-separated null variants (conditioned_resample,block_bootstrap,time_shift)")
    parser.add_argument("--robustness-resamples", type=int, help="Override number of null resamples for robustness sweeps")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.robustness_thresholds:
        cfg.robustness.thresholds = [float(x) for x in args.robustness_thresholds.split(",") if x]
    if args.robustness_masks:
        masks = []
        for item in args.robustness_masks.split(","):
            if not item:
                continue
            name, coverage = item.split(":")
            masks.append((name, float(coverage)))
        cfg.robustness.mask_variants = masks
    if args.robustness_nulls:
        cfg.robustness.null_variants = [(n, NullType(n)) for n in args.robustness_nulls.split(",") if n]
    if args.robustness_resamples:
        cfg.robustness.n_resamples = args.robustness_resamples

    # Adjust config based on mode
    if args.mode == "sanity":
        print("=== SANITY MODE: Minimal resampling, full coverage ===")
        cfg.stats.null_realizations = 32  # Minimal but sufficient for validation
        cfg.geometry.n_shuffles = 50
    elif args.mode == "full":
        print("=== FULL MODE: Full empirical p-values ===")
        # Use config defaults (256 resamples)
    elif args.mode == "robustness":
        print("=== ROBUSTNESS MODE: Full + sweeps ===")

    # Create output directory
    output_dir = create_output_dir(cfg.output.results_dir)
    print(f"Output directory: {output_dir}")

    # Load all data
    print("\n=== Loading data ===")
    per_sensor_series, positions, mag_paths = load_all_data(cfg)

    # Run main analysis
    print("\n=== Running main analysis ===")
    stats, geom, candidate_frame, window_frame, windows = run_single_analysis(
        per_sensor_series, positions, cfg, "main", mode=args.mode
    )

    thresholds_for_counts = cfg.candidates.robustness_thresholds or sorted({
        cfg.candidates.subthreshold_sigma,
        cfg.candidates.threshold_sigma,
        max(cfg.candidates.threshold_sigma, cfg.candidates.subthreshold_sigma) + 1.0,
    })
    counts_coverage = compute_counts_coverage(
        per_sensor_series=per_sensor_series,
        windows=windows,
        candidate_frame=candidate_frame,
        neighbor_time=cfg.candidates.neighbor_time_seconds,
        neighbor_angle=cfg.candidates.neighbor_angle_degrees,
        thresholds=thresholds_for_counts,
        geom=geom,
    )

    # Run robustness sweeps if requested
    robustness_results = None
    if args.mode == "robustness":
        print("\n=== Running robustness sweeps ===")
        robustness_results = run_robustness_sweeps(per_sensor_series, positions, cfg)

        # Generate robustness report
        robustness_path = generate_robustness_report(output_dir, robustness_results)
        print(f"Robustness report: {robustness_path}")

    # Generate plots
    print("\n=== Generating plots ===")
    generate_all_plots(
        output_dir=output_dir,
        candidate_frame=candidate_frame,
        window_frame=window_frame,
        stats=stats,
        geom=geom,
        cfg=cfg,
    )

    # Generate main report
    print("\n=== Generating reports ===")
    report_path = generate_report(
        output_dir=output_dir,
        stats=stats,
        geom=geom,
        cross_modality_ok=bool(mag_paths) and not candidate_frame.empty,
        run_label=cfg.run_label,
        config_path=Path(args.config),
        counts_coverage=counts_coverage,
    )

    # Generate summary JSON
    summary_path = generate_summary_json(output_dir, stats, geom, counts_coverage, args.mode, robustness_results)

    print(f"\n=== Results ===")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    print(f"Output dir: {output_dir}")

    # Print kill condition summary
    print("\n=== Kill Condition Summary ===")
    k1_killed = stats.p_clustering > 0.05 or stats.c_excess <= 0
    k2_killed = geom.status == "COMPLETED" and geom.p_value is not None and geom.p_value > 0.05
    k3_killed = True  # Default

    print(f"K1 (No pockets beyond selection): {'KILLED' if k1_killed else 'NOT KILLED'}")
    print(f"   C_excess = {stats.c_excess:.4f} ± {stats.c_excess_std:.4f}, p = {stats.p_clustering:.4f}")
    print(f"K2 (No propagation geometry): {'KILLED' if k2_killed else 'NOT KILLED'}")
    if geom.status == "COMPLETED" and geom.p_value is not None:
        print(f"   χ² = {geom.chi2_obs:.2f}, p = {geom.p_value:.4f}")
    elif geom.status == "NOT_TESTABLE":
        print(f"   Geometry NOT TESTABLE: {geom.reason}")
    else:
        print(f"   (geometry test not run or insufficient events)")
    print(f"K3 (No cross-channel coherence): {'KILLED' if k3_killed else 'NOT KILLED'}")
    print(f"   (requires explicit cross-modality analysis)")


if __name__ == "__main__":
    main()
