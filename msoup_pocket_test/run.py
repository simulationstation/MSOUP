"""
CLI entrypoint for the MV-O1B pocket/intermittency test.

Modes:
  sanity: Full coverage with minimal resampling (quick validation)
  full: Full analysis with empirical p-values
  robustness: Full + robustness sweeps across thresholds/masking/nulls
  dry-run: Audit mode - prints counts without loading data
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .candidates import candidates_to_frame, extract_candidates
from .config import PocketConfig, load_config
from .geometry import GeometryResult, evaluate_geometry
from .io_igs import (
    count_paths,
    estimate_file_coverage,
    group_by_sensor,
    iter_clk_from_cache_streaming,
    load_positions_from_cache,
    resolve_paths,
)
from .io_mag import group_by_station, iter_magnetometer
from .guardrails import assert_finite_metrics
from .nulls import NullType
from .plots import generate_all_plots
from .preprocess import drop_bad_windows, standardize_magnetometer
from .report import compute_counts_coverage, generate_report, generate_robustness_report
from .resources import MemoryLimitExceeded, PipelineAborted, ResourceMonitor, is_wsl
from .stats import GlobalStats, compute_debiased_stats
from .windows import Window, derive_windows, windows_to_frame


def _process_clk_batch(batch: pd.DataFrame, cadence_seconds: float) -> Dict[str, pd.DataFrame]:
    df = batch.copy()
    df["value"] = df["bias_s"]
    df["sensor"] = df["satellite"]
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "sensor", "value"]]
    return group_by_sensor(df, "sensor")


def run_dry_run(cfg: PocketConfig) -> Dict[str, Any]:
    """
    Dry-run mode: print counts and estimates without loading data.

    This is essential for auditing before full runs to avoid memory issues.
    """
    print("=== DRY-RUN MODE ===")
    print("Counting files and estimating coverage WITHOUT loading data...\n")

    # Count files
    clk_count = count_paths(cfg.data.clk_glob, cfg.data.max_clk_files)
    sp3_count = count_paths(cfg.data.sp3_glob, cfg.data.max_clk_files)
    mag_count = count_paths(cfg.data.magnetometer_glob, cfg.data.max_mag_files)

    print(f"CLK files: {clk_count}")
    if cfg.data.max_clk_files > 0:
        print(f"  (limited to {cfg.data.max_clk_files} by config)")
    print(f"SP3 files: {sp3_count}")
    print(f"MAG files: {mag_count}")
    if cfg.data.max_mag_files > 0:
        print(f"  (limited to {cfg.data.max_mag_files} by config)")

    # Estimate CLK coverage
    clk_paths = resolve_paths(cfg.data.clk_glob, cfg.data.max_clk_files)
    if clk_paths:
        coverage = estimate_file_coverage(clk_paths)
        print(f"\nCLK coverage estimate (from sampling first 5 files):")
        if coverage["time_range"]:
            start, end = coverage["time_range"]
            duration = (end - start).total_seconds() / 3600
            print(f"  Time range: {start} to {end}")
            print(f"  Duration: {duration:.1f} hours")
        print(f"  Estimated sensors: {coverage['n_sensors_estimated']}")

    # Compute chunk estimates
    hours_per_file = 24  # typical
    total_hours = clk_count * hours_per_file
    chunk_hours = cfg.data.chunk_hours
    n_chunks = max(1, int(total_hours / chunk_hours))

    print(f"\nProcessing plan:")
    print(f"  Chunk size: {chunk_hours} hours")
    print(f"  Estimated chunks: {n_chunks}")
    print(f"  pair_mode: {cfg.resources.pair_mode}")
    print(f"  null_realizations: {cfg.stats.null_realizations}")
    print(f"  max_rss_gb: {cfg.resources.max_rss_gb}")
    print(f"  max_workers: {cfg.resources.max_workers}")

    # Memory estimate
    from .resources import estimate_memory_requirements
    mem_est = estimate_memory_requirements(
        n_files=clk_count,
        n_sensors=32,  # typical
        n_resamples=cfg.stats.null_realizations,
        chunk_hours=chunk_hours,
        pair_mode=cfg.resources.pair_mode,
    )

    print(f"\nMemory estimates:")
    print(f"  File loading: {mem_est['file_load_mb']:.1f} MB")
    print(f"  Candidates: {mem_est['candidate_mb']:.1f} MB")
    print(f"  Null catalogs: {mem_est['null_mb']:.1f} MB (streaming, one at a time)")
    print(f"  Clustering: {mem_est['clustering_mb']:.1f} MB")
    print(f"  Peak estimate: {mem_est['peak_estimate_gb']:.2f} GB")
    print(f"  System memory: {mem_est['system_memory_gb']:.1f} GB")
    print(f"  Safe: {'Yes' if mem_est['safe'] else 'NO - may exceed memory'}")

    if mem_est['recommendations']:
        print(f"\nRecommendations:")
        for rec in mem_est['recommendations']:
            print(f"  - {rec}")

    return {
        "mode": "dry-run",
        "clk_files": clk_count,
        "sp3_files": sp3_count,
        "mag_files": mag_count,
        "estimated_chunks": n_chunks,
        "memory_estimate": mem_est,
    }


def load_all_data(
    cfg: PocketConfig,
    monitor: ResourceMonitor | None = None
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, List[Path]]:
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

    if monitor:
        monitor.check("after_file_discovery")

    if not clk_paths:
        raise RuntimeError(f"No CLK files found matching: {cfg.data.clk_glob}")

    # Use streaming loader
    clk_batches = iter_clk_from_cache_streaming(cache_dir, clk_paths, cfg.data.chunk_hours)
    positions = load_positions_from_cache(cache_dir, sp3_paths)

    if monitor:
        monitor.check("after_positions_load")

    per_sensor_series: Dict[str, pd.DataFrame] = {}

    # Process CLK batches (streaming)
    batch_idx = 0
    for batch in tqdm(clk_batches, desc="Processing CLK"):
        per_sensor_series.update(_process_clk_batch(batch, cfg.preprocess.cadence_seconds))
        if monitor:
            monitor.check(f"clk_batch_{batch_idx}")
            monitor.mark_chunk_complete(f"clk_{batch_idx}")
        batch_idx += 1

    # Process magnetometer data
    mag_batch_idx = 0
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
        if monitor:
            monitor.check(f"mag_batch_{mag_batch_idx}")
            monitor.mark_chunk_complete(f"mag_{mag_batch_idx}")
        mag_batch_idx += 1

    print(f"Loaded {len(per_sensor_series)} sensors total")
    return per_sensor_series, positions, mag_paths


def run_single_analysis(
    per_sensor_series: Dict[str, pd.DataFrame],
    positions: pd.DataFrame,
    cfg: PocketConfig,
    label: str = "default",
    mode: str = "full",
    monitor: ResourceMonitor | None = None,
    unsafe: bool = False,
) -> Tuple[GlobalStats, GeometryResult, pd.DataFrame, pd.DataFrame, List[Window]]:
    """Run a single analysis pass with given config."""
    windows = derive_windows(per_sensor_series, cfg.preprocess, cfg.windows)
    window_frame = windows_to_frame(windows)

    candidates = extract_candidates(per_sensor_series, windows, cfg.candidates)
    candidate_frame = candidates_to_frame(candidates)

    if monitor:
        monitor.check("after_candidate_extraction")

    print(f"[{label}] Extracted {len(candidate_frame)} candidates from {len(windows)} windows")

    stats = compute_debiased_stats(candidate_frame, window_frame, cfg.stats, cfg.candidates, monitor=monitor, unsafe=unsafe)

    if cfg.geometry.enable_geometry_test and not positions.empty:
        geom = evaluate_geometry(candidate_frame, positions, cfg.geometry, run_mode=mode)
    else:
        geom = GeometryResult(chi2_obs=None, null_chi2s=None, p_value=None, n_events=0, speed_kmps=None, normal=np.zeros(3), status="NOT_TESTABLE", reason="geometry disabled or positions unavailable")

    return stats, geom, candidate_frame, window_frame, windows


def run_robustness_sweeps(
    per_sensor_series: Dict[str, pd.DataFrame],
    positions: pd.DataFrame,
    base_cfg: PocketConfig,
    monitor: ResourceMonitor | None = None,
    unsafe: bool = False,
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
                        per_sensor_series, positions, cfg, label, mode="robustness", monitor=monitor, unsafe=unsafe
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
    resources: Optional[dict] = None,
    robustness_results: Optional[List[Dict]] = None,
    aborted: bool = False,
    abort_reason: Optional[str] = None,
) -> Path:
    """Generate summary.json with all key metrics."""
    summary = {
        "timestamp": dt.datetime.now().isoformat(),
        "status": "ABORTED" if aborted else "COMPLETED",
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
            "pvalue_resolution": stats.pvalue_resolution,
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

    if resources:
        summary["resources"] = resources

    if robustness_results:
        summary["robustness"] = robustness_results

    if aborted:
        summary["abort_reason"] = abort_reason

    allow_null = {"geometry.chi2_obs", "geometry.p_value", "kill_conditions.K2_no_propagation_geometry.chi2_obs", "kill_conditions.K2_no_propagation_geometry.p_value"}
    assert_finite_metrics(summary, allow_null, mode=run_mode)

    path = output_dir / "summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return path


def generate_abort_report(output_dir: Path, monitor: ResourceMonitor, run_mode: str) -> Path:
    """Generate a partial report for an aborted run."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "REPORT.md"

    lines = [
        "# MV-O1B Pocket/Intermittency Test Report",
        "",
        f"**Generated:** {dt.datetime.utcnow().isoformat()}Z",
        f"**Status:** ABORTED",
        "",
        "---",
        "",
        "## Abort Information",
        "",
        f"**Reason:** {monitor.abort_reason or 'Unknown'}",
        f"**Last completed chunk:** {monitor.last_completed_chunk or 'None'}",
        f"**Peak RSS:** {monitor.peak_rss_gb:.2f} GB",
        f"**Max RSS limit:** {monitor.max_rss_gb:.2f} GB",
        "",
        "---",
        "",
        "## Recommended Actions",
        "",
        "1. Reduce `null_realizations` in config (try 64 or 128)",
        "2. Use `--mode sanity` for validation runs",
        "3. Increase `chunk_days` to process smaller time windows",
        "4. Ensure `pair_mode = binned` (not exact)",
        "5. Run with `max_clk_files` and `max_mag_files` limits",
        "6. Consider running on a machine with more RAM",
        "",
    ]

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Also write JSON
    abort_data = monitor.abort_report()
    abort_data["mode"] = run_mode
    json_path = output_dir / "summary.json"
    with open(json_path, "w") as f:
        json.dump(abort_data, f, indent=2)

    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MV-O1B pocket/intermittency test.")
    parser.add_argument("--config", type=str, default="configs/pocket_default.yaml",
                        help="Path to YAML config.")
    parser.add_argument("--mode", type=str, choices=["sanity", "full", "robustness", "dry-run"],
                        default="full", help="Run mode: sanity (quick), full, robustness, or dry-run")
    parser.add_argument("--dry-run", action="store_true", help="Alias for --mode dry-run")
    parser.add_argument("--robustness-thresholds", type=str, help="Comma-separated thresholds for robustness sweeps")
    parser.add_argument("--robustness-masks", type=str, help="Comma-separated mask variants name:coverage (e.g., strict:0.6,lenient:0.4)")
    parser.add_argument("--robustness-nulls", type=str, help="Comma-separated null variants (conditioned_resample,block_bootstrap,time_shift)")
    parser.add_argument("--robustness-resamples", type=int, help="Override number of null resamples for robustness sweeps")
    parser.add_argument("--unsafe", action="store_true", help="Allow resource-heavy execution (disables WSL safety overrides)")
    args = parser.parse_args()

    # Handle --dry-run alias
    if args.dry_run:
        args.mode = "dry-run"

    cfg = load_config(args.config)
    cfg.data.chunk_hours = cfg.resources.chunk_days * 24
    cfg.stats.n_processes = cfg.resources.max_workers
    cfg.stats.pair_mode = cfg.resources.pair_mode
    cfg.stats.time_bin_seconds = cfg.resources.time_bin_seconds
    cfg.stats.max_candidates_in_memory = cfg.resources.max_candidates_in_memory

    if args.mode == "sanity":
        cfg.stats.null_realizations = cfg.resources.resamples_sanity_default
    else:
        cfg.stats.null_realizations = cfg.resources.resamples_full_default

    if is_wsl() and not args.unsafe:
        print("Detected WSL2 — applying safe defaults (max_workers=1, binned pairs, bounded resamples)")
        cfg.resources.max_workers = 1
        cfg.stats.n_processes = 1
        cfg.stats.pair_mode = "binned"
        cfg.stats.null_realizations = cfg.resources.resamples_sanity_default if args.mode == "sanity" else min(
            cfg.resources.resamples_full_default, cfg.stats.null_realizations
        )
    elif is_wsl() and cfg.resources.max_workers > 1:
        print("WARNING: WSL2 detected: max_workers>1 may increase memory pressure.")

    # Handle dry-run mode
    if args.mode == "dry-run":
        result = run_dry_run(cfg)
        print("\n=== DRY-RUN COMPLETE ===")
        print("No data was loaded. Review the estimates above before running full analysis.")
        return

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
        # null_realizations already set from config's resamples_sanity_default
        cfg.geometry.n_shuffles = 50
    elif args.mode == "full":
        print("=== FULL MODE: Full empirical p-values ===")
        # Use config defaults (256 resamples)
    elif args.mode == "robustness":
        print("=== ROBUSTNESS MODE: Full + sweeps ===")

    monitor = ResourceMonitor(
        max_rss_gb=cfg.resources.max_rss_gb,
        rss_check_interval=cfg.resources.rss_check_interval_steps,
    )

    # Create output directory
    output_dir = create_output_dir(cfg.output.results_dir)
    print(f"Output directory: {output_dir}")

    try:
        # Load all data
        print("\n=== Loading data ===")
        per_sensor_series, positions, mag_paths = load_all_data(cfg, monitor)

        # Run main analysis
        print("\n=== Running main analysis ===")
        stats, geom, candidate_frame, window_frame, windows = run_single_analysis(
            per_sensor_series, positions, cfg, "main", mode=args.mode, monitor=monitor, unsafe=args.unsafe
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
            robustness_results = run_robustness_sweeps(per_sensor_series, positions, cfg, monitor=monitor, unsafe=args.unsafe)

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

        resource_snapshot = {
            "peak_rss_gb": monitor.peak_rss_gb,
            "max_workers": cfg.resources.max_workers,
            "chunk_days": cfg.resources.chunk_days,
            "pair_mode": cfg.stats.pair_mode,
            "time_bin_seconds": cfg.stats.time_bin_seconds,
            "n_resamples": stats.n_resamples,
            "max_rss_gb": cfg.resources.max_rss_gb,
            "rss_check_interval": cfg.resources.rss_check_interval_steps,
            "aborted": monitor.aborted,
        }

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
            resources=resource_snapshot,
        )

        # Generate summary JSON
        summary_path = generate_summary_json(output_dir, stats, geom, counts_coverage, args.mode, resource_snapshot, robustness_results)

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
            print(f"   chi2 = {geom.chi2_obs:.2f}, p = {geom.p_value:.4f}")
        elif geom.status == "NOT_TESTABLE":
            print(f"   Geometry NOT TESTABLE: {geom.reason}")
        else:
            print(f"   (geometry test not run or insufficient events)")
        print(f"K3 (No cross-channel coherence): {'KILLED' if k3_killed else 'NOT KILLED'}")
        print(f"   (requires explicit cross-modality analysis)")

    except MemoryLimitExceeded as e:
        print(f"\n=== ABORTED: MEMORY LIMIT EXCEEDED ===")
        print(str(e))
        report_path = generate_abort_report(output_dir, monitor, args.mode)
        print(f"\nPartial report: {report_path}")
        print(f"Output dir: {output_dir}")
        sys.exit(1)

    except Exception as e:
        print(f"\n=== ERROR ===")
        print(f"Unexpected error: {e}")
        monitor.aborted = True
        monitor.abort_reason = str(e)
        report_path = generate_abort_report(output_dir, monitor, args.mode)
        print(f"\nPartial report: {report_path}")
        raise


if __name__ == "__main__":
    main()
