"""
CLI entrypoint for the MV-O1B pocket/intermittency test.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from .candidates import candidates_to_frame, extract_candidates
from .config import load_config
from .geometry import GeometryResult, evaluate_geometry
from .io_igs import (
    group_by_sensor,
    iter_clk_series,
    load_clk_from_cache,
    load_positions_from_cache,
    resolve_paths,
)
from .io_mag import group_by_station, iter_magnetometer
from .preprocess import apply_quality, drop_bad_windows, standardize_magnetometer
from .report import generate_report
from .stats import GlobalStats, compute_debiased_stats
from .windows import derive_windows, windows_to_frame


def _process_clk_batch(batch: pd.DataFrame, cadence_seconds: float) -> Dict[str, pd.DataFrame]:
    df = batch.copy()
    df["value"] = df["bias_s"]
    df["sensor"] = df["satellite"]
    df["time"] = pd.to_datetime(df["time"])
    df = df[["time", "sensor", "value"]]
    return group_by_sensor(df, "sensor")


def run_pipeline(config_path: str | None) -> Path:
    """Run the full pipeline from CLI."""
    cfg = load_config(config_path).adjusted_for_fast_mode()
    cache_dir = Path(cfg.output.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    clk_paths = resolve_paths(cfg.data.clk_glob)
    sp3_paths = resolve_paths(cfg.data.sp3_glob)

    clk_batches = load_clk_from_cache(cache_dir, clk_paths, cfg.data.chunk_hours)
    positions = load_positions_from_cache(cache_dir, sp3_paths)

    # Magnetometer streaming
    mag_paths = resolve_paths(cfg.data.magnetometer_glob)

    per_sensor_series: Dict[str, pd.DataFrame] = {}
    for batch in tqdm(clk_batches, desc="CLK batches"):
        per_sensor_series.update(_process_clk_batch(batch, cfg.preprocess.cadence_seconds))

    for batch in tqdm(iter_magnetometer(mag_paths, cfg.data.chunk_hours), desc="MAG batches"):
        mag = standardize_magnetometer(batch, cfg.preprocess)
        mag = drop_bad_windows(mag)
        per_station = group_by_station(mag)
        for station, df in per_station.items():
            df = df.rename(columns={"btot": "value", "station": "sensor"})
            per_sensor_series[station] = df[["time", "sensor", "value"]]

    windows = derive_windows(per_sensor_series, cfg.preprocess, cfg.windows)
    window_frame = windows_to_frame(windows)

    candidates = extract_candidates(per_sensor_series, windows, cfg.candidates)
    candidate_frame = candidates_to_frame(candidates)

    stats: GlobalStats = compute_debiased_stats(candidate_frame, window_frame, cfg.stats, cfg.candidates)
    geom: GeometryResult = evaluate_geometry(candidate_frame, positions, cfg.geometry) if cfg.geometry.enable_geometry_test else GeometryResult(float("nan"), None, 1.0, 0)

    report_path = generate_report(
        output_dir=Path(cfg.output.results_dir),
        stats=stats,
        geom=geom,
        cross_modality_ok=not mag_paths or not candidate_frame.empty,
        run_label=cfg.run_label,
        config_path=Path(config_path) if config_path else None,
    )
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MV-O1B pocket/intermittency test.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--fast", action="store_true", help="Enable fast sanity run.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.fast_mode = args.fast or cfg.fast_mode
    cfg = cfg.adjusted_for_fast_mode()
    tmp_path = Path(".tmp_config.yaml")
    cfg.save(tmp_path)

    try:
        report = run_pipeline(str(tmp_path))
    finally:
        tmp_path.unlink(missing_ok=True)

    print(f"Report written to {report}")


if __name__ == "__main__":
    main()
