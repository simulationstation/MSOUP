"""
Report generation for the MV-O1B pocket test.
"""

from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .geometry import GeometryResult
from .stats import GlobalStats
from .windows import Window


def format_kill_conditions(stats: GlobalStats, geom: GeometryResult, cross_modality_ok: bool) -> Dict[str, Dict[str, Any]]:
    """Produce kill-condition status with details."""
    k1_killed = stats.p_clustering > 0.05 or stats.c_excess <= 0
    k2_killed = geom.status == "COMPLETED" and geom.p_value is not None and geom.p_value > 0.05
    k3_killed = not cross_modality_ok

    return {
        "K1": {
            "status": "KILLED" if k1_killed else "NOT KILLED",
            "killed": k1_killed,
            "description": "C_excess consistent with 0 under robustness sweeps",
        },
        "K2": {
            "status": "KILLED" if k2_killed else ("NOT_TESTABLE" if geom.status == "NOT_TESTABLE" else "NOT KILLED"),
            "killed": k2_killed,
            "description": "Propagation geometry indistinguishable from shuffled nulls",
            "reason": geom.reason,
        },
        "K3": {
            "status": "KILLED" if k3_killed else "NOT KILLED",
            "killed": k3_killed,
            "description": "No cross-channel (GNSS-magnetometer) coherence",
        },
    }


def _quantiles(series: Iterable[float], keys: Tuple[float, ...]) -> Dict[str, Optional[float]]:
    arr = pd.Series(list(series), dtype=float)
    if arr.empty:
        return {f"p{int(k*100)}": None for k in keys}
    return {f"p{int(k*100)}": float(arr.quantile(k)) for k in keys}


def _basic_stats(series: Iterable[float]) -> Dict[str, Optional[float]]:
    arr = pd.Series(list(series), dtype=float)
    if arr.empty:
        return {"min": None, "median": None, "max": None}
    return {
        "min": float(arr.min()),
        "median": float(arr.median()),
        "max": float(arr.max()),
    }


def compute_counts_coverage(
    per_sensor_series: Dict[str, pd.DataFrame],
    windows: List[Window],
    candidate_frame: pd.DataFrame,
    neighbor_time: float,
    neighbor_angle: Optional[float],
    thresholds: List[float],
    geom: GeometryResult,
) -> Dict[str, Any]:
    """Aggregate counts/coverage metrics for auditing."""
    n_sensors_total = len(per_sensor_series)
    window_frame = pd.DataFrame([w.__dict__ for w in windows]) if windows else pd.DataFrame(columns=["sensor", "start", "end", "quality_score", "cadence_seconds"])
    n_windows_total = len(window_frame)
    window_durations = ((pd.to_datetime(window_frame["end"]) - pd.to_datetime(window_frame["start"])).dt.total_seconds()) if not window_frame.empty else pd.Series([], dtype=float)
    coverage_fraction = window_frame.groupby("sensor")["quality_score"].mean() if "quality_score" in window_frame else pd.Series([], dtype=float)
    masked_fraction = 1 - coverage_fraction

    # Candidate statistics
    n_candidates_total = len(candidate_frame)
    candidates_per_sensor = candidate_frame.groupby("sensor").size() if not candidate_frame.empty else pd.Series([], dtype=int)
    candidates_per_window = candidate_frame[candidate_frame["window_index"] >= 0].groupby("window_index").size() if not candidate_frame.empty else pd.Series([], dtype=int)
    snr_series = candidate_frame["sigma"] if "sigma" in candidate_frame else pd.Series([], dtype=float)
    threshold_counts = {f">={thr}": int((candidate_frame["sigma"] >= thr).sum()) if not candidate_frame.empty else 0 for thr in thresholds}

    # Pair statistics
    n_events = len(candidate_frame)
    n_pairs_total = int(n_events * (n_events - 1) // 2)
    sensor_counts = candidate_frame.groupby("sensor").size() if not candidate_frame.empty else pd.Series([], dtype=int)
    pairs_per_sensor = []
    total_events = sensor_counts.sum()
    for _, count in sensor_counts.items():
        pairs = count * (total_events - count) + count * (count - 1) / 2
        pairs_per_sensor.append(pairs)

    return {
        "input_coverage": {
            "n_sensors_total": n_sensors_total,
            "n_sensors_used": int(window_frame["sensor"].nunique()) if not window_frame.empty else 0,
            "n_windows_total": n_windows_total,
            "window_duration_s": {**_basic_stats(window_durations), **_quantiles(window_durations, (0.1, 0.5, 0.9, 0.99))},
            "per_sensor_masked_fraction": {**_basic_stats(masked_fraction), **_quantiles(masked_fraction, (0.1, 0.5, 0.9, 0.99))},
        },
        "candidate_catalog": {
            "n_candidates_total": n_candidates_total,
            "per_sensor": {**_basic_stats(candidates_per_sensor), **_quantiles(candidates_per_sensor, (0.1, 0.5, 0.9, 0.99))},
            "per_window": {**_basic_stats(candidates_per_window), **_quantiles(candidates_per_window, (0.1, 0.5, 0.9, 0.99))},
            "sigma_distribution": {**_basic_stats(snr_series), **_quantiles(snr_series, (0.5, 0.9, 0.99))},
            "threshold_counts": threshold_counts,
        },
        "pair_statistics": {
            "n_pairs_total": n_pairs_total,
            "n_sensors_with_pairs": int(sensor_counts.shape[0]),
            "pairs_per_sensor": {**_basic_stats(pairs_per_sensor), **_quantiles(pairs_per_sensor, (0.1, 0.5, 0.9, 0.99))},
            "neighbor_time_s": neighbor_time,
            "neighbor_angle_deg": neighbor_angle,
        },
        "geometry_preconditions": {
            "n_candidate_events_total": n_events,
            "n_coincidences_used": int(sensor_counts.count()),
            "n_events_tested_geometry": geom.n_events,
            "reason": geom.reason,
        },
    }


def generate_report(
    output_dir: Path,
    stats: GlobalStats,
    geom: GeometryResult,
    cross_modality_ok: bool,
    run_label: str,
    config_path: Optional[Path],
    counts_coverage: Optional[Dict[str, Any]] = None,
    resources: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write markdown report with kill conditions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "REPORT.md"
    kill = format_kill_conditions(stats, geom, cross_modality_ok)

    lines = [
        "# MV-O1B Pocket/Intermittency Test Report",
        "",
        f"**Generated:** {datetime.utcnow().isoformat()}Z",
        f"**Run label:** {run_label}",
        "",
        "---",
        "",
        "## Kill Condition Summary",
        "",
        "| Code | Condition | Status |",
        "|------|-----------|--------|",
        f"| K1 | {kill['K1']['description']} | **{kill['K1']['status']}** |",
        f"| K2 | {kill['K2']['description']} | **{kill['K2']['status']}** |",
        f"| K3 | {kill['K3']['description']} | **{kill['K3']['status']}** |",
        "",
        "---",
        "",
        "## Counts & Coverage",
        "",
    ]

    if counts_coverage:
        icov = counts_coverage["input_coverage"]
        cat = counts_coverage["candidate_catalog"]
        pairs = counts_coverage["pair_statistics"]
        geom_pre = counts_coverage["geometry_preconditions"]
        lines.extend([
            f"- **Sensors total/used:** {icov['n_sensors_total']} / {icov['n_sensors_used']}",
            f"- **Windows:** {icov['n_windows_total']}",
            f"- **Window duration (s):** min={icov['window_duration_s']['min']}, median={icov['window_duration_s']['median']}, max={icov['window_duration_s']['max']}",
            f"- **Masked fraction per sensor (p50/p90/p99):** {icov['per_sensor_masked_fraction']['p50']}, {icov['per_sensor_masked_fraction']['p90']}, {icov['per_sensor_masked_fraction']['p99']}",
            f"- **Candidates total:** {cat['n_candidates_total']}",
            f"- **Candidates per sensor (p50/p90/p99):** {cat['per_sensor']['p50']}, {cat['per_sensor']['p90']}, {cat['per_sensor']['p99']}",
            f"- **Candidates per window (p50/p90/p99):** {cat['per_window']['p50']}, {cat['per_window']['p90']}, {cat['per_window']['p99']}",
            f"- **Sigma distribution (p50/p90/p99):** {cat['sigma_distribution']['p50']}, {cat['sigma_distribution']['p90']}, {cat['sigma_distribution']['p99']}",
            f"- **Threshold counts:** {cat['threshold_counts']}",
            f"- **Total pairs evaluated:** {pairs['n_pairs_total']}",
            f"- **Pairs per sensor (p50/p90/p99):** {pairs['pairs_per_sensor']['p50']}, {pairs['pairs_per_sensor']['p90']}, {pairs['pairs_per_sensor']['p99']}",
            f"- **Clustering neighborhood:** Δt={pairs['neighbor_time_s']} s, Δθ={pairs['neighbor_angle_deg']}",
            f"- **Geometry preconditions:** n_candidate_events={geom_pre['n_candidate_events_total']}, n_events_tested={geom_pre['n_events_tested_geometry']}, reason={geom_pre.get('reason')}",
            "",
            "---",
            "",
        ])

    lines.extend([
        "## Fano Factor Statistics",
        "",
        f"- **Fano observed:** {stats.fano_obs:.4f}",
        f"- **Fano null mean:** {stats.fano_null_mean:.4f}",
        f"- **Fano null std:** {stats.fano_null_std:.4f}",
        f"- **Fano Z-score:** {stats.fano_z:.4f}",
        f"- **Fano p-value:** {stats.p_fano:.4f}",
        "",
        "## Clustering Statistics",
        "",
        f"- **C_obs:** {stats.c_obs:.6f}",
        f"- **C_null mean:** {stats.c_null_mean:.6f}",
        f"- **C_null std:** {stats.c_null_std:.6f}",
        f"- **C_excess:** {stats.c_excess:.6f} ± {stats.c_excess_std:.6f}",
        f"- **Clustering Z-score (signed):** {stats.c_z:.4f}",
        f"- **Clustering p-value (one-sided):** {stats.p_clustering:.4f}",
        f"- **Clustering p-value (two-sided):** {getattr(stats, 'p_clustering_two_sided', math.nan):.4f}",
        f"- **Number of resamples:** {stats.n_resamples}",
        f"- **P-value resolution:** {stats.pvalue_resolution:.4f}",
        "",
        "---",
        "",
        "## Resources",
        "",
        f"- **Peak RSS (GB):** {stats.peak_rss_gb:.3f}",
        f"- **Max RSS limit (GB):** {resources.get('max_rss_gb') if resources else 'n/a'}",
        f"- **Max workers:** {resources.get('max_workers') if resources else 'n/a'}",
        f"- **Chunk days:** {resources.get('chunk_days') if resources else 'n/a'}",
        f"- **Pair mode:** {resources.get('pair_mode') if resources else 'n/a'}",
        f"- **Time bin (s):** {resources.get('time_bin_seconds') if resources else 'n/a'}",
        "",
        "---",
        "",
        "## Geometry Consistency (K2)",
        "",
    ])

    if geom.status == "COMPLETED":
        lines.extend([
            f"- **χ² observed:** {geom.chi2_obs:.4f}",
            f"- **p-value:** {geom.p_value:.4f}",
            f"- **Number of events:** {geom.n_events}",
            f"- **Fitted speed:** {geom.speed_kmps:.2f} km/s",
        ])
    elif geom.status == "NOT_TESTABLE":
        lines.append(f"*Geometry not testable: {geom.reason}*")
    else:
        lines.append("*Geometry test not run.*")

    lines.extend([
        "",
        "---",
        "",
        "## K1 Details: Pockets Beyond Selection",
        "",
        "The K1 kill condition tests whether observed clustering (C_excess) is",
        "statistically distinguishable from null expectations. The hypothesis is",
        "killed if:",
        "",
        "1. C_excess ≤ 0 (no excess clustering observed), OR",
        "2. p_clustering > 0.05 (excess not statistically significant)",
        "",
        f"**Result:** C_excess = {stats.c_excess:.6f}, p = {stats.p_clustering:.4f}",
        f"**Status:** {kill['K1']['status']}",
        "",
        "---",
        "",
        "## K2 Details: Propagation Geometry",
        "",
        "The K2 kill condition tests whether candidate arrival times show coherent",
        "plane-wave propagation geometry. The hypothesis is killed if the fitted",
        "geometry chi-squared is indistinguishable from shuffled nulls (p > 0.05).",
        "",
    ])

    if geom.status == "COMPLETED":
        lines.extend([
            f"**Result:** χ² = {geom.chi2_obs:.4f}, p = {geom.p_value:.4f}",
            f"**Status:** {kill['K2']['status']}",
        ])
    elif geom.status == "NOT_TESTABLE":
        lines.extend([
            f"**Result:** NOT TESTABLE ({geom.reason})",
            f"**Status:** {kill['K2']['status']}",
        ])
    else:
        lines.extend([
            "**Result:** Geometry test not run",
            f"**Status:** {kill['K2']['status']}",
        ])

    lines.extend([
        "",
        "---",
        "",
        "## K3 Details: Cross-Channel Coherence",
        "",
        "The K3 kill condition requires cross-modality (GNSS-magnetometer) coherence",
        "for gravitational claims. This requires explicit cross-band analysis.",
        "",
        f"**Cross-modality data available:** {'Yes' if cross_modality_ok else 'No'}",
        f"**Status:** {kill['K3']['status']}",
        "",
        "---",
        "",
        "## Sensor Weights",
        "",
        "Sensors weighted inversely by candidate count to avoid dominance:",
        "",
        "| Sensor | Weight |",
        "|--------|--------|",
    ])

    sorted_weights = sorted(stats.sensor_weights.items(), key=lambda x: x[1], reverse=True)
    for sensor, weight in sorted_weights[:20]:
        lines.append(f"| {sensor} | {weight:.6f} |")

    if len(sorted_weights) > 20:
        lines.append(f"| ... | ({len(sorted_weights) - 20} more sensors) |")

    lines.extend([
        "",
        "---",
        "",
    ])

    if config_path:
        lines.extend([
            "## Configuration",
            "",
            f"Config path: `{config_path}`",
            "",
        ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate_robustness_report(
    output_dir: Path,
    robustness_results: List[Dict[str, Any]],
) -> Path:
    """Generate ROBUSTNESS.md summarizing sweep results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "ROBUSTNESS.md"

    lines = [
        "# MV-O1B Robustness Analysis",
        "",
        f"**Generated:** {datetime.utcnow().isoformat()}Z",
        "",
        "This report summarizes robustness sweeps across:",
        "- 3 detection thresholds (3σ, 4σ, 5σ)",
        "- 2 masking variants (full coverage 60%, relaxed 40%)",
        "- 2 null constructions (conditioned resample, block bootstrap)",
        "",
        f"**Total configurations tested:** {len(robustness_results)}",
        "",
        "---",
        "",
        "## Summary Table",
        "",
        "| Threshold | Masking | Null Type | N_cand | Fano_obs | C_excess | p_clust | K1 | K2 |",
        "|-----------|---------|-----------|--------|----------|----------|---------|----|----|",
    ]

    for r in robustness_results:
        if "error" in r:
            lines.append(f"| {r.get('threshold_sigma', '?')} | {r.get('masking', '?')} | {r.get('null_type', '?')} | ERROR | - | - | - | - | - |")
            continue

        k1 = "✗" if r.get("K1_killed", True) else "✓"
        k2 = "✗" if r.get("K2_killed", True) else "✓"
        fano = f"{r.get('fano_obs', np.nan):.2f}"
        c_excess = f"{r.get('c_excess', 0):.4f}"
        p_clust = f"{r.get('p_clustering', 1):.3f}"

        lines.append(
            f"| {r.get('threshold_sigma', '?')}σ | {r.get('masking', '?')} | {r.get('null_type', '?')} | "
            f"{r.get('n_candidates', 0)} | {fano} | {c_excess} | {p_clust} | {k1} | {k2} |"
        )

    lines.extend([
        "",
        "**Legend:** ✓ = NOT KILLED (signal persists), ✗ = KILLED (null hypothesis holds)",
        "",
        "---",
        "",
        "## Kill Condition Robustness",
        "",
    ])

    k1_killed_count = sum(1 for r in robustness_results if r.get("K1_killed", True) and "error" not in r)
    k1_total = sum(1 for r in robustness_results if "error" not in r)

    lines.extend([
        "### K1: Pockets Beyond Selection",
        "",
        f"- Killed in {k1_killed_count}/{k1_total} configurations ({100*k1_killed_count/max(k1_total,1):.1f}%)",
        "",
    ])

    if k1_killed_count == k1_total:
        lines.append("**Conclusion:** K1 robustly killed across all configurations.")
    elif k1_killed_count == 0:
        lines.append("**Conclusion:** K1 NOT killed in any configuration - signal persists.")
    else:
        lines.append("**Conclusion:** K1 result varies with configuration - marginal evidence.")

    k2_killed_count = sum(1 for r in robustness_results if r.get("K2_killed", True) and "error" not in r)

    lines.extend([
        "",
        "### K2: Propagation Geometry",
        "",
        f"- Killed in {k2_killed_count}/{k1_total} configurations ({100*k2_killed_count/max(k1_total,1):.1f}%)",
        "",
    ])

    if k2_killed_count == k1_total:
        lines.append("**Conclusion:** K2 robustly killed across all configurations.")
    elif k2_killed_count == 0:
        lines.append("**Conclusion:** K2 NOT killed in any configuration - geometry signal persists.")
    else:
        lines.append("**Conclusion:** K2 result varies with configuration - marginal evidence.")

    lines.extend([
        "",
        "---",
        "",
        "## Detailed Results by Threshold",
        "",
    ])

    for thresh in {r.get("threshold_sigma") for r in robustness_results if "threshold_sigma" in r and "error" not in r}:
        thresh_results = [r for r in robustness_results if r.get("threshold_sigma") == thresh and "error" not in r]
        if not thresh_results:
            continue

        lines.extend([
            f"### Threshold: {thresh}σ",
            "",
        ])

        for r in thresh_results:
            lines.extend([
                f"**{r.get('masking', '?')} / {r.get('null_type', '?')}:**",
                f"- Candidates: {r.get('n_candidates', 0)}, Windows: {r.get('n_windows', 0)}",
                f"- Fano: {r.get('fano_obs', np.nan):.3f} (null: {r.get('fano_null_mean', np.nan):.3f} ± {r.get('fano_null_std', np.nan):.3f})",
                f"- C_excess: {r.get('c_excess', 0):.6f} ± {r.get('c_excess_std', 0):.6f}",
                f"- p_clustering: {r.get('p_clustering', 1):.4f}",
                f"- K1: {'KILLED' if r.get('K1_killed', True) else 'NOT KILLED'}",
                f"- K2: {'KILLED' if r.get('K2_killed', True) else 'NOT KILLED'}",
                "",
            ])

    lines.extend([
        "---",
        "",
        "## Interpretation",
        "",
        "For a robust detection claim, kill conditions should be NOT KILLED across",
        "all reasonable analysis choices. Sensitivity to specific thresholds, masking,",
        "or null constructions indicates fragile evidence.",
        "",
    ])

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
