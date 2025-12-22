"""
Report generation for the MV-O1B pocket test.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .geometry import GeometryResult
from .stats import GlobalStats


def format_kill_conditions(stats: GlobalStats, geom: GeometryResult, cross_modality_ok: bool) -> Dict[str, Dict[str, Any]]:
    """Produce kill-condition status with details."""
    k1_killed = stats.p_clustering > 0.05 or stats.c_excess <= 0
    k2_killed = geom.p_value > 0.05 if not np.isnan(geom.p_value) else True
    k3_killed = not cross_modality_ok

    return {
        "K1": {
            "status": "KILLED" if k1_killed else "NOT KILLED",
            "killed": k1_killed,
            "description": "C_excess consistent with 0 under robustness sweeps",
        },
        "K2": {
            "status": "KILLED" if k2_killed else "NOT KILLED",
            "killed": k2_killed,
            "description": "Propagation geometry indistinguishable from shuffled nulls",
        },
        "K3": {
            "status": "KILLED" if k3_killed else "NOT KILLED",
            "killed": k3_killed,
            "description": "No cross-channel (GNSS-magnetometer) coherence",
        },
    }


def generate_report(
    output_dir: Path,
    stats: GlobalStats,
    geom: GeometryResult,
    cross_modality_ok: bool,
    run_label: str,
    config_path: Optional[Path],
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
        f"- **Clustering Z-score:** {stats.c_z:.4f}",
        f"- **Clustering p-value:** {stats.p_clustering:.4f}",
        f"- **Number of resamples:** {stats.n_resamples}",
        "",
        "---",
        "",
        "## Geometry Consistency (K2)",
        "",
    ]

    if not np.isnan(geom.chi2_obs):
        lines.extend([
            f"- **χ² observed:** {geom.chi2_obs:.4f}",
            f"- **p-value:** {geom.p_value:.4f}",
            f"- **Number of events:** {geom.n_events}",
            f"- **Fitted speed:** {geom.speed_kmps:.2f} km/s",
        ])
    else:
        lines.append("*Geometry test not run or insufficient events.*")

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

    if not np.isnan(geom.chi2_obs):
        lines.extend([
            f"**Result:** χ² = {geom.chi2_obs:.4f}, p = {geom.p_value:.4f}",
            f"**Status:** {kill['K2']['status']}",
        ])
    else:
        lines.extend([
            "**Result:** Insufficient events for geometry test",
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

    # Show top 20 sensors by weight
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

    # Analyze K1 robustness
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

    # Analyze K2 robustness
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

    # Group by threshold
    for thresh in [3.0, 4.0, 5.0]:
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
