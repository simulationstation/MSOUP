"""
Report generation for the MV-O1B pocket test.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .geometry import GeometryResult
from .stats import GlobalStats


def format_kill_conditions(stats: GlobalStats, geom: GeometryResult, cross_modality_ok: bool) -> Dict[str, str]:
    """Produce kill-condition status strings."""
    k1 = "FAIL" if abs(stats.c_excess) < (stats.null_std * 1.0 + 1e-6) else "PASS"
    k2 = "FAIL" if geom.p_value > 0.1 else "PASS"
    k3 = "FAIL" if not cross_modality_ok else "PASS"
    return {"K1": k1, "K2": k2, "K3": k3}


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
        "# MV-O1B Pocket/Intermittency Test",
        "",
        f"Generated: {datetime.utcnow().isoformat()}Z",
        f"Run label: {run_label}",
        "",
        "## Summary Statistics",
        "",
        f"- Fano factor: {stats.fano_factor:.3f}",
        f"- Observed clustering C_obs: {stats.c_obs:.4f}",
        f"- Debiased clustering C_excess: {stats.c_excess:.4f}",
        f"- Null mean: {stats.null_mean:.4f} (std {stats.null_std:.4f})",
        f"- Empirical p-value: {stats.p_value:.4f}",
        "",
        "## Geometry Consistency",
        f"- Sensors used: {geom.n_sensors}",
        f"- Speed (km/s): {geom.speed_kmps:.2f}",
        f"- Shuffle p-value: {geom.p_value:.4f}",
        "",
        "## Kill Conditions",
        "",
        "| Code | Condition | Status | Meaning |",
        "|------|-----------|--------|---------|",
        f"| K1 | C_excess consistent with 0 under robustness sweeps | {kill['K1']} | Reject if pockets not distinguishable |",
        f"| K2 | Propagation geometry inconsistent with shuffled | {kill['K2']} | Reject if no coherent traversal |",
        f"| K3 | GNSS–magnetometer coherence present | {kill['K3']} | Reject if no cross-modality support |",
        "",
    ]

    lines.extend(
        [
            "## Sensor Weights",
            "",
            "| Sensor | Weight |",
            "|--------|--------|",
        ]
    )
    for sensor, weight in stats.sensor_weights.items():
        lines.append(f"| {sensor} | {weight:.4f} |")
    lines.append("")

    if config_path:
        lines.extend(
            [
                "## Configuration",
                "",
                f"Config path: {config_path}",
                "",
            ]
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
