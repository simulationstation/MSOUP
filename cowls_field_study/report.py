"""Markdown reporting for the field study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class LensResult:
    lens_id: str
    band: str
    score_bin: str
    mode: str
    t_corr: float
    t_pow: float
    z_corr: float
    z_pow: float
    t_corr_hp: float | None = None
    t_pow_hp: float | None = None
    z_corr_hp: float | None = None
    z_pow_hp: float | None = None


@dataclass
class ReportBundle:
    subset_label: str
    n_processed: int
    n_used: int
    model_count: int
    approx_count: int
    z_corr_mean: float
    z_pow_mean: float
    z_corr_mean_approx: float
    z_pow_mean_approx: float
    z_corr_global: float
    z_pow_global: float
    z_corr_hp_mean: float
    z_pow_hp_mean: float
    z_corr_hp_mean_approx: float
    z_pow_hp_mean_approx: float
    z_corr_hp_global: float
    z_pow_hp_global: float
    lens_results: List[LensResult] = field(default_factory=list)


def _format_ci(samples: np.ndarray, alpha: float = 0.16) -> str:
    low = float(np.nanpercentile(samples, 100 * alpha))
    high = float(np.nanpercentile(samples, 100 * (1 - alpha)))
    mean = float(np.nanmean(samples))
    return f"{mean:.3f} [{low:.3f}, {high:.3f}]"


def write_report(out_dir: Path, bundle: ReportBundle) -> Path:
    """
    Write a concise markdown report summarizing the field analysis.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "report.md"
    lines = [
        f"# COWLS field-level study: {bundle.subset_label}",
        "",
        f"- processed lenses: {bundle.n_processed}",
        f"- used in stats: {bundle.n_used}",
        f"- model residual lenses: {bundle.model_count}",
        f"- approx residual lenses: {bundle.approx_count}",
        "",
        f"Mean Z_corr (model): {bundle.z_corr_mean:.3f}",
        f"Mean Z_pow  (model): {bundle.z_pow_mean:.3f}",
        f"Mean Z_corr (approx): {bundle.z_corr_mean_approx:.3f}",
        f"Mean Z_pow  (approx): {bundle.z_pow_mean_approx:.3f}",
        f"Global Z_corr: {bundle.z_corr_global:.3f}",
        f"Global Z_pow: {bundle.z_pow_global:.3f}",
        f"Mean Z_corr_hp (model): {bundle.z_corr_hp_mean:.3f}",
        f"Mean Z_pow_hp  (model): {bundle.z_pow_hp_mean:.3f}",
        f"Mean Z_corr_hp (approx): {bundle.z_corr_hp_mean_approx:.3f}",
        f"Mean Z_pow_hp  (approx): {bundle.z_pow_hp_mean_approx:.3f}",
        f"Global Z_corr_hp: {bundle.z_corr_hp_global:.3f}",
        f"Global Z_pow_hp: {bundle.z_pow_hp_global:.3f}",
        "",
        "## Per-lens summary",
    ]

    for res in bundle.lens_results:
        lines.append(
            f"- {res.lens_id} ({res.band}, {res.score_bin}, {res.mode}): "
            f"T_corr={res.t_corr:.3f}, T_pow={res.t_pow:.3f}, "
            f"Z_corr={res.z_corr:.2f}, Z_pow={res.z_pow:.2f}, "
            f"T_corr_hp={res.t_corr_hp if res.t_corr_hp is not None else float('nan'):.3f}, "
            f"T_pow_hp={res.t_pow_hp if res.t_pow_hp is not None else float('nan'):.3f}, "
            f"Z_corr_hp={res.z_corr_hp if res.z_corr_hp is not None else float('nan'):.2f}, "
            f"Z_pow_hp={res.z_pow_hp if res.z_pow_hp is not None else float('nan'):.2f}"
        )

    report_path.write_text("\n".join(lines))
    return report_path
