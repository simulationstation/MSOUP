from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from typing import Dict, List, Optional

from .config import MsoupConfig


def _probe_summary(probe_results: Dict[str, dict]) -> str:
    lines = []
    for name, res in probe_results.items():
        if name == "fit":
            continue
        lines.append(f"### Probe: {name}")
        lines.append(f"- N = {res['N']}")
        lines.append(f"- z range = [{res['z_min']:.3f}, {res['z_max']:.3f}], z_eff={res['z_eff']:.3f}")
        zb = res["z_below"]
        lines.append(f"- kernel mass: <=0.1 {zb[0.1]:.3f}, <=0.2 {zb[0.2]:.3f}, <=0.5 {zb[0.5]:.3f}")
        lines.append(f"- f_eff = {res['f_eff']:.4f}, H_inf = {res['H_inf']:.3f}")
        lines.append("")
    return "\n".join(lines)


def _distance_summary(distance_results: Optional[Dict]) -> str:
    """Generate markdown summary of distance-mode residual results."""
    if not distance_results:
        return "_No distance-mode results available._\n"

    lines = []
    probes = distance_results.get("probes", {})
    global_stats = distance_results.get("global", {})

    for probe_name, res in probes.items():
        status = res.get("status", "ERROR")
        lines.append(f"### Probe: {probe_name}")
        lines.append(f"- Status: {status}")

        if status == "OK":
            n = res.get("n", 0)
            chi2 = res.get("chi2", float("nan"))
            dof = res.get("dof", 0)
            chi2_dof = res.get("chi2_dof", float("nan"))
            method = res.get("method", "unknown")
            marginalized = res.get("marginalized_offset", False)
            b_hat = res.get("b_hat", 0.0)
            z_min = res.get("z_min", float("nan"))
            z_max = res.get("z_max", float("nan"))

            lines.append(f"- N = {n}")
            lines.append(f"- z range = [{z_min:.3f}, {z_max:.3f}]")
            lines.append(f"- chi2 = {chi2:.2f}, dof = {dof}, chi2/dof = {chi2_dof:.3f}")
            lines.append(f"- Method: {method}")
            if marginalized:
                lines.append(f"- SN offset marginalized: b_hat = {b_hat:.4f}")
        elif status == "NOT_TESTABLE":
            reason = res.get("reason", "unknown")
            lines.append(f"- Reason: {reason}")
        else:
            reason = res.get("reason", "unknown error")
            lines.append(f"- Error: {reason}")

        lines.append("")

    # Global summary
    total_chi2 = global_stats.get("chi2_total", 0.0)
    total_dof = global_stats.get("dof_total", 0)
    elapsed = global_stats.get("elapsed_s", 0.0)
    peak_rss = global_stats.get("peak_rss_mb", 0.0)

    lines.append("### Global Summary")
    lines.append(f"- Total chi2 = {total_chi2:.2f}")
    lines.append(f"- Total dof = {total_dof}")
    if total_dof > 0:
        lines.append(f"- Global chi2/dof = {total_chi2 / total_dof:.3f}")
    lines.append(f"- Elapsed: {elapsed:.1f}s")
    lines.append(f"- Peak RSS: {peak_rss:.0f} MB")
    lines.append("")

    return "\n".join(lines)


def write_report(
    cfg: MsoupConfig,
    probe_results: Dict[str, dict],
    output_dir: pathlib.Path,
    distance_results: Optional[List[dict]] = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "REPORT.md"
    delta_m_star = probe_results.get("fit", {}).get("delta_m_star", 0.0)

    lines = [
        "# Purist MSOUP Closure Report",
        "",
        "## Config Snapshot",
        "```yaml",
        json.dumps(asdict(cfg), indent=2, default=str),
        "```",
        "",
        f"## Fitted Delta_m*: {delta_m_star:.4f}",
        "",
        "## Probe Summaries (Kernel Mode)",
        _probe_summary(probe_results),
    ]

    # Add distance-mode section if results exist
    if distance_results:
        lines.extend([
            "## Distance-Mode Residuals",
            "",
            _distance_summary(distance_results),
        ])

    lines.extend([
        "## Diagnostics",
        "- K1: Delta_m* should be O(1); check bounds.",
        "- K2: ordering local > mid > high expected when kernels provided.",
        "- K3: sensitivity weights not implemented; skipped.",
        "",
        "## Plots",
        "- af_z.png",
        "- hinf_vs_delta_m.png",
        "- kernel_histograms.png",
        "- h_ratio.png",
        "- delta_mu.png",
        "- bao_dv.png",
    ])

    # Add distance-mode plots if available
    if distance_results:
        lines.append("")
        lines.append("### Distance-Mode Fit Plots")
        probes_data = distance_results.get("probes", {})
        for probe_name, res in probes_data.items():
            status = res.get("status", "ERROR")
            if status == "OK":
                lines.append(f"- {probe_name}_fit.png")

    report_path.write_text("\n".join(lines), encoding="utf-8")

    # Build summary data
    summary_data = {"config": asdict(cfg), "probes": probe_results}
    if distance_results:
        summary_data["distance_results"] = distance_results

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)
