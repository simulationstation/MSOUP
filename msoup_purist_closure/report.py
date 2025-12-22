from __future__ import annotations

import json
import pathlib
from dataclasses import asdict
from typing import Dict

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


def write_report(cfg: MsoupConfig, probe_results: Dict[str, dict], output_dir: pathlib.Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "REPORT.md"
    delta_m_star = probe_results["fit"]["delta_m_star"]

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
        "## Probe Summaries",
        _probe_summary(probe_results),
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
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump({"config": asdict(cfg), "probes": probe_results}, f, indent=2, default=str)
