from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from .config import MaxIDConfig


def _format_table(rows: List[Dict[str, object]], headers: List[str]) -> str:
    lines = ["|" + "|".join(headers) + "|", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        line = "|" + "|".join(str(row.get(h, "")) for h in headers) + "|"
        lines.append(line)
    return "\n".join(lines)


def write_report(cfg: MaxIDConfig, result: Dict[str, object], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "REPORT.md"
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    lines = [
        "# TD-only maximally identifying inference report",
        "",
        "## How to run",
        "```\npython3 -m msoup_td_maxid.run --config configs/td_maxid_default.yaml --mode base\n```",
        "",
    ]

    if result.get("status") == "ABORTED":
        lines.append(f"**ABORTED**: {result.get('reason', 'Unknown reason')}")
        lines.append("")
        lines.append(f"Peak RSS: {result.get('resource', {}).get('peak_rss_gb', 'n/a')} GB")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    summary = result.get("summary", {})
    f0_summary = result.get("f0_summary", {})
    lines.extend(
        [
            "## Posterior summary",
            f"- δm median: {summary.get('median', 'n/a'):.4f} (+{summary.get('high68', 0)-summary.get('median', 0):.4f}/"
            f"-{summary.get('median', 0)-summary.get('low68', 0):.4f})",
            f"- f0 median: {f0_summary.get('median', 'n/a'):.4f}",
            "",
            "## Dominance kill table",
        ]
    )

    kill = result.get("kill_table", {})
    lines.append(_format_table([kill], ["K-DOM", "K-PSIS", "K-LOO", "K-PPC"]))
    lines.append("")

    lines.append(f"**Verdict:** {result.get('verdict', 'n/a')}")
    lines.append("")

    info_rows = result.get("info_fraction", [])
    if info_rows:
        lines.append("## Information fractions")
        lines.append(_format_table(info_rows, ["lens_id", "fraction"]))
        lines.append("")

    loo_rows = result.get("loo", [])
    if loo_rows:
        lines.append("## Leave-one-out diagnostics")
        lines.append(_format_table(loo_rows, ["lens_id", "pareto_k", "delta_m_shift", "median_loo", "median_full"]))
        lines.append("")

    ppc_rows = result.get("ppc", [])
    if ppc_rows:
        lines.append("## Per-lens pulls at MAP")
        lines.append(_format_table(ppc_rows, ["lens_id", "residual", "sigma", "pull", "support"]))
        lines.append("")

    resource = result.get("resource", {})
    lines.append("## Resource usage")
    lines.append(f"- Peak RSS: {resource.get('peak_rss_gb', 'n/a')} GB (limit {cfg.compute.max_rss_gb} GB)")
    lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
