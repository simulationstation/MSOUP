"""Report generation for neutrality experiments."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .experiments import ExperimentOutputs
from .plotting import plot_coarsegrain, plot_mu_over_t


def classify_result(combined: dict) -> str:
    mu_rate = combined["mu_hit_rate"]
    cg1 = combined["cg1_rate"]
    cg2 = combined["cg2_rate"]
    if mu_rate >= 0.6 and cg1 >= 0.6 and cg2 >= 0.6:
        return "RESULT: DERIVATION FOUND"
    if mu_rate < 0.2 or min(cg1, cg2) < 0.3:
        return "RESULT: NO-GO"
    return "RESULT: INCONCLUSIVE"


def write_report(outputs: ExperimentOutputs, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    mu_plot = plot_mu_over_t(outputs.dual_sweep, figures_dir)
    cg_plot = plot_coarsegrain(outputs.coarsegrain, figures_dir)

    combined = dict(outputs.combined)
    combined["decision"] = classify_result(combined)

    report = {
        "config": outputs.config,
        "entropy_checks": outputs.entropy_checks,
        "dual_sweep": outputs.dual_sweep,
        "coarsegrain": outputs.coarsegrain,
        "combined": combined,
        "figures": {
            "mu_over_t": mu_plot,
            "delta_m_coarsegrain": cg_plot,
        },
    }

    json_path = output_dir / "REPORT.json"
    json_path.write_text(json.dumps(report, indent=2))

    md_path = output_dir / "REPORT.md"
    md_path.write_text(_render_markdown(report))

    return report


def _render_markdown(report: dict) -> str:
    combined = report["combined"]
    lines = []
    lines.append("# MSoup/Mverse Neutrality Report")
    lines.append("")
    lines.append("## Definitions")
    lines.append("- Δm: rank(A_+) − rank(A_-) from GF(2) constraint matrices.")
    lines.append("- ΔS = −Δm ln 2 from exact solution counting.")
    lines.append("- μ/T: inferred from Δ(F/T) per added independent constraint.")
    lines.append("- Coarse-graining: projection of constraints onto blocks or random projections.")
    lines.append("")

    lines.append("## Entropy sanity checks")
    for entry in report["entropy_checks"]:
        lines.append(
            f"- n={entry['n']} Δm={entry['delta_m']} ΔS={entry['delta_s']:.4f} (expected {-entry['delta_m'] * np.log(2):.4f})"
        )
    lines.append("")

    lines.append("## Dual-variable sweep")
    lines.append("See figure: `figures/mu_over_t.png`.")
    lines.append("")

    lines.append("## Coarse-graining")
    lines.append("See figure: `figures/delta_m_coarsegrain.png`.")
    lines.append("")

    lines.append("## Sensitivity summary")
    lines.append(
        f"- μ/T within {combined['tolerance']:.2f} of ln2 for {combined['mu_hit_rate']:.2%} of sweep points."
    )
    lines.append(
        f"- CG1 Δm(ℓ) stays O(1) for {combined['cg1_rate']:.2%} of instances."
    )
    lines.append(
        f"- CG2 Δm(ℓ) stays O(1) for {combined['cg2_rate']:.2%} of instances."
    )
    lines.append("")

    lines.append(combined["decision"])
    lines.append("")

    return "\n".join(lines)
