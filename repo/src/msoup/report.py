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
            f"- n={entry['n']} Δm={entry['delta_m']} ΔS={entry['delta_s']:.4f} "
            f"(expected {-entry['delta_m'] * np.log(2):.4f})"
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
        f"- μ/T within {combined['tolerance']:.2f} of ln2 for "
        f"{combined['mu_hit_rate']:.2%} of sweep points."
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


def write_harness_report(results: dict, output_dir: Path, executed: bool = True) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    report = dict(results)
    report["executed"] = bool(executed)

    figure_paths = {}
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415

        mu_fig = output_dir / "mu_over_t.png"
        delta_fig = output_dir / "delta_m_coarsegrain.png"

        _plot_mu_over_t(results["q1"]["settings"], mu_fig)
        _plot_delta_m(results["q2"], delta_fig)
        figure_paths["mu_over_t"] = str(mu_fig)
        figure_paths["delta_m_coarsegrain"] = str(delta_fig)
        plt.close("all")
    except Exception:
        figure_paths["mu_over_t"] = None
        figure_paths["delta_m_coarsegrain"] = None

    report["figures"] = figure_paths

    json_path = output_dir / "REPORT.json"
    json_path.write_text(json.dumps(report, indent=2))

    md_path = output_dir / "REPORT.md"
    md_path.write_text(_render_harness_markdown(report))
    return report


def _plot_mu_over_t(settings: list[dict], path: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.figure(figsize=(7, 4))
    for setting in settings:
        label = f"n={setting['n']}, m/n={setting['density']:.2f}"
        j_over_t = np.array(setting["j_over_t"], dtype=float)
        mu_mean = np.array(setting["mu_over_t_mean"], dtype=float)
        plt.plot(j_over_t, mu_mean, marker="o", label=label)
    plt.xscale("log")
    plt.axhline(np.log(2.0), color="black", linestyle="--", linewidth=1, label="ln 2")
    plt.xlabel("J/T")
    plt.ylabel("μ/T")
    plt.title("μ/T vs J/T")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)


def _plot_delta_m(q2: dict, path: Path) -> None:
    import matplotlib.pyplot as plt  # noqa: PLC0415

    plt.figure(figsize=(7, 4))
    scales = sorted(int(s) for s in q2["block"]["median"].keys())
    block_median = [q2["block"]["median"][s] for s in scales]
    random_median = [q2["random"]["median"][s] for s in scales]
    plt.plot(scales, block_median, marker="o", label="block-sum")
    plt.plot(scales, random_median, marker="o", label="random projection")
    plt.xlabel("coarse-grain scale ℓ")
    plt.ylabel("median Δm(ℓ)")
    plt.title("Δm(ℓ) under coarse-graining")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path)


def _render_harness_markdown(report: dict) -> str:
    config = report["config"]
    q1 = report["q1"]
    q2 = report["q2"]
    lines = []
    lines.append("# Decision Harness Report")
    lines.append("")
    lines.append("## Definitions")
    lines.append("- XOR-SAT constraints over GF(2) with soft penalties E(x)=J * (#violated).")
    lines.append("- Partition function Z=Σ_x exp(−E/T_int).")
    lines.append("- Free energy F=−T_int ln Z.")
    lines.append("- μ := −[F(A_+) − F(A_-)]/Δm, so μ/T_int = (ln Z_+ − ln Z_-)/Δm.")
    lines.append("- Δm = rank(A_+) − rank(A_-).")
    lines.append("")
    if not report.get("executed", True):
        lines.append("**not executed here**")
        lines.append("")

    lines.append("## Configuration")
    lines.append(f"- n values: {config['n_values']}")
    lines.append(f"- m/n densities: {config['densities']}")
    lines.append(f"- Δm: {config['delta_m']}")
    lines.append(
        f"- J/T grid: logspace({np.log10(config['j_over_t_grid'][0]):.2f}, "
        f"{np.log10(config['j_over_t_grid'][-1]):.2f}, {len(config['j_over_t_grid'])})"
    )
    lines.append(f"- instances per setting: {config['instances_per_setting']}")
    lines.append("")

    lines.append("## Q1: μ/T vs J/T")
    for setting in q1["settings"]:
        lines.append("")
        lines.append(f"### n={setting['n']}, m/n={setting['density']:.2f}")
        lines.append(f"Success rate: {setting['success_rate']:.2%}")
        lines.append("")
        lines.append("| J/T | μ/T (mean) | μ/T (std) |")
        lines.append("| --- | --- | --- |")
        for j_over_t, mean, std in zip(
            setting["j_over_t"], setting["mu_over_t_mean"], setting["mu_over_t_std"]
        ):
            lines.append(f"| {j_over_t:.4g} | {mean:.4f} | {std:.4f} |")

    lines.append("")
    lines.append("## Q2: Δm(ℓ) coarse-graining")
    lines.append("")
    lines.append("| ℓ | median Δm(ℓ) block-sum | median Δm(ℓ) random |")
    lines.append("| --- | --- | --- |")
    for scale in sorted(int(s) for s in q2["block"]["median"].keys()):
        lines.append(
            f"| {scale} | {q2['block']['median'][scale]:.2f} | "
            f"{q2['random']['median'][scale]:.2f} |"
        )

    lines.append("")
    lines.append("## Verdict")
    lines.append(report["verdict"])
    lines.append("")
    if report.get("figures"):
        lines.append("## Plots")
        mu_path = report["figures"].get("mu_over_t")
        delta_path = report["figures"].get("delta_m_coarsegrain")
        lines.append(f"- mu_over_t: {mu_path}")
        lines.append(f"- delta_m_coarsegrain: {delta_path}")
    lines.append("")
    return "\n".join(lines)
