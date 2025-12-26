"""Reporting utilities for the universality harness."""
from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .rg_bkt import RGResult
from .xy_mc import MCResult


def _plot_rg_flows(rg: RGResult, output_path: Path) -> None:
    plt.figure(figsize=(6, 4))
    for (k0, y0), flow in rg.flow_grid.items():
        plt.plot(flow[:, 1], flow[:, 2], alpha=0.6)
    plt.xlabel("K")
    plt.ylabel("y")
    plt.title("BKT RG flows")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_mc_curves(mc: MCResult, output_dir: Path) -> None:
    temps = np.array(mc.temperatures)
    plt.figure(figsize=(6, 4))
    for l_size, values in mc.helicity_modulus.items():
        plt.plot(temps, values, marker="o", label=f"L={l_size}")
    plt.xlabel("T")
    plt.ylabel("Helicity modulus Υ")
    plt.title("Helicity modulus vs T")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "helicity_modulus.png")
    plt.close()

    plt.figure(figsize=(6, 4))
    for l_size, values in mc.alpha_eff.items():
        plt.plot(temps, values, marker="o", label=f"L={l_size}")
    plt.xlabel("T")
    plt.ylabel("alpha_eff")
    plt.title("alpha_eff vs T")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "alpha_eff_vs_T.png")
    plt.close()


def _estimate_universal_alpha(mc: MCResult) -> Tuple[float, float, float]:
    temps = np.array(mc.temperatures)
    alpha_stack = np.vstack([mc.alpha_eff[l] for l in mc.lattice_sizes])
    mean_alpha = alpha_stack.mean(axis=0)
    std_alpha = alpha_stack.std(axis=0)
    relative_std = std_alpha / np.maximum(mean_alpha, 1e-12)
    best_idx = int(np.argmin(relative_std))
    return float(temps[best_idx]), float(mean_alpha[best_idx]), float(relative_std[best_idx])


def _verdict(rg: RGResult, mc_alpha: float, mc_rel_std: float) -> str:
    target = 1.0 / 137.0
    rg_close = 0.5 * target <= rg.alpha_eff_c <= 2.0 * target
    mc_close = 0.5 * target <= mc_alpha <= 2.0 * target
    stable = mc_rel_std <= 0.15
    if rg_close and mc_close and stable:
        return "DERIVATION CANDIDATE"
    return "NO-GO"


def write_report(rg: RGResult, mc: MCResult, output_dir: Path) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    _plot_rg_flows(rg, figures_dir / "rg_flows.png")
    _plot_mc_curves(mc, figures_dir)

    temp_star, alpha_star, rel_std = _estimate_universal_alpha(mc)
    verdict = _verdict(rg, alpha_star, rel_std)

    report = {
        "rg": {
            "k_c": rg.k_c,
            "alpha_eff_c": rg.alpha_eff_c,
            "sanity_pass": rg.sanity_pass,
            "message": rg.message,
        },
        "mc": {
            "temperatures": list(mc.temperatures),
            "lattice_sizes": list(mc.lattice_sizes),
            "alpha_eff": {str(k): v.tolist() for k, v in mc.alpha_eff.items()},
            "helicity_modulus": {
                str(k): v.tolist() for k, v in mc.helicity_modulus.items()
            },
            "best_temp": temp_star,
            "best_alpha": alpha_star,
            "relative_std": rel_std,
        },
        "mapping": {
            "definition": "K_R := Υ / T, g_eff^2 := 1/K_R, alpha_eff := g_eff^2/(4*pi)",
        },
        "verdict": verdict,
        "notes": [
            "This harness tests universality in BKT/XY models only; it is not a fit to electromagnetism.",
            "Even if universality exists, it may not yield 1/137; that outcome is the point of this test.",
        ],
    }

    report_md = [
        "# Universality Harness Report",
        "",
        "## What was tested",
        "Analytic BKT RG flows and Monte Carlo XY simulations with helicity modulus-based coupling mapping.",
        "",
        "## RG result",
        f"* K_c: {rg.k_c:.4f}",
        f"* alpha_eff_c: {rg.alpha_eff_c:.6f}",
        f"* Sanity (K_c ~ 2/pi): {'PASS' if rg.sanity_pass else 'FAIL'}",
        "",
        "## MC result",
        f"* Best clustering temperature: {temp_star:.3f}",
        f"* alpha_eff at best point: {alpha_star:.6f}",
        f"* Relative L-spread: {rel_std:.3f}",
        "",
        "## Verdict",
        verdict,
        "",
        "## Notes",
        "* This harness tests universality in BKT/XY models only; it is not a fit to electromagnetism.",
        "* Even if universality exists, it may not yield 1/137; that outcome is the point of this test.",
    ]

    (output_dir / "REPORT.md").write_text("\n".join(report_md), encoding="utf-8")
    (output_dir / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report
