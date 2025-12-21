"""Reporting utilities for the ejection simulation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .dynamics import SimulationResult
from .inference import infer_expansion, morphology_stats
from .projection import robust_mean
from .calibrate import _build_config
from .dynamics import run_simulation


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, data: Dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _time_array(history: Dict[str, np.ndarray]):
    n = len(next(iter(history.values())))
    return np.linspace(0, 1, n)


def _plot_timeseries(history: Dict[str, np.ndarray], results_dir: Path):
    t = _time_array(history)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, history["H_global"], label="H_global")
    ax.set_xlabel("t")
    ax.set_ylabel("H surrogate")
    ax.legend()
    path = results_dir / "figures" / "H_global.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, history["X_V"], label="X_V")
    ax.plot(t, history["X_rho"], label="X_rho")
    ax.set_xlabel("t")
    ax.set_ylabel("Thinness drivers")
    ax.legend()
    path = results_dir / "figures" / "X_metrics.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, history["mean_dm3"], label="<rho_dm3>")
    ax.plot(t, history["V_EM"], label="V_EM")
    ax.set_xlabel("t")
    ax.legend()
    path = results_dir / "figures" / "dm3_and_vem.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_maps(sim: SimulationResult, cfg: SimulationConfig, results_dir: Path):
    times = ["t0", "tmid", "tfinal"]
    fields = ["rho_tot", "rho_dm3", "A", "wlt2"]
    fig, axes = plt.subplots(len(times), len(fields), figsize=(12, 9))
    for i, tkey in enumerate(times):
        snap = sim.snapshots.get(tkey, None)
        if snap is None:
            continue
        rho_tot = snap["rho_tot"]
        for j, fkey in enumerate(fields):
            data = snap[fkey] if fkey != "rho_tot" else rho_tot
            ax = axes[i, j]
            im = ax.imshow(data, cmap="viridis")
            ax.set_xticks([])
            ax.set_yticks([])
            if i == 0:
                ax.set_title(fkey)
            if j == len(fields) - 1:
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if j == 0:
                ax.set_ylabel(tkey)
    fig.tight_layout()
    path = results_dir / "figures" / "maps.png"
    fig.savefig(path)
    plt.close(fig)

    final = sim.final_fields
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].hist(final["A"].flatten(), bins=40, color="C0")
    axes[0].set_title("A final")
    rho_tot_final = final["rho_b"] + cfg.g_dm * final["rho_dm3"]
    axes[1].hist(np.log(rho_tot_final.flatten() + 1e-8), bins=40, color="C1")
    axes[1].set_title("log rho_tot")
    axes[2].hist(final["rho_dm3"].flatten(), bins=40, color="C2")
    axes[2].set_title("rho_dm3 final")
    fig.tight_layout()
    path = results_dir / "figures" / "histograms.png"
    fig.savefig(path)
    plt.close(fig)


def _plot_sensitivity(cfg: SimulationConfig, base_params: Dict[str, float], results_dir: Path):
    params_to_probe = ["beta", "beta_loc", "gamma0", "tau_dm3_0"]
    deltas = []
    for key in params_to_probe:
        base_value = base_params.get(key, getattr(cfg, key))
        trial = {k: base_params.get(k, getattr(cfg, k)) for k in params_to_probe}
        trial[key] = base_value * 1.1
        coarse_cfg = _build_config(cfg, trial, max(64, cfg.grid // 4), max(80, cfg.steps // 5), seed=cfg.seed + 99)
        sim = run_simulation(coarse_cfg)
        diag = infer_expansion(coarse_cfg, sim.history, sim.final_fields)
        deltas.append(diag["Delta_H0"])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(params_to_probe, deltas, color="C3")
    ax.set_ylabel("Delta_H0 (probe)")
    fig.tight_layout()
    path = results_dir / "figures" / "sensitivity.png"
    fig.savefig(path)
    plt.close(fig)


def generate_report(results_dir: str, cfg: SimulationConfig, sim: SimulationResult, params: Dict[str, float], calibration_used: bool = False):
    results_path = Path(results_dir)
    _ensure_dir(results_path)
    _ensure_dir(results_path / "figures")

    diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
    morph = morphology_stats(sim.final_fields, cfg)
    report_data = {
        "params": params,
        "diagnostics": diagnostics,
        "morphology": morph,
    }
    _save_json(results_path / "run_summary.json", report_data)

    _plot_timeseries(sim.history, results_path)
    _plot_maps(sim, cfg, results_path)
    _plot_sensitivity(cfg, params, results_path)

    report_lines = []
    report_lines.append("# Msoup Ejection + Decompression Simulation")
    report_lines.append("")
    report_lines.append("## Parameter set")
    for k, v in sorted(params.items()):
        report_lines.append(f"- **{k}**: {v:.4f}")
    report_lines.append("")
    report_lines.append("## Inference readouts")
    report_lines.append(f"- H0_early_inferred: {diagnostics['H0_early']:.3f}")
    report_lines.append(f"- H0_late_inferred: {diagnostics['H0_late']:.3f}")
    report_lines.append(f"- H0_late_global: {diagnostics['H0_late_global']:.3f}")
    report_lines.append(f"- Delta H0: {diagnostics['Delta_H0']:.3f}")
    report_lines.append("")
    report_lines.append("## Morphology")
    for k, v in morph.items():
        report_lines.append(f"- {k}: {v:.4f}")
    report_lines.append("")
    report_lines.append("## Diagnostics")
    report_lines.append("- X_V(t): early minimum {:.4f}, final {:.4f}".format(
        float(np.min(sim.history["X_V"])), float(sim.history["X_V"][-1])
    ))
    report_lines.append("- X_rho(final): {:.4f}".format(float(sim.history["X_rho"][-1])))
    report_lines.append("- mean(rho_dm3)(final): {:.4e}".format(float(sim.history["mean_dm3"][-1])))
    report_lines.append("- V_EM(final): {:.4e}".format(float(sim.history["V_EM"][-1])))
    report_lines.append("")
    report_lines.append("## Figures")
    report_lines.append("- H_global(t): figures/H_global.png")
    report_lines.append("- X metrics: figures/X_metrics.png")
    report_lines.append("- dm3 + V_EM: figures/dm3_and_vem.png")
    report_lines.append("- Maps: figures/maps.png")
    report_lines.append("- Histograms: figures/histograms.png")
    report_lines.append("- Sensitivity: figures/sensitivity.png")

    if calibration_used:
        report_lines.append("")
        report_lines.append("Calibration artifacts stored in best_params.json and best_candidates.csv.")

    final_report = "\n".join(report_lines)
    with open(results_path / "FINAL_REPORT.md", "w", encoding="utf-8") as f:
        f.write(final_report)

    return report_data
