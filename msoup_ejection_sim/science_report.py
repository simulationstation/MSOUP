"""Generate comprehensive science report with all analysis sections."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import asdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .config import SimulationConfig
from .dynamics import run_simulation
from .inference import infer_expansion, morphology_stats
from .analysis import (
    run_multiseed, run_convergence, run_identifiability_scan,
    compute_environment_dependence, compute_mechanism_fingerprints,
    aggregate_environment_curves_multiseed, aggregate_fingerprints_multiseed,
    MultiSeedSummary, ConvergenceSummary, IdentifiabilityResult
)


def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def _save_json(path: Path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def plot_multiseed_histograms(summary: MultiSeedSummary, results_dir: Path):
    """Plot histograms of key metrics across seeds."""
    results = summary.individual_results

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    h0_early = [r["H0_early"] for r in results]
    h0_late = [r["H0_late"] for r in results]
    delta_h0 = [r["Delta_H0"] for r in results]
    void_frac = [r["void_fraction"] for r in results]

    axes[0, 0].hist(h0_early, bins=20, color="C0", alpha=0.7, edgecolor="black")
    axes[0, 0].axvline(67.0, color="red", linestyle="--", label="Target 67")
    axes[0, 0].set_xlabel("H0_early")
    axes[0, 0].set_ylabel("Count")
    axes[0, 0].set_title(f"H0_early: {np.mean(h0_early):.2f} +/- {np.std(h0_early):.2f}")
    axes[0, 0].legend()

    axes[0, 1].hist(h0_late, bins=20, color="C1", alpha=0.7, edgecolor="black")
    axes[0, 1].axvline(73.0, color="red", linestyle="--", label="Target 73")
    axes[0, 1].set_xlabel("H0_late")
    axes[0, 1].set_title(f"H0_late: {np.mean(h0_late):.2f} +/- {np.std(h0_late):.2f}")
    axes[0, 1].legend()

    axes[1, 0].hist(delta_h0, bins=20, color="C2", alpha=0.7, edgecolor="black")
    axes[1, 0].set_xlabel("Delta_H0")
    axes[1, 0].set_title(f"Delta_H0: {np.mean(delta_h0):.2f} +/- {np.std(delta_h0):.2f}")

    axes[1, 1].hist(void_frac, bins=20, color="C3", alpha=0.7, edgecolor="black")
    axes[1, 1].set_xlabel("void_fraction")
    axes[1, 1].set_title(f"void_fraction: {np.mean(void_frac):.3f} +/- {np.std(void_frac):.3f}")

    fig.tight_layout()
    fig.savefig(results_dir / "figures" / "multiseed_histograms.png", dpi=150)
    plt.close(fig)


def plot_multiseed_scatter(summary: MultiSeedSummary, results_dir: Path):
    """Scatter plot of H0_early vs H0_late across seeds."""
    results = summary.individual_results

    h0_early = [r["H0_early"] for r in results]
    h0_late = [r["H0_late"] for r in results]

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(h0_early, h0_late, alpha=0.6, edgecolors="black", linewidth=0.5)
    ax.axhline(73.0, color="red", linestyle="--", alpha=0.5, label="H0_late target")
    ax.axvline(67.0, color="blue", linestyle="--", alpha=0.5, label="H0_early target")
    ax.set_xlabel("H0_early")
    ax.set_ylabel("H0_late")
    ax.set_title(f"H0 Scatter (N={len(results)} seeds)")
    ax.legend()

    fig.tight_layout()
    fig.savefig(results_dir / "figures" / "multiseed_scatter.png", dpi=150)
    plt.close(fig)


def plot_convergence(conv_summary: ConvergenceSummary, results_dir: Path):
    """Plot convergence of metrics with resolution."""
    points = conv_summary.points if isinstance(conv_summary.points[0], dict) else [asdict(p) for p in conv_summary.points]

    grids = [p["grid"] for p in points]
    h0_early = [p["H0_early"] for p in points]
    h0_late = [p["H0_late"] for p in points]
    delta_h0 = [p["Delta_H0"] for p in points]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].plot(grids, h0_early, "o-", color="C0", markersize=8)
    axes[0].axhline(67.0, color="red", linestyle="--", alpha=0.5)
    axes[0].set_xlabel("Grid size")
    axes[0].set_ylabel("H0_early")
    axes[0].set_title("H0_early convergence")

    axes[1].plot(grids, h0_late, "o-", color="C1", markersize=8)
    axes[1].axhline(73.0, color="red", linestyle="--", alpha=0.5)
    axes[1].set_xlabel("Grid size")
    axes[1].set_ylabel("H0_late")
    axes[1].set_title("H0_late convergence")

    axes[2].plot(grids, delta_h0, "o-", color="C2", markersize=8)
    axes[2].set_xlabel("Grid size")
    axes[2].set_ylabel("Delta_H0")
    axes[2].set_title("Delta_H0 convergence")

    fig.tight_layout()
    fig.savefig(results_dir / "figures" / "convergence.png", dpi=150)
    plt.close(fig)


def plot_environment_curves(env_curves: Dict, results_dir: Path):
    """Plot H_local vs environment (density, binding)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Density curve
    if "density_H_mean" in env_curves:
        centers = env_curves["bin_centers"]
        mean = env_curves["density_H_mean"]
        std = env_curves["density_H_std"]

        axes[0].plot(centers, mean, "o-", color="C0", label="H_local mean")
        axes[0].fill_between(centers,
                             np.array(mean) - np.array(std),
                             np.array(mean) + np.array(std),
                             alpha=0.3, color="C0")
        axes[0].set_xlabel("Density percentile")
        axes[0].set_ylabel("H_local")
        axes[0].set_title("H_local vs Density")
        axes[0].legend()

    # Binding curve
    if "binding_H_mean" in env_curves:
        mean = env_curves["binding_H_mean"]
        std = env_curves["binding_H_std"]

        axes[1].plot(centers, mean, "o-", color="C1", label="H_local mean")
        axes[1].fill_between(centers,
                             np.array(mean) - np.array(std),
                             np.array(mean) + np.array(std),
                             alpha=0.3, color="C1")
        axes[1].set_xlabel("Binding percentile")
        axes[1].set_ylabel("H_local")
        axes[1].set_title("H_local vs Binding")
        axes[1].legend()

    fig.tight_layout()
    fig.savefig(results_dir / "figures" / "environment_curves.png", dpi=150)
    plt.close(fig)


def plot_identifiability_corner(ident_result: IdentifiabilityResult, results_dir: Path):
    """Plot corner-like scatter for key parameters."""
    corner_data = ident_result.corner_data if isinstance(ident_result.corner_data, dict) else ident_result.corner_data

    key_params = ["beta", "beta_loc", "gamma0", "tau_dm3_0"]
    n_params = len(key_params)

    fig, axes = plt.subplots(n_params, n_params, figsize=(12, 12))

    stage1_valid = np.array(corner_data["stage1_valid"])
    stage2_valid = np.array(corner_data["stage2_valid"])

    for i, pi in enumerate(key_params):
        for j, pj in enumerate(key_params):
            ax = axes[i, j]

            if i == j:
                # Histogram on diagonal
                ax.hist(corner_data[pi], bins=30, alpha=0.5, color="gray")
                valid_vals = np.array(corner_data[pi])[stage1_valid]
                if len(valid_vals) > 0:
                    ax.hist(valid_vals, bins=30, alpha=0.7, color="C0", label="Stage1 valid")
                ax.set_xlabel(pi)
            elif i > j:
                # Scatter below diagonal
                x = np.array(corner_data[pj])
                y = np.array(corner_data[pi])
                ax.scatter(x[~stage1_valid], y[~stage1_valid], alpha=0.2, s=5, color="gray")
                ax.scatter(x[stage1_valid], y[stage1_valid], alpha=0.6, s=10, color="C0", label="Stage1")
                ax.scatter(x[stage2_valid], y[stage2_valid], alpha=0.8, s=15, color="C2", label="Stage2")
                ax.set_xlabel(pj)
                ax.set_ylabel(pi)
            else:
                # Upper triangle - correlation with H0_late
                x = np.array(corner_data[pj])
                y = np.array(corner_data["H0_late"])
                ax.scatter(x, y, alpha=0.3, s=5, c=corner_data["Delta_H0"], cmap="viridis")
                ax.set_xlabel(pj)
                ax.set_ylabel("H0_late")

    fig.tight_layout()
    fig.savefig(results_dir / "figures" / "identifiability_corner.png", dpi=150)
    plt.close(fig)


def generate_science_report(
    results_dir: str,
    cfg: SimulationConfig,
    stage1_result: Optional[Dict] = None,
    stage2_result: Optional[Dict] = None,
    multiseed_summary: Optional[MultiSeedSummary] = None,
    convergence_summary: Optional[ConvergenceSummary] = None,
    environment_curves: Optional[Dict] = None,
    fingerprints: Optional[Dict] = None,
    identifiability_result: Optional[IdentifiabilityResult] = None,
) -> str:
    """Generate the comprehensive SCIENCE_REPORT.md."""

    results_path = Path(results_dir)
    _ensure_dir(results_path)
    _ensure_dir(results_path / "figures")

    lines = []
    lines.append("# Msoup Ejection Simulation: Scientific Validation Report")
    lines.append("")
    lines.append("This report presents the scientific validation of the Msoup ejection + decompression")
    lines.append("model, including out-of-sample predictions, robustness checks, and convergence analysis.")
    lines.append("")

    # === Stage 1 Analysis ===
    if stage1_result:
        lines.append("## 1. Stage 1 Calibration (Early-Only)")
        lines.append("")
        lines.append("**Critical test**: Does fitting ONLY H0_early produce H0_late near 73 as an *emergent* prediction?")
        lines.append("")
        lines.append(f"- **Target**: H0_early = 67.0")
        lines.append(f"- **Achieved**: H0_early = {stage1_result['H0_early_achieved']:.3f}")
        lines.append(f"- **Emergent H0_late** (NOT targeted): {stage1_result['H0_late_emergent']:.3f}")
        lines.append(f"- **Delta_H0**: {stage1_result['Delta_H0_emergent']:.3f}")
        lines.append(f"- **Deviation from 73**: {stage1_result['H0_late_deviation_from_73']:+.3f}")
        lines.append("")

        dev = abs(stage1_result['H0_late_deviation_from_73'])
        if dev < 1.0:
            lines.append("**RESULT**: H0_late is CLOSE to 73 without explicit targeting - the late-time effect is largely EMERGENT.")
        elif dev < 3.0:
            lines.append("**RESULT**: H0_late is MODERATELY close to 73 - partial emergence with some tuning needed.")
        else:
            lines.append("**RESULT**: H0_late is FAR from 73 - the late-time effect requires explicit targeting.")
        lines.append("")

    # === Stage 2 Analysis ===
    if stage2_result:
        lines.append("## 2. Stage 2 Calibration (Both Targets)")
        lines.append("")
        lines.append(f"- **Targets**: H0_early = 67.0, H0_late = 73.0")
        lines.append(f"- **Achieved H0_early**: {stage2_result['H0_early_achieved']:.3f}")
        lines.append(f"- **Achieved H0_late**: {stage2_result['H0_late_achieved']:.3f}")
        lines.append(f"- **Delta_H0**: {stage2_result['Delta_H0_achieved']:.3f}")
        lines.append("")

        if stage2_result.get("param_drift_from_stage1"):
            lines.append("**Parameter drift from Stage 1**:")
            for k, v in stage2_result["param_drift_from_stage1"].items():
                lines.append(f"  - {k}: {v:+.4f}")
            lines.append("")

    # === Multi-Seed Robustness ===
    if multiseed_summary:
        lines.append("## 3. Seed Robustness Analysis")
        lines.append("")
        lines.append(f"Ran {multiseed_summary.n_seeds} seeds with fixed parameters.")
        lines.append("")
        lines.append("| Metric | Mean | Std | CV (%) |")
        lines.append("|--------|------|-----|--------|")

        cv_early = 100 * multiseed_summary.H0_early_std / multiseed_summary.H0_early_mean if multiseed_summary.H0_early_mean != 0 else 0
        cv_late = 100 * multiseed_summary.H0_late_std / multiseed_summary.H0_late_mean if multiseed_summary.H0_late_mean != 0 else 0
        cv_delta = 100 * multiseed_summary.Delta_H0_std / multiseed_summary.Delta_H0_mean if multiseed_summary.Delta_H0_mean != 0 else 0

        lines.append(f"| H0_early | {multiseed_summary.H0_early_mean:.3f} | {multiseed_summary.H0_early_std:.3f} | {cv_early:.2f} |")
        lines.append(f"| H0_late | {multiseed_summary.H0_late_mean:.3f} | {multiseed_summary.H0_late_std:.3f} | {cv_late:.2f} |")
        lines.append(f"| Delta_H0 | {multiseed_summary.Delta_H0_mean:.3f} | {multiseed_summary.Delta_H0_std:.3f} | {cv_delta:.2f} |")
        lines.append(f"| void_fraction | {multiseed_summary.void_fraction_mean:.4f} | {multiseed_summary.void_fraction_std:.4f} | - |")
        lines.append("")
        lines.append("**Quantiles (Delta_H0)**:")
        if multiseed_summary.quantiles and "Delta_H0" in multiseed_summary.quantiles:
            q = multiseed_summary.quantiles["Delta_H0"]
            lines.append(f"  - 5th: {q.get('q5', 'N/A'):.2f}, 25th: {q.get('q25', 'N/A'):.2f}, 50th: {q.get('q50', 'N/A'):.2f}, 75th: {q.get('q75', 'N/A'):.2f}, 95th: {q.get('q95', 'N/A'):.2f}")
        lines.append("")
        lines.append("![Multiseed Histograms](figures/multiseed_histograms.png)")
        lines.append("")
        lines.append("![Multiseed Scatter](figures/multiseed_scatter.png)")
        lines.append("")

        # Plot
        plot_multiseed_histograms(multiseed_summary, results_path)
        plot_multiseed_scatter(multiseed_summary, results_path)

    # === Convergence ===
    if convergence_summary:
        lines.append("## 4. Resolution Convergence")
        lines.append("")
        points = convergence_summary.points if isinstance(convergence_summary.points[0], dict) else [asdict(p) for p in convergence_summary.points]

        lines.append("| Grid | Steps | H0_early | H0_late | Delta_H0 | Runtime (s) |")
        lines.append("|------|-------|----------|---------|----------|-------------|")
        for p in points:
            lines.append(f"| {p['grid']} | {p['steps']} | {p['H0_early']:.3f} | {p['H0_late']:.3f} | {p['Delta_H0']:.3f} | {p['runtime_estimate']:.2f} |")
        lines.append("")

        lines.append(f"**Drift (low to high res)**:")
        lines.append(f"  - H0_early: {convergence_summary.H0_early_drift:.3f}")
        lines.append(f"  - H0_late: {convergence_summary.H0_late_drift:.3f}")
        lines.append(f"  - Delta_H0: {convergence_summary.Delta_H0_drift:.3f}")
        lines.append(f"  - Converged (threshold {convergence_summary.convergence_threshold}): {'YES' if convergence_summary.converged else 'NO'}")
        lines.append("")
        lines.append("![Convergence](figures/convergence.png)")
        lines.append("")

        plot_convergence(convergence_summary, results_path)

    # === Non-Targeted Predictions ===
    lines.append("## 5. Non-Targeted Predictions")
    lines.append("")

    if environment_curves:
        lines.append("### 5a. Environment Dependence")
        lines.append("")
        lines.append("These curves show how H_local varies with local density and binding.")
        lines.append("This is a PREDICTION, not a calibration target.")
        lines.append("")
        lines.append("![Environment Curves](figures/environment_curves.png)")
        lines.append("")

        plot_environment_curves(environment_curves, results_path)

    if fingerprints:
        lines.append("### 5b. Mechanism Fingerprints")
        lines.append("")
        lines.append("Correlations that test the underlying mechanism:")
        lines.append("")
        lines.append(f"- dm3_decay vs A_increase correlation: {fingerprints.get('dm3_decay_A_increase_corr_mean', 'N/A'):.3f} +/- {fingerprints.get('dm3_decay_A_increase_corr_std', 0):.3f}")
        lines.append(f"- dm3_pockets vs H_hotspots correlation: {fingerprints.get('dm3_pockets_H_hotspots_corr_mean', 'N/A'):.3f} +/- {fingerprints.get('dm3_pockets_H_hotspots_corr_std', 0):.3f}")
        lines.append(f"- dm3_decay vs wlt2 correlation: {fingerprints.get('dm3_decay_wlt2_corr_mean', 'N/A'):.3f} +/- {fingerprints.get('dm3_decay_wlt2_corr_std', 0):.3f}")
        lines.append("")

        # Interpretation
        dm3_h_corr = fingerprints.get('dm3_pockets_H_hotspots_corr_mean', 0)
        if dm3_h_corr < -0.1:
            lines.append("**Interpretation**: Negative dm3-H correlation supports the mechanism - regions that had more dm3 initially now show lower H_local.")
        elif dm3_h_corr > 0.1:
            lines.append("**Interpretation**: Positive dm3-H correlation - regions with more initial dm3 show higher H_local (peel-off dominated).")
        else:
            lines.append("**Interpretation**: Weak dm3-H correlation suggests complex interplay of mechanisms.")
        lines.append("")

    # === Identifiability ===
    if identifiability_result:
        lines.append("## 6. Parameter Identifiability")
        lines.append("")
        lines.append(f"Scanned {identifiability_result.n_samples} parameter sets around the best solution.")
        lines.append("")
        lines.append(f"- **Stage 1 valid** (H0_early within {identifiability_result.stage1_tolerance}): {identifiability_result.n_stage1_valid} ({100*identifiability_result.n_stage1_valid/identifiability_result.n_samples:.1f}%)")
        lines.append(f"- **Stage 2 valid** (both within tolerance): {identifiability_result.n_stage2_valid} ({100*identifiability_result.n_stage2_valid/identifiability_result.n_samples:.1f}%)")
        lines.append("")

        if identifiability_result.n_stage1_valid > identifiability_result.n_samples * 0.1:
            lines.append("**Assessment**: Parameter space is relatively BROAD - solutions are not fine-tuned.")
        elif identifiability_result.n_stage1_valid > identifiability_result.n_samples * 0.01:
            lines.append("**Assessment**: Parameter space is MODERATELY constrained - some tuning required.")
        else:
            lines.append("**Assessment**: Parameter space is NARROW - solutions appear fine-tuned.")
        lines.append("")
        lines.append("![Identifiability Corner](figures/identifiability_corner.png)")
        lines.append("")

        plot_identifiability_corner(identifiability_result, results_path)

    # === Summary ===
    lines.append("## 7. Summary")
    lines.append("")

    if stage1_result:
        dev = stage1_result['H0_late_deviation_from_73']
        lines.append(f"1. **Stage 1 (early-only) H0_late emergence**: {stage1_result['H0_late_emergent']:.2f} (deviation from 73: {dev:+.2f})")

    if multiseed_summary:
        lines.append(f"2. **Seed sensitivity of Delta_H0**: std = {multiseed_summary.Delta_H0_std:.3f} (mean = {multiseed_summary.Delta_H0_mean:.2f})")

    if convergence_summary:
        lines.append(f"3. **Resolution convergence**: {'CONVERGED' if convergence_summary.converged else 'NOT fully converged'} (drift = {convergence_summary.Delta_H0_drift:.3f})")

    if identifiability_result:
        frac = 100 * identifiability_result.n_stage1_valid / identifiability_result.n_samples
        lines.append(f"4. **Parameter identifiability**: {frac:.1f}% of parameter space valid for Stage 1")

    lines.append("")
    lines.append("---")
    lines.append("*Report generated by msoup_ejection_sim science pipeline*")

    report_text = "\n".join(lines)

    with open(results_path / "SCIENCE_REPORT.md", "w", encoding="utf-8") as f:
        f.write(report_text)

    return report_text
