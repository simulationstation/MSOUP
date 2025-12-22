"""
Plotting utilities for the MV-O1B pocket test.

Generates all required diagnostic plots:
- Candidate count histograms
- Fano factor vs threshold
- C_obs/C_excess distributions
- Null calibration plots
- Geometry fit diagnostics (if enabled)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import PocketConfig
    from .geometry import GeometryResult
    from .stats import GlobalStats


def plot_candidate_histogram(candidates: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Plot distribution of candidate counts per sensor."""
    if candidates.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "candidate_count_histogram.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Counts per sensor
    counts = candidates.groupby("sensor").size()
    axes[0].hist(counts, bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel("Candidates per sensor")
    axes[0].set_ylabel("Number of sensors")
    axes[0].set_title(f"Candidate Distribution (N={len(candidates)}, {len(counts)} sensors)")
    axes[0].axvline(counts.mean(), color='red', linestyle='--', label=f'Mean={counts.mean():.1f}')
    axes[0].legend()

    # Peak values
    if "peak_value" in candidates.columns:
        axes[1].hist(candidates["peak_value"].dropna(), bins=50, edgecolor='black', alpha=0.7)
        axes[1].set_xlabel("Peak value (standardized)")
        axes[1].set_ylabel("Count")
        axes[1].set_title("Candidate Amplitude Distribution")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_fano_vs_threshold(
    candidates: pd.DataFrame,
    thresholds: List[float],
    output_dir: Path,
) -> Optional[Path]:
    """Plot Fano factor as function of detection threshold."""
    if candidates.empty or "peak_value" not in candidates.columns:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "fano_vs_threshold.png"

    fanos = []
    n_candidates = []
    for thresh in thresholds:
        subset = candidates[candidates["peak_value"].abs() >= thresh]
        if subset.empty:
            fanos.append(np.nan)
            n_candidates.append(0)
            continue
        counts = subset.groupby("sensor").size()
        mean = counts.mean()
        var = counts.var(ddof=1) if len(counts) > 1 else 0
        fanos.append(var / mean if mean > 0 else np.nan)
        n_candidates.append(len(subset))

    fig, ax1 = plt.subplots(figsize=(8, 5))

    ax1.plot(thresholds, fanos, 'b-o', label='Fano factor')
    ax1.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='Poisson (Fano=1)')
    ax1.set_xlabel("Detection threshold (σ)")
    ax1.set_ylabel("Fano factor", color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    ax2.plot(thresholds, n_candidates, 'r--s', alpha=0.7, label='N candidates')
    ax2.set_ylabel("Number of candidates", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    plt.title("Fano Factor vs Detection Threshold")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_clustering_distribution(
    stats: "GlobalStats",
    output_dir: Path,
) -> Optional[Path]:
    """Plot C_obs vs null distribution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "clustering_distribution.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # C_obs vs null
    x = np.linspace(
        stats.c_null_mean - 4 * stats.c_null_std,
        max(stats.c_obs, stats.c_null_mean + 4 * stats.c_null_std),
        100
    )
    if stats.c_null_std > 0:
        y = np.exp(-0.5 * ((x - stats.c_null_mean) / stats.c_null_std) ** 2)
        axes[0].fill_between(x, y, alpha=0.3, color='blue', label='Null distribution')
        axes[0].plot(x, y, 'b-')
    axes[0].axvline(stats.c_obs, color='red', linewidth=2, label=f'C_obs = {stats.c_obs:.4f}')
    axes[0].axvline(stats.c_null_mean, color='blue', linestyle='--', label=f'Null mean = {stats.c_null_mean:.4f}')
    axes[0].set_xlabel("Clustering statistic C")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"C_obs vs Null (p = {stats.p_clustering:.4f})")
    axes[0].legend()

    # C_excess with uncertainty
    axes[1].bar(['C_excess'], [stats.c_excess], yerr=[stats.c_excess_std], capsize=10, color='steelblue', alpha=0.7)
    axes[1].axhline(0, color='gray', linestyle='--')
    axes[1].set_ylabel("C_excess")
    axes[1].set_title(f"C_excess = {stats.c_excess:.4f} ± {stats.c_excess_std:.4f}")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_null_calibration(
    stats: "GlobalStats",
    output_dir: Path,
) -> Optional[Path]:
    """Plot null calibration diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "null_calibration.png"

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Fano null distribution
    x = np.linspace(
        stats.fano_null_mean - 4 * max(stats.fano_null_std, 0.1),
        max(stats.fano_obs, stats.fano_null_mean + 4 * max(stats.fano_null_std, 0.1)) if not np.isnan(stats.fano_obs) else stats.fano_null_mean + 4 * max(stats.fano_null_std, 0.1),
        100
    )
    if stats.fano_null_std > 0:
        y = np.exp(-0.5 * ((x - stats.fano_null_mean) / stats.fano_null_std) ** 2)
        axes[0].fill_between(x, y, alpha=0.3, color='green')
        axes[0].plot(x, y, 'g-')
    if not np.isnan(stats.fano_obs):
        axes[0].axvline(stats.fano_obs, color='red', linewidth=2, label=f'Fano_obs = {stats.fano_obs:.2f}')
    axes[0].axvline(1.0, color='gray', linestyle='--', label='Poisson (Fano=1)')
    axes[0].set_xlabel("Fano factor")
    axes[0].set_ylabel("Density")
    axes[0].set_title(f"Fano: obs={stats.fano_obs:.2f}, null={stats.fano_null_mean:.2f}±{stats.fano_null_std:.2f}")
    axes[0].legend()

    # Z-scores summary
    z_scores = [stats.fano_z, stats.c_z]
    labels = ['Fano Z', 'Clustering Z']
    colors = ['green' if abs(z) < 2 else 'orange' if abs(z) < 3 else 'red' for z in z_scores]
    axes[1].bar(labels, z_scores, color=colors, alpha=0.7)
    axes[1].axhline(0, color='gray', linestyle='-')
    axes[1].axhline(2, color='orange', linestyle='--', alpha=0.5)
    axes[1].axhline(-2, color='orange', linestyle='--', alpha=0.5)
    axes[1].set_ylabel("Z-score")
    axes[1].set_title("Summary Z-scores (|Z|<2: green, 2-3: orange, >3: red)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_geometry_diagnostics(
    geom: "GeometryResult",
    output_dir: Path,
) -> Optional[Path]:
    """Plot geometry fit diagnostics."""
    if geom.chi2_obs is None or geom.n_events == 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "geometry_diagnostics.png"

    fig, ax = plt.subplots(figsize=(8, 5))

    # Chi-squared distribution under null
    if geom.null_chi2s is not None and len(geom.null_chi2s) > 0:
        ax.hist(geom.null_chi2s, bins=30, alpha=0.5, density=True, label='Null χ²')

    ax.axvline(geom.chi2_obs, color='red', linewidth=2, label=f'χ²_obs = {geom.chi2_obs:.2f}')
    ax.set_xlabel("χ² statistic")
    ax.set_ylabel("Density")
    ax.set_title(f"Geometry Test: χ² = {geom.chi2_obs:.2f}, p = {geom.p_value:.4f}, N = {geom.n_events}")
    ax.legend()

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def plot_window_coverage(windows: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Plot coverage per sensor."""
    if windows.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "window_coverage.png"

    fig, ax = plt.subplots(figsize=(10, max(6, len(windows["sensor"].unique()) * 0.15)))

    coverage = (windows["end"] - windows["start"]).dt.total_seconds() / 3600
    coverage_by_sensor = coverage.groupby(windows["sensor"]).sum().sort_values()

    # Limit to top 50 sensors for readability
    if len(coverage_by_sensor) > 50:
        coverage_by_sensor = coverage_by_sensor.tail(50)

    coverage_by_sensor.plot(kind="barh", ax=ax, color='steelblue', alpha=0.7)
    ax.set_xlabel("Coverage (hours)")
    ax.set_title(f"Sensor Coverage ({len(windows['sensor'].unique())} sensors)")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return path


def generate_all_plots(
    output_dir: Path,
    candidate_frame: pd.DataFrame,
    window_frame: pd.DataFrame,
    stats: "GlobalStats",
    geom: "GeometryResult",
    cfg: "PocketConfig",
) -> List[Path]:
    """Generate all required diagnostic plots."""
    plots = []

    # Candidate histogram
    p = plot_candidate_histogram(candidate_frame, output_dir)
    if p:
        plots.append(p)

    # Fano vs threshold
    thresholds = np.linspace(2.0, 6.0, 9).tolist()
    p = plot_fano_vs_threshold(candidate_frame, thresholds, output_dir)
    if p:
        plots.append(p)

    # Clustering distribution
    p = plot_clustering_distribution(stats, output_dir)
    if p:
        plots.append(p)

    # Null calibration
    p = plot_null_calibration(stats, output_dir)
    if p:
        plots.append(p)

    # Geometry diagnostics
    p = plot_geometry_diagnostics(geom, output_dir)
    if p:
        plots.append(p)

    # Window coverage
    p = plot_window_coverage(window_frame, output_dir)
    if p:
        plots.append(p)

    print(f"Generated {len(plots)} plots")
    return plots
