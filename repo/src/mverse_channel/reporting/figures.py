"""Figure generation.

Produces visualization of coherence magnitude (phase-robust),
detection curves with calibrated thresholds, and other diagnostics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from mverse_channel.metrics.correlations import cross_spectral_density, coherence_magnitude


def plot_coherence(series: Dict[str, np.ndarray], fs: float, out_path: Path) -> None:
    """Plot coherence magnitude spectrum (phase-robust)."""
    freqs, coh_mag = coherence_magnitude(series["ia"], series["ib"], fs)
    plt.figure(figsize=(6, 4))
    plt.plot(freqs, coh_mag)
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Coherence Magnitude |Coh(f)|")
    plt.title("Cross-channel Coherence Magnitude (Phase-Robust)")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_anomaly_scores(scores: Dict[str, float], out_path: Path) -> None:
    """Plot anomaly scores as bar chart."""
    labels = list(scores.keys())
    values = list(scores.values())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Anomaly Score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_detection_curve(
    results: Dict[str, list[float]],
    out_path: Path,
    alpha: Optional[float] = None,
) -> None:
    """Plot detection probability vs epsilon with calibrated thresholds.

    Args:
        results: Sweep results with epsilon and detection_prob
        out_path: Output file path
        alpha: If provided, show horizontal line for target FPR
    """
    epsilon = np.array(results["epsilon"])
    detection = np.array(results["detection_prob"])
    unique_eps = sorted(set(epsilon))
    avg_detection = [float(np.mean(detection[epsilon == eps])) for eps in unique_eps]

    plt.figure(figsize=(6, 4))
    plt.plot(unique_eps, avg_detection, marker="o", linewidth=2, label="Detection prob")

    # Show alpha line if provided
    if alpha is not None:
        plt.axhline(y=alpha, color="r", linestyle="--", alpha=0.7,
                    label=f"Target FPR (α={alpha})")

    plt.xlabel("Hidden Channel Strength (ε)")
    plt.ylabel("Detection Probability")
    plt.title("Power Curve with Calibrated Thresholds")
    plt.xscale("linear")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_null_distribution(
    null_values: list[float],
    threshold: float,
    metric_name: str,
    out_path: Path,
) -> None:
    """Plot null distribution with threshold line."""
    plt.figure(figsize=(6, 4))
    plt.hist(null_values, bins=20, density=True, alpha=0.7, label="Null dist")
    plt.axvline(x=threshold, color="r", linestyle="--", linewidth=2,
                label=f"Threshold={threshold:.3f}")
    plt.xlabel(metric_name)
    plt.ylabel("Density")
    plt.title(f"Null Distribution: {metric_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
