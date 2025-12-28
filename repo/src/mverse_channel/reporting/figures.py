"""Figure generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np

from mverse_channel.metrics.correlations import cross_spectral_density


def plot_coherence(series: Dict[str, np.ndarray], fs: float, out_path: Path) -> None:
    freqs, _, _, coherence = cross_spectral_density(series["ia"], series["ib"], fs)
    plt.figure(figsize=(6, 4))
    plt.plot(freqs, coherence)
    plt.xlabel("Frequency")
    plt.ylabel("Coherence")
    plt.title("Cross-channel coherence")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_anomaly_scores(scores: Dict[str, float], out_path: Path) -> None:
    labels = list(scores.keys())
    values = list(scores.values())
    plt.figure(figsize=(6, 4))
    plt.bar(labels, values)
    plt.ylabel("Anomaly score")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_detection_curve(results: Dict[str, list[float]], out_path: Path) -> None:
    epsilon = np.array(results["epsilon"])
    detection = np.array(results["detection_prob"])
    unique_eps = sorted(set(epsilon))
    avg_detection = [float(np.mean(detection[epsilon == eps])) for eps in unique_eps]
    plt.figure(figsize=(6, 4))
    plt.plot(unique_eps, avg_detection, marker="o")
    plt.xlabel("Hidden channel scale")
    plt.ylabel("Detection probability (avg over tau/rho)")
    plt.xscale("linear")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
