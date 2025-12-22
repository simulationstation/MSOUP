"""
Optional plotting utilities for the pocket test.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd


def plot_candidate_histogram(candidates: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Plot distribution of candidate amplitudes."""
    if candidates.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "candidate_histogram.png"
    plt.figure(figsize=(6, 4))
    candidates["peak_value"].hist(bins=30)
    plt.xlabel("Peak value")
    plt.ylabel("Count")
    plt.title("Candidate amplitude distribution")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_window_coverage(windows: pd.DataFrame, output_dir: Path) -> Optional[Path]:
    """Plot coverage per sensor."""
    if windows.empty:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "window_coverage.png"
    plt.figure(figsize=(6, 4))
    coverage = (windows["end"] - windows["start"]).dt.total_seconds() / 3600
    coverage.groupby(windows["sensor"]).sum().sort_values().plot(kind="barh")
    plt.xlabel("Coverage (hours)")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
