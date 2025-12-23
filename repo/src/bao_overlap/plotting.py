"""Plotting utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_wedge(s: np.ndarray, xi: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(s, xi, marker="o")
    ax.set_xlabel(r"$s$ [$h^{-1}$ Mpc]")
    ax.set_ylabel(r"$\xi$")
    ax.set_title(title)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_beta_null(beta_values: np.ndarray, path: Path, beta_obs: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(beta_values, bins=20, alpha=0.7)
    if beta_obs is not None:
        ax.axvline(beta_obs, color="red", linestyle="--", label="observed")
        ax.legend()
    ax.set_xlabel(r"$\beta$")
    ax.set_ylabel("Count")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
