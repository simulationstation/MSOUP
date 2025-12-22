from __future__ import annotations

import pathlib
from typing import Dict, Iterable

import matplotlib.pyplot as plt
import numpy as np

from .model import h_eff, leak_fraction, pinned_fraction


def plot_af(z_grid: np.ndarray, delta_m: float, output_dir: pathlib.Path, omega_m0: float, omega_L0: float):
    a_vals = pinned_fraction(z_grid, delta_m)
    f_vals = leak_fraction(z_grid, delta_m, omega_m0, omega_L0)
    fig, ax = plt.subplots()
    ax.plot(z_grid, a_vals, label="A(z)")
    ax.plot(z_grid, f_vals, label="f(z)")
    ax.set_xlabel("z")
    ax.legend()
    fig.tight_layout()
    output = output_dir / "af_z.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_hinf_curves(delta_m_grid: np.ndarray, probe_results: Dict[str, dict], output_dir: pathlib.Path, h_early: float):
    fig, ax = plt.subplots()
    for name, res in probe_results.items():
        if name == "fit":
            continue
        f_eff = res.get("f_eff_curve")
        if f_eff is None:
            continue
        h_vals = h_early / np.sqrt(1.0 - np.asarray(f_eff))
        ax.plot(delta_m_grid, h_vals, label=name)
    ax.set_xlabel("Delta_m")
    ax.set_ylabel("H_inf")
    ax.legend()
    fig.tight_layout()
    output = output_dir / "hinf_vs_delta_m.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_kernel_histograms(kernels: Dict[str, tuple], output_dir: pathlib.Path):
    fig, ax = plt.subplots()
    for name, (hist, bin_edges) in kernels.items():
        centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        ax.step(centers, hist, where="mid", label=name)
    ax.set_xlabel("z")
    ax.set_ylabel("Kernel weight density")
    ax.legend()
    fig.tight_layout()
    output = output_dir / "kernel_histograms.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_distance_mode(z_grid: np.ndarray, h_ratio: np.ndarray, delta_mu: np.ndarray, dv: np.ndarray, output_dir: pathlib.Path):
    fig, ax = plt.subplots()
    ax.plot(z_grid, h_ratio, label="H_eff/H_LCDM")
    ax.set_xlabel("z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "h_ratio.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(z_grid, delta_mu, label="Δmu")
    ax.set_xlabel("z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "delta_mu.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(z_grid, dv, label="D_V")
    ax.set_xlabel("z")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "bao_dv.png", dpi=150)
    plt.close(fig)
