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


def plot_sn_fit(
    z_grid: np.ndarray,
    mu_model: np.ndarray,
    z_obs: np.ndarray,
    mu_obs: np.ndarray,
    sigma: np.ndarray,
    probe_name: str,
    output_dir: pathlib.Path,
    b_hat: float = 0.0,
):
    """Plot SN distance modulus: theory curve + observed data points."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Theory curve
    ax.plot(z_grid, mu_model, "b-", lw=1.5, label="Model", zorder=1)

    # Observed data points with error bars (apply offset correction for display)
    mu_corrected = mu_obs - b_hat
    ax.errorbar(
        z_obs, mu_corrected, yerr=sigma,
        fmt=".", color="gray", alpha=0.3, markersize=2,
        elinewidth=0.5, capsize=0, label=f"Data (n={len(z_obs)})", zorder=0
    )

    ax.set_xlabel("Redshift z")
    ax.set_ylabel("Distance modulus μ")
    ax.set_title(f"SN {probe_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    output = output_dir / f"sn_{probe_name}_fit.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_bao_fit(
    z_grid: np.ndarray,
    dv_rd_model: np.ndarray,
    z_obs: np.ndarray,
    obs_values: np.ndarray,
    sigma: np.ndarray,
    probe_name: str,
    output_dir: pathlib.Path,
):
    """Plot BAO DV/rd: theory curve + observed data points."""
    fig, ax = plt.subplots(figsize=(8, 5))

    # Theory curve
    ax.plot(z_grid, dv_rd_model, "b-", lw=1.5, label="Model", zorder=1)

    # Observed data points with error bars
    ax.errorbar(
        z_obs, obs_values, yerr=sigma,
        fmt="o", color="red", markersize=5,
        elinewidth=1.5, capsize=3, label=f"Data (n={len(z_obs)})", zorder=2
    )

    ax.set_xlabel("Redshift z")
    ax.set_ylabel("DV / rd")
    ax.set_title(f"BAO {probe_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    output = output_dir / f"bao_{probe_name}_fit.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)


def plot_td_fit(
    z_lens: np.ndarray,
    ddt_obs: np.ndarray,
    ddt_pred: np.ndarray,
    sigma: np.ndarray,
    probe_name: str,
    output_dir: pathlib.Path,
):
    """Plot time-delay D_dt: predicted vs observed."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # Data points with error bars
    ax.errorbar(
        z_lens, ddt_obs, yerr=sigma,
        fmt="o", color="red", markersize=6,
        elinewidth=1.5, capsize=3, label="Observed", zorder=2
    )

    # Predicted values
    ax.scatter(z_lens, ddt_pred, marker="s", color="blue", s=40, label="Predicted", zorder=3)

    # Connect obs-pred pairs
    for i in range(len(z_lens)):
        ax.plot([z_lens[i], z_lens[i]], [ddt_obs[i], ddt_pred[i]], "k--", lw=0.8, alpha=0.5)

    ax.set_xlabel("Lens redshift z_l")
    ax.set_ylabel("D_dt [Mpc]")
    ax.set_title(f"Time-Delay {probe_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()

    output = output_dir / f"td_{probe_name}_fit.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)
