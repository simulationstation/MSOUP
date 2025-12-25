"""Plotting helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_mu_over_t(dual_sweep: list[dict], output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)
    ln2 = np.log(2)
    densities = sorted({entry["density"] for entry in dual_sweep})
    fig, ax = plt.subplots(figsize=(7, 4))
    for density in densities:
        subset = [e for e in dual_sweep if e["density"] == density]
        j_vals = np.array([e["j_over_t"] for e in subset])
        mu_vals = np.array([e["mu_over_t"] for e in subset])
        order = np.argsort(j_vals)
        ax.plot(j_vals[order], mu_vals[order], marker="o", label=f"m/n={density}")
    ax.axhline(ln2, color="black", linestyle="--", label="ln2")
    ax.set_xscale("log")
    ax.set_xlabel("J/T")
    ax.set_ylabel(r"$\mu/T$")
    ax.set_title("Constraint chemical potential")
    ax.legend(fontsize=8)
    path = output_dir / "mu_over_t.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)


def plot_coarsegrain(coarsegrain: dict, output_dir: Path) -> str:
    output_dir.mkdir(parents=True, exist_ok=True)

    def aggregate(records: list[list[dict]]):
        blocks = sorted({entry["block_size"] for record in records for entry in record})
        means = []
        stds = []
        for block in blocks:
            values = [
                entry["delta_m"]
                for record in records
                for entry in record
                if entry["block_size"] == block
            ]
            means.append(np.mean(values))
            stds.append(np.std(values))
        return np.array(blocks), np.array(means), np.array(stds)

    blocks1, mean1, std1 = aggregate(coarsegrain["cg1"])
    blocks2, mean2, std2 = aggregate(coarsegrain["cg2"])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(blocks1, mean1, yerr=std1, label="CG1 block-sum", marker="o")
    ax.errorbar(blocks2, mean2, yerr=std2, label="CG2 random", marker="s")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("block size ℓ")
    ax.set_ylabel("Δm(ℓ)")
    ax.set_title("Coarse-grained Δm")
    ax.legend(fontsize=8)
    path = output_dir / "delta_m_coarsegrain.png"
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return str(path)
