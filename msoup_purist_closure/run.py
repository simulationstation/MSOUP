from __future__ import annotations

import argparse
import pathlib
import sys
from typing import Dict

import numpy as np
from .config import MsoupConfig, load_config
from .distances import angular_diameter_distance, comoving_distance, h_eff_ratio, luminosity_distance
from .fit import fit_delta_m
from .kernels import kernel_weights, load_probe_csv
from .observables import bao_dv, distance_modulus
from .plots import plot_af, plot_distance_mode, plot_hinf_curves, plot_kernel_histograms
from .report import write_report


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Purist MSOUP closure runner")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["kernel", "distance", "both"], default="kernel")
    return parser.parse_args(argv)


def run_kernel_mode(cfg: MsoupConfig, output_dir: pathlib.Path) -> Dict[str, dict]:
    delta_m_star, probe_results = fit_delta_m(cfg)

    # curves for plotting
    delta_m_grid = np.linspace(cfg.fit.delta_m_bounds[0], cfg.fit.delta_m_bounds[1], 100)
    kernels_hist = {}
    for probe in cfg.probes:
        df = load_probe_csv(probe.path, z_column=probe.z_column, sigma_column=probe.sigma_column)
        kw = kernel_weights(df, probe.kernel, z_column=probe.z_column, sigma_column=probe.sigma_column)
        f_curve = []
        for dm in delta_m_grid:
            from .model import leak_fraction

            f_vals = leak_fraction(kw.z, dm, cfg.omega_m0, cfg.omega_L0)
            f_curve.append(float(np.sum(kw.weights * f_vals)))
        probe_results[probe.name]["f_eff_curve"] = f_curve
        kernels_hist[probe.name] = kw.histogram()

    from .plots import plot_af

    z_grid = np.linspace(0, 2, 200)
    plot_af(z_grid, delta_m_star, output_dir, cfg.omega_m0, cfg.omega_L0)
    plot_hinf_curves(delta_m_grid, probe_results, output_dir, cfg.h_early)
    plot_kernel_histograms(kernels_hist, output_dir)
    return probe_results


def run_distance_mode(cfg: MsoupConfig, delta_m_star: float, output_dir: pathlib.Path):
    z_max = cfg.distance.z_max
    z_grid = np.linspace(0, z_max, cfg.distance.num_z_samples)
    h_ratio = h_eff_ratio(z_grid, delta_m_star, cfg.h_early, cfg.omega_m0, cfg.omega_L0)

    mu = distance_modulus(z_grid, delta_m=delta_m_star, h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0, nuisance_M=cfg.distance.nuisance_M)
    lcdm_mu = distance_modulus(z_grid, delta_m=0.0, h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0, nuisance_M=cfg.distance.nuisance_M)
    delta_mu = mu - lcdm_mu
    dv = bao_dv(z_grid, delta_m=delta_m_star, h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0)
    plot_distance_mode(z_grid, h_ratio, delta_mu, dv, output_dir)


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    output_dir = cfg.results_dir / cfg.timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_results = None
    if args.mode in {"kernel", "both"}:
        probe_results = run_kernel_mode(cfg, output_dir)
    if args.mode in {"distance", "both"}:
        if probe_results is None:
            delta_m_star = 0.0
        else:
            delta_m_star = probe_results["fit"]["delta_m_star"]
        run_distance_mode(cfg, delta_m_star, output_dir)

    if probe_results is None:
        probe_results = {"fit": {"delta_m_star": 0.0}}
    write_report(cfg, probe_results, output_dir)
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
