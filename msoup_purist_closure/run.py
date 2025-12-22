from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MsoupConfig, ProbeConfig, load_config
from .distances import angular_diameter_distance, comoving_distance, h_eff_ratio, luminosity_distance
from .fit import fit_delta_m
from .kernels import kernel_weights, load_probe_csv
from .observables import bao_dv, bao_dm, bao_dh, bao_predict, distance_modulus, lens_time_delay_scaling
from .plots import plot_af, plot_distance_mode, plot_hinf_curves, plot_kernel_histograms, plot_bao_fit_multi
from .report import write_report
from .residuals import (
    ProbeResidualResult,
    compute_bao_residuals,
    compute_sn_residuals,
    compute_td_residuals,
    load_covariance,
)


# Memory guard threshold (MB)
MAX_RSS_MB = 6000


def get_rss_mb() -> float:
    """Get current RSS in MB using psutil if available."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def check_memory_guard() -> bool:
    """Check if memory usage exceeds guard threshold."""
    rss = get_rss_mb()
    return rss > MAX_RSS_MB


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


def load_sn_observed_data(probe: ProbeConfig) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], str]:
    """
    Load SN observed data: z, mu_obs (or m_b), sigma, and optionally covariance.
    Returns (DataFrame, Covariance or None, status_reason).
    """
    probe_path = pathlib.Path(probe.path)

    # Try to find the data file
    # Check if it points to pantheonplus_mu.csv or similar
    if not probe_path.exists():
        # Try to find in the probe's directory based on type
        return None, None, f"file not found: {probe.path}"

    try:
        df = pd.read_csv(probe_path)
    except Exception as e:
        return None, None, f"CSV read error: {e}"

    # Check for required columns
    z_col = probe.z_column or 'z'
    if z_col not in df.columns:
        return None, None, f"missing column: {z_col}"

    # Look for mu_obs or m_b
    mu_col = None
    for col in ['mu_obs', 'mu', 'm_b', 'mb', 'MU_SH0ES']:
        if col in df.columns:
            mu_col = col
            break
    if mu_col is None:
        return None, None, "missing observed column: mu_obs/mu/m_b"

    # Look for sigma column
    sigma_col = probe.sigma_column or 'sigma'
    if sigma_col not in df.columns:
        for col in ['mu_err_diag', 'sigma', 'err', 'uncertainty']:
            if col in df.columns:
                sigma_col = col
                break

    if sigma_col not in df.columns:
        return None, None, f"missing sigma column"

    # Standardize columns
    df = df.rename(columns={z_col: 'z', mu_col: 'mu_obs', sigma_col: 'sigma'})

    # Load covariance if available
    cov = None
    cov_path = probe_path.parent / (probe_path.stem.replace('_mu', '_cov_stat_sys') + '.npz')
    if cov_path.exists():
        cov = load_covariance(cov_path, len(df))

    return df, cov, "OK"


def load_bao_observed_data(probe: ProbeConfig) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load BAO observed data: z, observable type, value, sigma.
    Returns (DataFrame, status_reason).
    """
    probe_path = pathlib.Path(probe.path)
    if not probe_path.exists():
        return None, f"file not found: {probe.path}"

    try:
        df = pd.read_csv(probe_path)
    except Exception as e:
        return None, f"CSV read error: {e}"

    # Check for required columns
    required = ['z', 'value', 'sigma']
    for col in required:
        if col not in df.columns:
            return None, f"missing column: {col}"

    # Check for observable type column
    if 'observable' not in df.columns:
        return None, "missing column: observable"

    return df, "OK"


def load_td_observed_data(probe: ProbeConfig) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load time-delay observed data: z_lens, z_source, D_dt, sigma.
    Returns (DataFrame, status_reason).
    """
    probe_path = pathlib.Path(probe.path)
    if not probe_path.exists():
        return None, f"file not found: {probe.path}"

    try:
        df = pd.read_csv(probe_path)
    except Exception as e:
        return None, f"CSV read error: {e}"

    # Check for required columns
    z_lens_col = None
    for col in ['z_lens', 'z', 'zlens']:
        if col in df.columns:
            z_lens_col = col
            break
    if z_lens_col is None:
        return None, "missing column: z_lens"

    z_source_col = None
    for col in ['z_source', 'zsource', 'zs']:
        if col in df.columns:
            z_source_col = col
            break
    if z_source_col is None:
        return None, "missing column: z_source"

    ddt_col = None
    for col in ['D_dt', 'Ddt', 'D_dt_obs']:
        if col in df.columns:
            ddt_col = col
            break
    if ddt_col is None:
        return None, "missing column: D_dt"

    sigma_col = None
    for col in ['sigma_D_dt', 'sigma', 'sigma_Ddt']:
        if col in df.columns:
            sigma_col = col
            break
    if sigma_col is None:
        return None, "missing column: sigma_D_dt"

    # Standardize columns
    df = df.rename(columns={
        z_lens_col: 'z_lens',
        z_source_col: 'z_source',
        ddt_col: 'D_dt',
        sigma_col: 'sigma',
    })

    return df, "OK"


def compute_ddt_prediction(z_lens: float, z_source: float, delta_m: float, cfg: MsoupConfig) -> float:
    """
    Compute predicted D_dt = (1+z_l) * D_l * D_ls / D_s (time-delay distance).
    Uses angular diameter distances.
    """
    c_km_s = cfg.distance.speed_of_light

    # Get angular diameter distances
    d_l = angular_diameter_distance(
        np.array([z_lens]), delta_m=delta_m,
        h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0
    )[0]

    d_s = angular_diameter_distance(
        np.array([z_source]), delta_m=delta_m,
        h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0
    )[0]

    # D_ls requires special treatment for angular diameter distance between lens and source
    # D_ls = D_s / (1 + z_s) - D_l / (1 + z_l) for flat universe
    chi_l = comoving_distance(
        np.array([z_lens]), delta_m=delta_m,
        h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0
    )[0]
    chi_s = comoving_distance(
        np.array([z_source]), delta_m=delta_m,
        h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0
    )[0]
    d_ls = (chi_s - chi_l) / (1 + z_source)

    # D_dt = (1 + z_l) * D_l * D_s / D_ls
    # Units: distances are in Mpc (c/H0 units)
    if d_ls <= 0:
        return np.nan
    ddt = (1 + z_lens) * d_l * d_s / d_ls
    return ddt


def compute_bao_dv_rd(z: float, delta_m: float, cfg: MsoupConfig, rd: float = 147.09) -> float:
    """
    Compute DV/rd prediction.
    rd is the sound horizon at drag epoch in Mpc.
    """
    dv = bao_dv(
        np.array([z]), delta_m=delta_m,
        h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0
    )[0]
    return dv / rd


def run_distance_mode(
    cfg: MsoupConfig,
    delta_m_star: float,
    output_dir: pathlib.Path,
    probe_results: Optional[Dict[str, dict]] = None,
) -> Dict[str, dict]:
    """
    Run distance mode: compute residuals and chi-square for each probe.
    Returns distance_results dictionary.
    """
    start_time = time.time()
    start_rss = get_rss_mb()

    distance_results = {
        "global": {
            "delta_m_star": delta_m_star,
            "chi2_total": 0.0,
            "dof_total": 0,
            "chi2_dof_total": np.nan,
            "elapsed_s": 0.0,
            "peak_rss_mb": 0.0,
            "status": "OK",
        },
        "probes": {},
    }

    # Standard cosmology grid for plots
    z_max = cfg.distance.z_max
    z_grid = np.linspace(0.01, z_max, cfg.distance.num_z_samples)
    h_ratio = h_eff_ratio(z_grid, delta_m_star, cfg.h_early, cfg.omega_m0, cfg.omega_L0)

    mu_model = distance_modulus(
        z_grid, delta_m=delta_m_star, h_early=cfg.h_early,
        omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0,
        nuisance_M=cfg.distance.nuisance_M
    )
    mu_lcdm = distance_modulus(
        z_grid, delta_m=0.0, h_early=cfg.h_early,
        omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0,
        nuisance_M=cfg.distance.nuisance_M
    )
    delta_mu = mu_model - mu_lcdm
    dv_grid = bao_dv(
        z_grid, delta_m=delta_m_star, h_early=cfg.h_early,
        omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0
    )

    # Plot theoretical curves
    plot_distance_mode(z_grid, h_ratio, delta_mu, dv_grid, output_dir)

    # Import plotting for data overlays
    from .plots import plot_sn_fit, plot_bao_fit, plot_td_fit

    chi2_total = 0.0
    dof_total = 0
    peak_rss = start_rss

    for probe in cfg.probes:
        if check_memory_guard():
            distance_results["global"]["status"] = "ABORTED_OOM_GUARD"
            break

        current_rss = get_rss_mb()
        peak_rss = max(peak_rss, current_rss)

        probe_type = probe.type.lower()

        if probe_type == "sn":
            # Load SN data
            df, cov, status = load_sn_observed_data(probe)
            if df is None:
                result = ProbeResidualResult(
                    status="NOT_TESTABLE",
                    reason=status,
                )
                distance_results["probes"][probe.name] = asdict(result)
                continue

            # Compute predictions
            z_obs = df['z'].values
            mu_obs = df['mu_obs'].values
            sigma = df['sigma'].values

            mu_pred = distance_modulus(
                z_obs, delta_m=delta_m_star, h_early=cfg.h_early,
                omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0,
                nuisance_M=cfg.distance.nuisance_M
            )

            # Compute residuals with offset marginalization
            result = compute_sn_residuals(
                z_obs, mu_obs, mu_pred, sigma, C=cov, marginalize_offset=True
            )

            # Update totals
            if result.status == "OK" and np.isfinite(result.chi2):
                chi2_total += result.chi2
                dof_total += result.dof

            distance_results["probes"][probe.name] = asdict(result)

            # Plot with data points
            plot_sn_fit(
                z_grid, mu_model, z_obs, mu_obs, sigma,
                probe.name, output_dir, b_hat=result.b_hat
            )

        elif probe_type == "bao":
            # Load BAO data - requires 'value' column for chi-square computation
            df, status = load_bao_observed_data(probe)
            if df is None:
                result = ProbeResidualResult(
                    status="NOT_TESTABLE",
                    reason=status,
                )
                distance_results["probes"][probe.name] = asdict(result)
                continue

            # Use rd from config (fixed, not fitted)
            rd = cfg.distance.rd_mpc

            # Cosmology kwargs for bao_predict
            cosmo_kwargs = {
                "h_early": cfg.h_early,
                "omega_m0": cfg.omega_m0,
                "omega_L0": cfg.omega_L0,
                "c_km_s": cfg.distance.speed_of_light,
            }

            # Process ALL BAO rows using bao_predict (DV/rd, DM/rd, DH/rd)
            z_obs_list = []
            obs_values_list = []
            sigma_list = []
            pred_values_list = []
            obs_types_list = []
            n_dv, n_dm, n_dh = 0, 0, 0

            for _, row in df.iterrows():
                obs_type = row['observable']
                z = row['z']
                val = row['value']
                sig = row['sigma']

                try:
                    pred = bao_predict(z, obs_type, delta_m_star, rd, **cosmo_kwargs)
                except ValueError as e:
                    # Unknown observable type - skip with warning
                    print(f"Warning: {e}")
                    continue

                z_obs_list.append(z)
                obs_values_list.append(val)
                sigma_list.append(sig)
                pred_values_list.append(pred)
                obs_types_list.append(obs_type)

                # Count by type
                obs_upper = obs_type.upper()
                if 'DV' in obs_upper:
                    n_dv += 1
                elif 'DM' in obs_upper and 'DH' not in obs_upper:
                    n_dm += 1
                elif 'DH' in obs_upper:
                    n_dh += 1

            if len(z_obs_list) == 0:
                result = ProbeResidualResult(
                    status="NOT_TESTABLE",
                    reason="no supported observable (DV/rd, DM/rd, or DH/rd)",
                )
                distance_results["probes"][probe.name] = asdict(result)
                continue

            z_obs = np.array(z_obs_list)
            obs_values = np.array(obs_values_list)
            sigma = np.array(sigma_list)
            pred_values = np.array(pred_values_list)

            # Compute chi-square
            result = compute_bao_residuals(z_obs, obs_values, pred_values, sigma, "BAO")

            # Add observable breakdown to result
            obs_breakdown = f"DV/rd:{n_dv}, DM/rd:{n_dm}, DH/rd:{n_dh}"
            result.obs_column = obs_breakdown

            if result.status == "OK" and np.isfinite(result.chi2):
                chi2_total += result.chi2
                dof_total += result.dof

            distance_results["probes"][probe.name] = asdict(result)

            # Print BAO summary to console
            print(f"BAO {probe.name}: {len(z_obs_list)} rows ({obs_breakdown}), "
                  f"chi2={result.chi2:.2f}, dof={result.dof}, chi2/dof={result.chi2_dof:.3f}")

            # Plot with data points - grouped by observable type
            plot_bao_fit_multi(
                z_grid, delta_m_star, rd, cfg,
                z_obs_list, obs_values_list, sigma_list, obs_types_list,
                probe.name, output_dir
            )

        elif probe_type == "td":
            # Load TD data
            df, status = load_td_observed_data(probe)
            if df is None:
                result = ProbeResidualResult(
                    status="NOT_TESTABLE",
                    reason=status,
                )
                distance_results["probes"][probe.name] = asdict(result)
                continue

            z_lens = df['z_lens'].values
            z_source = df['z_source'].values
            ddt_obs = df['D_dt'].values
            sigma = df['sigma'].values

            # Compute predictions
            ddt_pred = np.array([
                compute_ddt_prediction(zl, zs, delta_m_star, cfg)
                for zl, zs in zip(z_lens, z_source)
            ])

            result = compute_td_residuals(z_lens, ddt_obs, ddt_pred, sigma)

            if result.status == "OK" and np.isfinite(result.chi2):
                chi2_total += result.chi2
                dof_total += result.dof

            distance_results["probes"][probe.name] = asdict(result)

            # Plot with data points
            plot_td_fit(
                z_lens, ddt_obs, ddt_pred, sigma,
                probe.name, output_dir
            )

        else:
            result = ProbeResidualResult(
                status="NOT_TESTABLE",
                reason=f"unknown probe type: {probe_type}",
            )
            distance_results["probes"][probe.name] = asdict(result)

    # Finalize
    elapsed = time.time() - start_time
    peak_rss = max(peak_rss, get_rss_mb())

    distance_results["global"]["chi2_total"] = chi2_total
    distance_results["global"]["dof_total"] = dof_total
    distance_results["global"]["chi2_dof_total"] = chi2_total / dof_total if dof_total > 0 else np.nan
    distance_results["global"]["elapsed_s"] = elapsed
    distance_results["global"]["peak_rss_mb"] = peak_rss

    return distance_results


def main(argv=None):
    args = parse_args(argv)
    cfg = load_config(args.config)
    output_dir = cfg.results_dir / cfg.timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_results = None
    distance_results = None

    if args.mode in {"kernel", "both"}:
        probe_results = run_kernel_mode(cfg, output_dir)

    if args.mode in {"distance", "both"}:
        if probe_results is None:
            delta_m_star = 0.0
        else:
            delta_m_star = probe_results["fit"]["delta_m_star"]
        distance_results = run_distance_mode(cfg, delta_m_star, output_dir, probe_results)

    if probe_results is None:
        probe_results = {"fit": {"delta_m_star": 0.0}}

    write_report(cfg, probe_results, output_dir, distance_results)
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
