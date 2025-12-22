from __future__ import annotations

import argparse
import pathlib
import sys
import time
from dataclasses import asdict
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .bao_benchmark import run_bao_benchmark
from .config import MsoupConfig, ProbeConfig, load_config
from .data_utils import compute_ddt_prediction, load_bao_observed_data, load_sn_observed_data, load_td_observed_data
from .distances import angular_diameter_distance, h_eff_ratio, luminosity_distance
from .fit import fit_delta_m
from .infer_f0 import run_inference
from .kernels import kernel_weights, load_probe_csv
from .observables import bao_dv, bao_dm, bao_dh, bao_predict, distance_modulus
from .bao_diagnostics import (
    run_bao_sanity_cases,
    write_bao_audit_csv,
    format_bao_sanity_markdown,
    format_bao_pulls_markdown,
    get_bao_sanity_summary_dict,
    compute_bao_pulls,
)
from .plots import plot_af, plot_distance_mode, plot_hinf_curves, plot_kernel_histograms, plot_bao_fit_multi
from .report import write_report
from .residuals import (
    ProbeResidualResult,
    compute_bao_residuals,
    compute_sn_residuals,
    compute_td_residuals,
)
from .td_inference import run_td_inference


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
    parser.add_argument("--config", required=False, help="Path to YAML config")
    parser.add_argument("--mode", choices=["kernel", "distance", "both", "infer-f0", "td_only"], default="kernel")
    parser.add_argument("--bao-sanity", action="store_true",
                        help="Run BAO sanity diagnostics: compare LCDM_BASELINE, MODEL_BEST, MODEL_WEAK")
    parser.add_argument("--bao-benchmark", action="store_true",
                        help="Run embedded BAO benchmark table without loading external data")
    parser.add_argument("--tdlmc", action="store_true", help="Run optional TDLMC bias calibration for TD-only mode")
    args = parser.parse_args(argv)
    if not args.bao_benchmark and args.config is None:
        parser.error("--config is required unless --bao-benchmark is specified")
    return args


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
            rd_fid_conventions: dict | None = {}

            for _, row in df.iterrows():
                obs_type = row['observable']
                z = row['z']
                val = row['value']
                sig = row['sigma']
                rd_fid_mpc = row.get('rd_fid_mpc')
                rd_scaling = row.get('rd_scaling')

                pred = bao_predict(
                    z,
                    obs_type,
                    delta_m_star,
                    rd,
                    rd_fid_mpc=rd_fid_mpc,
                    rd_scaling=rd_scaling,
                    rd_fid_conventions=rd_fid_conventions,
                    paper_tag=row.get('paper_tag'),
                    **cosmo_kwargs,
                )

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


def run_bao_sanity_diagnostics(
    cfg: MsoupConfig,
    delta_m_star: float,
    output_dir: pathlib.Path,
) -> Optional[Dict]:
    """
    Run BAO sanity diagnostics: compare LCDM_BASELINE, MODEL_BEST, MODEL_WEAK.

    Writes:
    - bao_audit.csv: per-row pulls for MODEL_BEST case
    - BAO sanity section in REPORT.md (via returned dict)

    Returns dict with sanity results for summary.json.
    """
    print("\n=== BAO Sanity Diagnostics ===")

    # Find BAO probe
    bao_probe = None
    for probe in cfg.probes:
        if probe.type == "bao":
            bao_probe = probe
            break

    if bao_probe is None:
        print("No BAO probe configured, skipping sanity diagnostics")
        return None

    # Load BAO data
    df, status = load_bao_observed_data(bao_probe)
    if df is None:
        print(f"Could not load BAO data: {status}")
        return None

    # Cosmology kwargs
    cosmo_kwargs = {
        "h_early": cfg.h_early,
        "omega_m0": cfg.omega_m0,
        "omega_L0": cfg.omega_L0,
        "c_km_s": cfg.distance.speed_of_light,
    }
    rd = cfg.distance.rd_mpc

    # Run three-case comparison
    print(f"Running sanity cases with delta_m_star={delta_m_star:.4f}, rd={rd:.2f} Mpc")
    results = run_bao_sanity_cases(df, delta_m_star, rd, cosmo_kwargs, rd_fid_conventions={})

    # Print summary to console
    print("\nBAO Sanity Summary:")
    print(f"{'Case':<16} {'delta_m':>10} {'chi2':>10} {'dof':>5} {'chi2/dof':>10}")
    print("-" * 55)
    for case_name in ["LCDM_BASELINE", "MODEL_BEST", "MODEL_WEAK"]:
        r = results[case_name]
        print(f"{case_name:<16} {r.delta_m:>10.4f} {r.total_chi2:>10.2f} {r.total_dof:>5} {r.chi2_dof:>10.3f}")

    # Print per-observable breakdown
    print("\nPer-observable chi2:")
    print(f"{'Case':<16} {'DV/rd':>15} {'DM/rd':>15} {'DH/rd':>15}")
    print("-" * 65)
    for case_name in ["LCDM_BASELINE", "MODEL_BEST", "MODEL_WEAK"]:
        r = results[case_name]
        dv = r.chi2_by_obs.get("DV/rd", (0, 0))
        dm = r.chi2_by_obs.get("DM/rd", (0, 0))
        dh = r.chi2_by_obs.get("DH/rd", (0, 0))
        print(f"{case_name:<16} {dv[0]:>7.2f} ({dv[1]:>2}) {dm[0]:>7.2f} ({dm[1]:>2}) {dh[0]:>7.2f} ({dh[1]:>2})")

    # Print top 5 worst pulls for LCDM_BASELINE
    print("\nTop 5 worst pulls (LCDM_BASELINE):")
    for wp in results["LCDM_BASELINE"].worst_pulls:
        print(f"  z={wp.z:.3f} {wp.observable}: value={wp.value:.3f}, pred={wp.pred:.3f}, pull={wp.pull:+.2f}")

    # Write audit CSV for MODEL_BEST
    audit_path = output_dir / "bao_audit.csv"
    write_bao_audit_csv(results["MODEL_BEST"].rows, audit_path)
    print(f"\nBAO audit CSV written to: {audit_path}")

    # Return summary dict for report
    return get_bao_sanity_summary_dict(results)


def main(argv=None):
    args = parse_args(argv)
    if args.bao_benchmark:
        results_root = pathlib.Path("results/msoup_purist_closure")
        if args.config:
            cfg_for_paths = load_config(args.config)
            results_root = cfg_for_paths.results_dir
        df, output_dir = run_bao_benchmark(results_root)
        print(df.to_string(index=False, float_format="%.4f"))
        print(f"BAO benchmark written to {output_dir}")
        return 0

    cfg = load_config(args.config)
    if args.mode == "infer-f0":
        output_dir = cfg.results_dir.parent / cfg.f0.output_subdir / cfg.timestamp
    else:
        output_dir = cfg.results_dir / cfg.timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_results = None
    distance_results = None
    bao_sanity_results = None

    if args.mode == "infer-f0":
        inference_results = run_inference(cfg, output_dir)
        print(f"f0 inference written to {output_dir}")
        return 0

    if args.mode == "td_only":
        td_probe = next((p for p in cfg.probes if p.type.lower() == "td"), None)
        if td_probe is None:
            raise RuntimeError("No TD probe configured for td_only mode.")
        td_results = run_td_inference(cfg, output_dir, td_probe, run_tdlmc=args.tdlmc)
        print(f"TD-only inference written to {output_dir}")
        summary = td_results.get("summary", {})
        print(f"delta_m mean={summary.get('mean')}, median={summary.get('median')}")
        return 0

    if args.mode in {"kernel", "both"}:
        probe_results = run_kernel_mode(cfg, output_dir)

    if args.mode in {"distance", "both"}:
        if probe_results is None:
            delta_m_star = 0.0
        else:
            delta_m_star = probe_results["fit"]["delta_m_star"]
        distance_results = run_distance_mode(cfg, delta_m_star, output_dir, probe_results)

        # Run BAO sanity diagnostics if requested
        bao_sanity = getattr(args, 'bao_sanity', False)
        if bao_sanity:
            bao_sanity_results = run_bao_sanity_diagnostics(
                cfg, delta_m_star, output_dir
            )

    if probe_results is None:
        probe_results = {"fit": {"delta_m_star": 0.0}}

    write_report(cfg, probe_results, output_dir, distance_results, bao_sanity_results)
    print(f"Results written to {output_dir}")


if __name__ == "__main__":
    sys.exit(main())
