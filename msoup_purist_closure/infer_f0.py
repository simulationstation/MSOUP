from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from .config import MsoupConfig
from .data_utils import compute_ddt_prediction, load_bao_observed_data, load_sn_observed_data, load_td_observed_data
from .model import leak_fraction
from .observables import bao_predict, distance_modulus
from .plots import plot_bao_fit_multi
from .residuals import compute_bao_residuals, compute_sn_residuals, compute_td_residuals


@dataclass
class ProfileResult:
    delta_m_grid: np.ndarray
    chi2: np.ndarray
    delta_m_hat: float
    sigma_delta_m: float
    f0_hat: float
    sigma_f0: float
    status: str
    reason: Optional[str] = None


def memory_guard_ok(max_rss_mb: int) -> bool:
    try:
        import psutil

        rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
        return rss_mb < max_rss_mb
    except Exception:
        return True


def _guard_delta_m_grid(cfg: MsoupConfig) -> np.ndarray:
    delta_m_min = cfg.f0.delta_m_min
    delta_m_max = cfg.f0.delta_m_max
    num_points = int(cfg.f0.num_points)
    num_points = max(10, min(2000, num_points))
    return np.linspace(delta_m_min, delta_m_max, num_points)


def _f0_from_delta_m(delta_m: np.ndarray, omega_m0: float, omega_L0: float) -> np.ndarray:
    z0 = np.array([0.0])
    dm_arr = np.asarray(delta_m, dtype=float).reshape(-1)
    f0_vals = [leak_fraction(z0, float(dm), omega_m0, omega_L0)[0] for dm in dm_arr]
    return np.array(f0_vals, dtype=float)


def _profile_sn(cfg: MsoupConfig, delta_m_grid: np.ndarray) -> ProfileResult:
    probe = next((p for p in cfg.probes if p.type.lower() == "sn"), None)
    if probe is None:
        return ProfileResult(delta_m_grid, np.full_like(delta_m_grid, np.nan), np.nan, np.nan, np.nan, np.nan, "NOT_TESTABLE", "no SN probe configured")

    df, cov, status = load_sn_observed_data(probe)
    if df is None:
        return ProfileResult(delta_m_grid, np.full_like(delta_m_grid, np.nan), np.nan, np.nan, np.nan, np.nan, "NOT_TESTABLE", status)

    z_obs = df["z"].values
    mu_obs = df["mu_obs"].values
    sigma = df["sigma"].values

    chi2_vals = []
    f0_vals = []
    for dm in delta_m_grid:
        mu_pred = distance_modulus(z_obs, delta_m=dm, h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0, nuisance_M=cfg.distance.nuisance_M)
        res = compute_sn_residuals(z_obs, mu_obs, mu_pred, sigma, C=cov, marginalize_offset=True)
        chi2_vals.append(res.chi2)
        f0_vals.append(_f0_from_delta_m(np.array([dm]), cfg.omega_m0, cfg.omega_L0)[0])

    return _summarize_profile(delta_m_grid, np.array(chi2_vals), np.array(f0_vals))


def _profile_bao(cfg: MsoupConfig, delta_m_grid: np.ndarray, output_dir: pathlib.Path | None = None) -> ProfileResult:
    probe = next((p for p in cfg.probes if p.type.lower() == "bao"), None)
    if probe is None:
        return ProfileResult(delta_m_grid, np.full_like(delta_m_grid, np.nan), np.nan, np.nan, np.nan, np.nan, "NOT_TESTABLE", "no BAO probe configured")

    df, status = load_bao_observed_data(probe)
    if df is None:
        return ProfileResult(delta_m_grid, np.full_like(delta_m_grid, np.nan), np.nan, np.nan, np.nan, np.nan, "NOT_TESTABLE", status)

    rd = cfg.distance.rd_mpc
    cosmo_kwargs = {
        "h_early": cfg.h_early,
        "omega_m0": cfg.omega_m0,
        "omega_L0": cfg.omega_L0,
        "c_km_s": cfg.distance.speed_of_light,
    }

    rows = list(df.itertuples(index=False))
    z_obs_list = [r.z for r in rows]
    obs_values_list = [r.value for r in rows]
    sigma_list = [r.sigma for r in rows]
    obs_types_list = [r.observable for r in rows]

    chi2_vals = []
    f0_vals = []
    for dm in delta_m_grid:
        pred_values = []
        for row in rows:
            pred = bao_predict(
                row.z,
                row.observable,
                dm,
                rd,
                rd_fid_mpc=getattr(row, "rd_fid_mpc", None),
                rd_scaling=getattr(row, "rd_scaling", None),
                rd_fid_conventions={},
                paper_tag=getattr(row, "paper_tag", None),
                **cosmo_kwargs,
            )
            pred_values.append(pred)

        res = compute_bao_residuals(np.array(z_obs_list), np.array(obs_values_list), np.array(pred_values), np.array(sigma_list), "BAO")
        chi2_vals.append(res.chi2)
        f0_vals.append(_f0_from_delta_m(np.array([dm]), cfg.omega_m0, cfg.omega_L0)[0])

    # Optional diagnostic plot to reuse existing helper
    if output_dir is not None:
        plot_bao_fit_multi(
            np.linspace(cfg.f0.delta_m_min, cfg.f0.delta_m_max, 100),
            float(delta_m_grid[np.nanargmin(chi2_vals)]),
            rd,
            cfg,
            z_obs_list,
            obs_values_list,
            sigma_list,
            obs_types_list,
            probe.name,
            output_dir,
        )

    return _summarize_profile(delta_m_grid, np.array(chi2_vals), np.array(f0_vals))


def _profile_td(cfg: MsoupConfig, delta_m_grid: np.ndarray) -> ProfileResult:
    probe = next((p for p in cfg.probes if p.type.lower() == "td"), None)
    if probe is None:
        return ProfileResult(delta_m_grid, np.full_like(delta_m_grid, np.nan), np.nan, np.nan, np.nan, np.nan, "NOT_TESTABLE", "no TD probe configured")

    df, status = load_td_observed_data(probe)
    if df is None:
        return ProfileResult(delta_m_grid, np.full_like(delta_m_grid, np.nan), np.nan, np.nan, np.nan, np.nan, "NOT_TESTABLE", status)

    z_lens = df["z_lens"].values
    z_source = df["z_source"].values
    ddt_obs = df["D_dt"].values
    sigma = df["sigma"].values

    chi2_vals = []
    f0_vals = []
    for dm in delta_m_grid:
        ddt_pred = np.array([
            compute_ddt_prediction(zl, zs, dm, cfg) for zl, zs in zip(z_lens, z_source)
        ])
        res = compute_td_residuals(z_lens, ddt_obs, ddt_pred, sigma)
        chi2_vals.append(res.chi2)
        f0_vals.append(_f0_from_delta_m(np.array([dm]), cfg.omega_m0, cfg.omega_L0)[0])

    return _summarize_profile(delta_m_grid, np.array(chi2_vals), np.array(f0_vals))


def _summarize_profile(delta_m_grid: np.ndarray, chi2_vals: np.ndarray, f0_vals: np.ndarray) -> ProfileResult:
    if not np.any(np.isfinite(chi2_vals)):
        return ProfileResult(delta_m_grid, chi2_vals, np.nan, np.nan, np.nan, np.nan, "NOT_TESTABLE", "no finite chi2")

    chi2_vals = np.asarray(chi2_vals)
    idx_min = int(np.nanargmin(chi2_vals))
    delta_m_hat = float(delta_m_grid[idx_min])

    # Quadratic fit around minimum for curvature
    window = slice(max(0, idx_min - 3), min(len(delta_m_grid), idx_min + 4))
    x = delta_m_grid[window]
    y = chi2_vals[window]
    if len(x) < 3 or np.any(~np.isfinite(y)):
        return ProfileResult(delta_m_grid, chi2_vals, delta_m_hat, np.nan, f0_vals[idx_min], np.nan, "ERROR", "insufficient points for curvature")

    coeffs = np.polyfit(x, y, 2)
    a = coeffs[0]
    if a <= 0:
        sigma_delta_m = np.nan
    else:
        sigma_delta_m = math.sqrt(1 / (2 * a))

    f0_hat = float(f0_vals[idx_min])
    # derivative of f0 w.r.t delta_m at minimum via finite differences
    if len(delta_m_grid) < 2 or not np.isfinite(sigma_delta_m):
        sigma_f0 = np.nan
    else:
        if idx_min == 0:
            df_ddm = float((f0_vals[1] - f0_vals[0]) / (delta_m_grid[1] - delta_m_grid[0]))
        elif idx_min == len(delta_m_grid) - 1:
            df_ddm = float((f0_vals[-1] - f0_vals[-2]) / (delta_m_grid[-1] - delta_m_grid[-2]))
        else:
            dm_left = delta_m_grid[idx_min - 1]
            dm_right = delta_m_grid[idx_min + 1]
            f_left = f0_vals[idx_min - 1]
            f_right = f0_vals[idx_min + 1]
            df_ddm = float((f_right - f_left) / (dm_right - dm_left))
        sigma_f0 = abs(df_ddm) * sigma_delta_m

    return ProfileResult(delta_m_grid, chi2_vals, delta_m_hat, sigma_delta_m, f0_hat, sigma_f0, "OK")


def _combine_inverse_variance(estimates: Dict[str, Tuple[float, float]], dominance_threshold: float) -> Tuple[float, float, Dict[str, float]]:
    weights = {}
    for name, (mu, sigma) in estimates.items():
        if not (np.isfinite(mu) and np.isfinite(sigma) and sigma > 0):
            continue
        weights[name] = 1.0 / (sigma ** 2)

    if not weights:
        return (np.nan, np.nan), (np.nan, np.nan), {}

    total_w = sum(weights.values())
    dominant = {k: v / total_w for k, v in weights.items()}
    mu = sum(weights[k] * estimates[k][0] for k in weights) / total_w
    sigma = math.sqrt(1.0 / total_w)

    filtered_weights = {k: w for k, w in weights.items() if dominant[k] <= dominance_threshold}
    if filtered_weights:
        total_w_filtered = sum(filtered_weights.values())
        mu_filtered = sum(filtered_weights[k] * estimates[k][0] for k in filtered_weights) / total_w_filtered
        sigma_filtered = math.sqrt(1.0 / total_w_filtered)
    else:
        mu_filtered = mu
        sigma_filtered = sigma

    return (mu, sigma), (mu_filtered, sigma_filtered), dominant


def _z_score_consistency(f0_hat: float, sigma_f0: float, fb_mean: float, fb_sigma: float) -> Tuple[float, float]:
    if not all(np.isfinite([f0_hat, sigma_f0, fb_mean, fb_sigma])) or sigma_f0 <= 0 or fb_sigma <= 0:
        return np.nan, np.nan
    z = (f0_hat - fb_mean) / math.sqrt(sigma_f0 ** 2 + fb_sigma ** 2)
    p = 2 * norm.sf(abs(z))
    return float(z), float(p)


def _plot_profiles(delta_m_grid: np.ndarray, results: Dict[str, ProfileResult], output_dir: pathlib.Path) -> None:
    fig, ax = plt.subplots()
    for name, res in results.items():
        if res.status != "OK":
            continue
        mask = np.isfinite(res.chi2)
        ax.plot(delta_m_grid[mask], np.array(res.chi2)[mask], label=name)
    ax.set_xlabel("delta_m")
    ax.set_ylabel("chi2")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "profile_scan.png", dpi=150)
    plt.close(fig)


def _plot_f0_vs_delta_m(delta_m_grid: np.ndarray, omega_m0: float, omega_L0: float, output_dir: pathlib.Path) -> None:
    f0_vals = _f0_from_delta_m(delta_m_grid, omega_m0, omega_L0)
    fig, ax = plt.subplots()
    ax.plot(delta_m_grid, f0_vals, color="black")
    ax.set_xlabel("delta_m")
    ax.set_ylabel("f0(z=0)")
    fig.tight_layout()
    fig.savefig(output_dir / "f0_vs_delta_m.png", dpi=150)
    plt.close(fig)


def _plot_f0_posteriors(summaries: Dict[str, dict], fb_mean: float, fb_sigma: float, output_dir: pathlib.Path) -> None:
    xs = []
    for info in summaries.values():
        if info.get("sigma_f0") and np.isfinite(info["sigma_f0"]) and np.isfinite(info["f0_hat"]):
            xs.extend([info["f0_hat"] - 5 * info["sigma_f0"], info["f0_hat"] + 5 * info["sigma_f0"]])
    if not xs:
        return
    x_grid = np.linspace(min(xs), max(xs), 400)
    fig, ax = plt.subplots()
    for name, info in summaries.items():
        mu, sig = info.get("f0_hat"), info.get("sigma_f0")
        if not (np.isfinite(mu) and np.isfinite(sig) and sig > 0):
            continue
        pdf = norm.pdf(x_grid, mu, sig)
        ax.plot(x_grid, pdf, label=name)
    if np.isfinite(fb_mean) and np.isfinite(fb_sigma) and fb_sigma > 0:
        ax.axvspan(fb_mean - fb_sigma, fb_mean + fb_sigma, color="gray", alpha=0.2, label="fb ±1σ")
        ax.axvline(fb_mean, color="gray", linestyle="--")
    ax.set_xlabel("f0")
    ax.set_ylabel("Gaussian approx")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "f0_posteriors.png", dpi=150)
    plt.close(fig)


def run_inference(cfg: MsoupConfig, output_dir: pathlib.Path) -> Dict:
    delta_m_grid = _guard_delta_m_grid(cfg)

    if not memory_guard_ok(cfg.f0.max_rss_mb):
        raise MemoryError("RSS above configured guard before starting inference")

    subsets: Dict[str, ProfileResult] = {}

    def maybe_store(name: str, fn):
        if not memory_guard_ok(cfg.f0.max_rss_mb):
            subsets[name] = ProfileResult(delta_m_grid, np.full_like(delta_m_grid, np.nan), np.nan, np.nan, np.nan, np.nan, "ABORTED_OOM_GUARD")
            return
        subsets[name] = fn()

    if cfg.f0.include_sn:
        maybe_store("SN", lambda: _profile_sn(cfg, delta_m_grid))
    if cfg.f0.include_bao:
        maybe_store("BAO", lambda: _profile_bao(cfg, delta_m_grid, output_dir))
    if cfg.f0.include_td:
        maybe_store("TD", lambda: _profile_td(cfg, delta_m_grid))

    # Combined subsets
    combined_results = {}
    combos = {
        "SN+BAO": ["SN", "BAO"],
        "SN+BAO+TD": ["SN", "BAO", "TD"],
    }
    for name, parts in combos.items():
        valid = [subsets[p] for p in parts if p in subsets and subsets[p].status == "OK"]
        if not valid:
            combined_results[name] = ProfileResult(delta_m_grid, np.full_like(delta_m_grid, np.nan), np.nan, np.nan, np.nan, np.nan, "NOT_TESTABLE", "missing components")
            continue
        chi2 = sum(v.chi2 for v in valid)
        f0_vals = _f0_from_delta_m(delta_m_grid, cfg.omega_m0, cfg.omega_L0)
        combined_results[name] = _summarize_profile(delta_m_grid, chi2, f0_vals)

    # Summaries
    summaries = {}
    fb_ref_path = cfg.f0.fb_reference
    fb_mean = np.nan
    fb_sigma = np.nan
    if fb_ref_path:
        ref_df = pd.read_csv(fb_ref_path)
        fb_mean = float(ref_df.iloc[0].get("fb_mean", ref_df.iloc[0].get("mean")))
        fb_sigma = float(ref_df.iloc[0].get("fb_sigma", ref_df.iloc[0].get("sigma")))

    dominance_threshold = cfg.f0.dominance_threshold
    estimates = {}
    for name, res in {**subsets, **combined_results}.items():
        if res.status != "OK":
            continue
        estimates[name] = (res.f0_hat, res.sigma_f0)
        z, p = _z_score_consistency(res.f0_hat, res.sigma_f0, fb_mean, fb_sigma) if fb_ref_path else (np.nan, np.nan)
        summaries[name] = {
            "delta_m_hat": res.delta_m_hat,
            "sigma_delta_m": res.sigma_delta_m,
            "f0_hat": res.f0_hat,
            "sigma_f0": res.sigma_f0,
            "z_to_fb": z,
            "p_to_fb": p,
            "status": res.status,
        }

    base_estimates = {k: v for k, v in estimates.items() if k in {"SN", "BAO", "TD"}}
    combined_est, filtered_est, dominance = _combine_inverse_variance(base_estimates, dominance_threshold)

    leave_one_out = {}
    for probe in ["SN", "BAO", "TD"]:
        reduced = {k: v for k, v in base_estimates.items() if k != probe}
        if len(reduced) >= 1:
            leave_one_out[probe] = _combine_inverse_variance(reduced, dominance_threshold)[0]

    summary_rows = []
    for name, res in {**subsets, **combined_results}.items():
        summary_rows.append({
            "subset": name,
            "status": res.status,
            "delta_m_hat": res.delta_m_hat,
            "sigma_delta_m": res.sigma_delta_m,
            "f0_hat": res.f0_hat,
            "sigma_f0": res.sigma_f0,
            "dominance": dominance.get(name),
            "z_to_fb": summaries.get(name, {}).get("z_to_fb", np.nan),
            "p_to_fb": summaries.get(name, {}).get("p_to_fb", np.nan),
        })

    summary_payload = {
        "subsets": summary_rows,
        "combined_inverse_variance": {
            "all": {"f0_hat": combined_est[0], "sigma_f0": combined_est[1]},
            "non_dominant": {"f0_hat": filtered_est[0], "sigma_f0": filtered_est[1]},
            "leave_one_out": {k: {"f0_hat": v[0], "sigma_f0": v[1]} for k, v in leave_one_out.items()},
            "dominance": dominance,
            "dominance_threshold": dominance_threshold,
        },
        "fb_reference": {"path": fb_ref_path, "mean": fb_mean, "sigma": fb_sigma},
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    profile_rows = []
    for name, res in {**subsets, **combined_results}.items():
        for dm, chi2 in zip(res.delta_m_grid, res.chi2):
            profile_rows.append({"subset": name, "delta_m": dm, "chi2": chi2})
    profile_df = pd.DataFrame(profile_rows)
    profile_df.to_csv(output_dir / "delta_m_profiles.csv", index=False)

    f0_post_rows = []
    for name, res in {**subsets, **combined_results}.items():
        f0_post_rows.append({
            "subset": name,
            "f0_hat": res.f0_hat,
            "sigma_f0": res.sigma_f0,
            "status": res.status,
        })
    pd.DataFrame(f0_post_rows).to_csv(output_dir / "f0_posterior_summary.csv", index=False)

    plot_summary = {row["subset"]: row for row in summary_rows if row["status"] == "OK" and np.isfinite(row["f0_hat"]) and np.isfinite(row["sigma_f0"])}
    all_results = {**subsets, **combined_results}
    _plot_profiles(delta_m_grid, all_results, output_dir)
    _plot_f0_vs_delta_m(delta_m_grid, cfg.omega_m0, cfg.omega_L0, output_dir)
    _plot_f0_posteriors(plot_summary, fb_mean, fb_sigma, output_dir)

    report_lines = [
        "# f0 inference summary",
        "",
        "This module profiles delta_m over a configured grid and maps it to f0(z=0) without imposing f0=fb.",
        "Inconsistency (falsification) would require mutually inconsistent probe posteriors or |f0 - fb| > 3 sigma across multiple probes.",
        "",
    ]

    if fb_ref_path:
        report_lines.append(f"fb reference: {fb_ref_path}, mean={fb_mean:.4g}, sigma={fb_sigma:.4g}")

    report_lines.append("\n## Subset posteriors")
    for name, res in {**subsets, **combined_results}.items():
        line = f"- {name}: f0 = {res.f0_hat:.4g} ± {res.sigma_f0:.4g} (delta_m = {res.delta_m_hat:.4g} ± {res.sigma_delta_m:.4g}) status={res.status}"
        if fb_ref_path and np.isfinite(summaries.get(name, {}).get("z_to_fb", np.nan)):
            z = summaries[name]["z_to_fb"]
            p = summaries[name]["p_to_fb"]
            line += f"; Z vs fb = {z:.3g} (p={p:.3g})"
        report_lines.append(line)

    if combined_est != (np.nan, np.nan):
        report_lines.append("\n## Combined estimates (inverse-variance)")
        report_lines.append(f"- All included: f0 = {combined_est[0]:.4g} ± {combined_est[1]:.4g}")
        report_lines.append(f"- Non-dominant (<= {dominance_threshold*100:.0f}% each): f0 = {filtered_est[0]:.4g} ± {filtered_est[1]:.4g}")
        for name, frac in dominance.items():
            report_lines.append(f"  - Weight {name}: {frac*100:.1f}%")
        if leave_one_out:
            report_lines.append("\n## Leave-one-probe-out combinations")
            for probe, est in leave_one_out.items():
                report_lines.append(f"- Drop {probe}: f0 = {est[0]:.4g} ± {est[1]:.4g}")

    report_path = output_dir / "REPORT.md"
    report_path.write_text("\n".join(report_lines))

    return {
        "profiles": subsets,
        "combined": combined_results,
        "summary": summaries,
        "combined_est": combined_est,
        "combined_filtered": filtered_est,
        "dominance": dominance,
        "leave_one_out": leave_one_out,
        "fb_reference": fb_ref_path,
        "fb_mean": fb_mean,
        "fb_sigma": fb_sigma,
    }


__all__ = ["run_inference"]
