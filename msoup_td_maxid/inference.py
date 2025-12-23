from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from msoup_purist_closure.model import h_inferred_from_kernel

from .config import MaxIDConfig
from .diagnostics import PPCEntry, dominance_kill_table, info_fractions, logsumexp, loo_diagnostics, weighted_median
from .io import LensPosterior, load_td_master
from .likelihoods import histogram_logpdf_from_samples, robust_residual_loglike
from .model import D_dt_pred, f0_from_delta_m
from .resources import MemoryLimitExceeded, ResourceMonitor

SUPPORTED_MODES = {"base", "hier", "contam"}


def _delta_grid(cfg: MaxIDConfig) -> np.ndarray:
    pts = max(25, min(2000, int(cfg.priors.delta_m_points)))
    return np.linspace(cfg.priors.delta_m_min, cfg.priors.delta_m_max, pts)


def _predict_observable(lens: LensPosterior, delta_m: float, cfg: MaxIDConfig) -> float:
    if lens.support.lower().startswith("h0"):
        f0 = float(f0_from_delta_m(np.array([delta_m]), cfg.omega_m0, cfg.omega_L0)[0])
        return h_inferred_from_kernel(f0, h_early=cfg.h_early)
    return D_dt_pred(lens.z_lens, lens.z_source, delta_m, cfg.h_early, cfg.omega_m0, cfg.omega_L0)


def _base_loglike(lens: LensPosterior, delta_m: float, cfg: MaxIDConfig, sigma_sys: float = 0.0) -> float:
    pred = _predict_observable(lens, delta_m, cfg)
    if not np.isfinite(pred):
        return -np.inf
    sigma_floor = cfg.priors.sigma_floor_frac * abs(pred)

    if lens.has_samples:
        return float(histogram_logpdf_from_samples(lens.samples, np.array([pred]), bandwidth=cfg.compute.bandwidth)[0])

    sigma = lens.sigma if lens.sigma is not None else lens.sample_std
    if sigma is None or sigma <= 0:
        return -np.inf
    sigma_tot = np.sqrt(sigma ** 2 + sigma_sys ** 2 + sigma_floor ** 2)
    resid = (lens.value if lens.value is not None else pred) - pred
    return float(robust_residual_loglike(np.array([resid]), np.array([sigma_tot]), df=cfg.priors.student_df)[0])


def _half_normal_logpdf(x: np.ndarray, scale: float) -> np.ndarray:
    scale = max(scale, 1e-8)
    coeff = np.log(np.sqrt(2 / np.pi)) - np.log(scale)
    return coeff - 0.5 * (x / scale) ** 2


def _hier_loglike(lens: LensPosterior, delta_m: float, cfg: MaxIDConfig) -> float:
    tau0 = cfg.priors.tau0
    sigma_grid = np.linspace(0.0, max(3 * tau0, tau0 + 1e-6), 12)
    log_prior = _half_normal_logpdf(sigma_grid, tau0)
    terms = []
    for sg, lp in zip(sigma_grid, log_prior):
        terms.append(lp + _base_loglike(lens, delta_m, cfg, sigma_sys=sg))
    return float(logsumexp(np.array(terms)) - logsumexp(log_prior))


def _contam_loglike(lens: LensPosterior, delta_m: float, cfg: MaxIDConfig) -> float:
    base = _base_loglike(lens, delta_m, cfg, sigma_sys=0.0)
    if not np.isfinite(base):
        return base
    pred = _predict_observable(lens, delta_m, cfg)
    sigma = lens.sigma if lens.sigma is not None else lens.sample_std
    if sigma is None or sigma <= 0:
        return base
    sigma_out = sigma * cfg.priors.mixture_scale
    resid = (lens.value if lens.value is not None else pred) - pred
    log_in = base
    log_out = float(robust_residual_loglike(np.array([resid]), np.array([sigma_out]), df=cfg.priors.student_df)[0])
    eps = np.clip(cfg.priors.mixture_eps, 1e-4, 0.5)
    # logsumexp for mixture
    comp = np.array([np.log(1 - eps) + log_in, np.log(eps) + log_out])
    return float(logsumexp(comp))


def _lens_loglikes(lens: LensPosterior, delta_grid: np.ndarray, cfg: MaxIDConfig, mode: str) -> np.ndarray:
    vals = []
    for dm in delta_grid:
        if mode == "hier":
            vals.append(_hier_loglike(lens, dm, cfg))
        elif mode == "contam":
            vals.append(_contam_loglike(lens, dm, cfg))
        else:
            vals.append(_base_loglike(lens, dm, cfg))
    return np.asarray(vals, dtype=float)


def _posterior_summary(grid: np.ndarray, weights: np.ndarray) -> Dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    weights = np.maximum(weights, 0)
    if weights.sum() <= 0:
        return {"mean": np.nan, "median": np.nan, "low68": np.nan, "high68": np.nan, "sigma": np.nan}
    w = weights / weights.sum()
    mean = float(np.sum(grid * w))
    median = weighted_median(grid, w)
    cdf = np.cumsum(w[np.argsort(grid)])
    grid_sorted = np.sort(grid)
    low_idx = np.searchsorted(cdf, 0.16)
    high_idx = np.searchsorted(cdf, 0.84)
    low68 = float(grid_sorted[min(low_idx, len(grid_sorted) - 1)])
    high68 = float(grid_sorted[min(high_idx, len(grid_sorted) - 1)])
    sigma = 0.5 * (high68 - low68)
    return {"mean": mean, "median": median, "low68": low68, "high68": high68, "sigma": sigma}


def _ppc_entries(lenses: List[LensPosterior], delta_map: float, cfg: MaxIDConfig) -> List[PPCEntry]:
    entries: List[PPCEntry] = []
    for lens in lenses:
        pred = _predict_observable(lens, delta_map, cfg)
        sigma = lens.sigma if lens.sigma is not None else lens.sample_std
        sigma = sigma if sigma is not None else np.nan
        if sigma is None or sigma <= 0:
            sigma = np.nan
        resid = (lens.value if lens.value is not None else pred) - pred
        pull = float(resid / sigma) if sigma and np.isfinite(sigma) else np.nan
        entries.append(PPCEntry(lens_id=lens.lens_id, residual=float(resid), sigma=float(sigma) if np.isfinite(sigma) else np.nan, pull=pull, support=lens.support))
    return entries


def run_inference(cfg: MaxIDConfig, mode: str = "base") -> Dict[str, object]:
    if mode not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported mode {mode}")
    lenses = load_td_master(cfg)
    monitor = ResourceMonitor(cfg.compute.max_rss_gb, cfg.compute.rss_check_interval)

    results: Dict[str, object] = {}
    output_dir = Path(cfg.paths.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        delta_grid = _delta_grid(cfg)
        loglike_matrix = np.zeros((len(lenses), delta_grid.size))
        for i, lens in enumerate(lenses):
            if monitor.should_check(i):
                monitor.check(context=f"lens_{lens.lens_id}")
            loglike_matrix[i] = _lens_loglikes(lens, delta_grid, cfg, mode)

        total_loglike = np.sum(loglike_matrix, axis=0)
        log_norm = logsumexp(total_loglike)
        weights = np.exp(total_loglike - log_norm)
        weights = np.where(np.isfinite(weights), weights, 0.0)
        if weights.sum() == 0:
            weights = np.ones_like(weights) / weights.size
        weights /= weights.sum()

        summary = _posterior_summary(delta_grid, weights)
        f0_grid = f0_from_delta_m(delta_grid, cfg.omega_m0, cfg.omega_L0)
        f0_summary = _posterior_summary(f0_grid, weights)

        if len(lenses) > 0:
            info_frac = info_fractions(loglike_matrix)
            sigma_strength = np.array(
                [1.0 / max((lens.sigma if lens.sigma is not None else lens.sample_std or 1.0), 1e-6) ** 2 for lens in lenses]
            )
            sigma_strength = sigma_strength / sigma_strength.sum()
            info_frac = [float(0.5 * (a + b)) for a, b in zip(info_frac, sigma_strength)]
        else:
            info_frac = []
        loo = loo_diagnostics([l.lens_id for l in lenses], delta_grid, loglike_matrix, weights) if len(lenses) > 0 else []
        delta_map = float(delta_grid[np.argmax(weights)])
        ppc = _ppc_entries(lenses, delta_map, cfg)
        kill = dominance_kill_table(info_frac, loo, ppc, summary.get("sigma", np.nan))
        verdict = "ROBUST" if all(kill.values()) else "DOMINATED"

        results = {
            "mode": mode,
            "delta_m_grid": delta_grid.tolist(),
            "weights": weights.tolist(),
            "summary": summary,
            "f0_summary": f0_summary,
            "info_fraction": [{"lens_id": l.lens_id, "fraction": float(f)} for l, f in zip(lenses, info_frac)],
            "loo": [asdict(entry) for entry in loo],
            "ppc": [asdict(entry) for entry in ppc],
            "kill_table": kill,
            "verdict": verdict,
            "resource": monitor.snapshot(),
        }
    except MemoryLimitExceeded as exc:
        results["status"] = "ABORTED"
        results["reason"] = str(exc)
        results["resource"] = monitor.snapshot()
    return results
