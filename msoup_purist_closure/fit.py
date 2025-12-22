from __future__ import annotations

import json
from dataclasses import asdict
from typing import Dict, Tuple

import numpy as np
from scipy.optimize import minimize_scalar

from .config import MsoupConfig, ProbeConfig
from .kernels import kernel_weights, load_probe_csv
from .model import h_inferred_from_kernel, leak_fraction


def _f_eff_for_probe(df, probe: ProbeConfig, delta_m: float, omega_m0: float, omega_L0: float) -> Tuple[float, float, float]:
    kw = kernel_weights(df, probe.kernel, z_column=probe.z_column, sigma_column=probe.sigma_column)
    f_vals = leak_fraction(kw.z, delta_m=delta_m, omega_m0=omega_m0, omega_L0=omega_L0)
    f_eff = float(np.sum(kw.weights * f_vals))
    z_min, z_max = float(kw.z.min()), float(kw.z.max())
    z_below = {0.1: float(np.sum(kw.weights[kw.z <= 0.1])), 0.2: float(np.sum(kw.weights[kw.z <= 0.2])), 0.5: float(np.sum(kw.weights[kw.z <= 0.5]))}
    return f_eff, z_min, z_max, kw.effective_redshift, z_below


def fit_delta_m(cfg: MsoupConfig) -> Tuple[float, Dict[str, dict]]:
    """Fit Delta_m using the local probe."""
    local_probe_name = cfg.fit.local_probe or next((p.name for p in cfg.probes if p.is_local), cfg.probes[0].name)
    probe_map: Dict[str, ProbeConfig] = {p.name: p for p in cfg.probes}
    if local_probe_name not in probe_map:
        raise ValueError(f"Local probe {local_probe_name} not found in probes")
    local_probe = probe_map[local_probe_name]

    probe_dataframes = {p.name: load_probe_csv(p.path, z_column=p.z_column, sigma_column=p.sigma_column) for p in cfg.probes}

    def neg_log_likelihood(delta_m: float) -> float:
        df = probe_dataframes[local_probe.name]
        f_eff, *_ = _f_eff_for_probe(df, local_probe, delta_m, cfg.omega_m0, cfg.omega_L0)
        h_pred = h_inferred_from_kernel(f_eff, cfg.h_early)
        return (h_pred - cfg.fit.h_local) ** 2 / (2.0 * cfg.fit.sigma_local ** 2)

    bounds = cfg.fit.delta_m_bounds
    result = minimize_scalar(neg_log_likelihood, bounds=bounds, method="bounded", options={"xatol": 1e-3})
    if not result.success:
        raise RuntimeError(f"Delta_m fit failed: {result.message}")
    delta_m_star = float(result.x)

    probe_results = {}
    for name, df in probe_dataframes.items():
        probe_cfg = probe_map[name]
        f_eff, z_min, z_max, z_eff, z_below = _f_eff_for_probe(df, probe_cfg, delta_m_star, cfg.omega_m0, cfg.omega_L0)
        h_pred = h_inferred_from_kernel(f_eff, cfg.h_early)
        probe_results[name] = {
            "f_eff": f_eff,
            "H_inf": h_pred,
            "N": int(len(df)),
            "z_min": z_min,
            "z_max": z_max,
            "z_eff": z_eff,
            "z_below": z_below,
            "kernel": probe_cfg.kernel,
        }

    probe_results["fit"] = {"delta_m_star": delta_m_star, "neg_logL": float(result.fun), "bounds": bounds}
    return delta_m_star, probe_results


def serialize_fit_results(cfg: MsoupConfig, probe_results: Dict[str, dict], output_path: str):
    to_save = {
        "config": asdict(cfg),
        "probes": probe_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(to_save, f, indent=2)
