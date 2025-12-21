"""Core simulation loop for the ejection + decompression toy model."""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any

from .config import SimulationConfig
from .density_field import evolve_sigma, lognormal_density
from .init_conditions import initialize_fields
from .gravity import GravitySolver
from .projection import compute_H_global


@dataclass
class SimulationResult:
    cfg: SimulationConfig
    history: Dict[str, np.ndarray]
    snapshots: Dict[str, Dict[str, np.ndarray]]
    final_fields: Dict[str, np.ndarray]


EPS = 1e-8


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    base_field, rho_b0, rho_dm3, A, wlt2, rho_mem = initialize_fields(cfg)
    grid = cfg.grid
    steps = cfg.steps
    dt = cfg.dt_internal
    solver = GravitySolver(grid)

    history = {k: [] for k in ["H_global", "X_V", "X_rho", "mean_dm3", "V_EM"]}
    snapshot_indices = {
        "t0": 0,
        "tmid": steps // 2,
        "tfinal": steps - 1,
    }
    snapshots: Dict[str, Dict[str, np.ndarray]] = {}

    for i in range(steps):
        t = i / (steps - 1)
        sigma_t = evolve_sigma(t, cfg.sigma0, cfg.sigmaF)
        rho_b = lognormal_density(base_field, sigma_t)

        rho_tot = rho_b + cfg.g_dm * rho_dm3
        B = solver.binding(rho_tot)

        rho_mem = (1 - cfg.lam_mem) * rho_mem + cfg.lam_mem * rho_tot
        rho_mem = np.clip(rho_mem, EPS, None)

        S = 1.0 / (1.0 + np.exp(-(np.log(rho_mem) - np.log(cfg.rho_c)) / cfg.s_rho))

        tau_eff = cfg.tau_dm3_0 * (1.0 + cfg.chi_B * B)
        decay_factor = np.exp(-dt / (tau_eff + EPS))
        rho_dm3_next = rho_dm3 * decay_factor
        dm3_loss = rho_dm3 - rho_dm3_next
        dm3_loss_norm = dm3_loss / (rho_tot.mean() + EPS)

        A += dt * (-cfg.k_relax * (1 - S) * (A - cfg.A_floor) + cfg.k_pin * S * (cfg.A0 - A))
        A += cfg.dm3_to_A * dm3_loss_norm
        A = clamp01(A)

        wlt2 += cfg.dm3_to_lt2 * dm3_loss_norm
        Gamma = cfg.gamma0 * (1 - A) ** cfg.pA * (1 - S) ** cfg.pR
        wlt2 += dt * Gamma * (1 - wlt2)
        wlt2 = clamp01(wlt2)

        rho_dm3 = rho_dm3_next
        w2 = 1.0 - wlt2

        X_V = float(wlt2.mean())
        X_rho = float((rho_tot * wlt2).sum() / (rho_tot.sum() + EPS))
        mean_dm3 = float(rho_dm3.mean())
        V_EM = float(((1 - A) * (1 - S) * w2).mean())
        H_global = compute_H_global(cfg, X_V, mean_dm3, rho_tot.mean())

        history["H_global"].append(H_global)
        history["X_V"].append(X_V)
        history["X_rho"].append(X_rho)
        history["mean_dm3"].append(mean_dm3)
        history["V_EM"].append(V_EM)

        if i in snapshot_indices.values():
            key = [k for k, v in snapshot_indices.items() if v == i][0]
            snapshots[key] = {
                "rho_tot": rho_tot.copy(),
                "rho_dm3": rho_dm3.copy(),
                "A": A.copy(),
                "wlt2": wlt2.copy(),
            }

    for k in history:
        history[k] = np.array(history[k])

    final_fields = {
        "rho_b": rho_b,
        "rho_dm3": rho_dm3,
        "A": A,
        "wlt2": wlt2,
        "rho_mem": rho_mem,
        "B": B,
    }

    return SimulationResult(cfg=cfg, history=history, snapshots=snapshots, final_fields=final_fields)
