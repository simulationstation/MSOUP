"""Core simulation loop for the ejection + decompression toy model.

v3: Adds M3 container constraint field K that:
- Holds M2 alignment (strengthens pinning where K is high)
- Suppresses peel-off until container decays
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Optional

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
    # v3: K diagnostics
    K_diagnostics: Optional[Dict[str, Any]] = None


EPS = 1e-8


def clamp01(x: np.ndarray) -> np.ndarray:
    return np.clip(x, 0.0, 1.0)


def run_simulation(cfg: SimulationConfig) -> SimulationResult:
    """Run the ejection + decompression simulation.

    v3: Includes M3 container constraint field K dynamics and couplings.
    """
    base_field, rho_b0, rho_dm3, A, wlt2, rho_mem, K = initialize_fields(cfg)
    grid = cfg.grid
    steps = cfg.steps
    dt = cfg.dt_internal
    solver = GravitySolver(grid)

    # v3: Extended history with K diagnostics
    history = {k: [] for k in [
        "H_global", "X_V", "X_rho", "mean_dm3", "V_EM",
        "mean_K", "K_p10", "K_p50", "K_p90", "corr_K_B"
    ]}
    snapshot_indices = {
        "t0": 0,
        "tmid": steps // 2,
        "tfinal": steps - 1,
    }
    snapshots: Dict[str, Dict[str, np.ndarray]] = {}

    # v3: Track K release and peel-off onset
    K_release_step = None  # First step where mean_K < 0.5
    wlt2_onset_step = None  # First step where mean_wlt2 > 0.1

    for i in range(steps):
        t = i / (steps - 1)
        sigma_t = evolve_sigma(t, cfg.sigma0, cfg.sigmaF)
        rho_b = lognormal_density(base_field, sigma_t)

        rho_tot = rho_b + cfg.g_dm * rho_dm3
        B = solver.binding(rho_tot)

        rho_mem = (1 - cfg.lam_mem) * rho_mem + cfg.lam_mem * rho_tot
        rho_mem = np.clip(rho_mem, EPS, None)

        S = 1.0 / (1.0 + np.exp(-(np.log(rho_mem) - np.log(cfg.rho_c)) / cfg.s_rho))

        # === dm3 decay ===
        tau_eff = cfg.tau_dm3_0 * (1.0 + cfg.chi_B * B)
        decay_factor = np.exp(-dt / (tau_eff + EPS))
        rho_dm3_next = rho_dm3 * decay_factor
        dm3_loss = rho_dm3 - rho_dm3_next
        dm3_loss_norm = dm3_loss / (rho_tot.mean() + EPS)

        # === K dynamics (v3) ===
        if cfg.K_enabled:
            # K decays with binding-dependent timescale
            tau_K_eff = cfg.tau_K0 * (1.0 + cfg.chi_K * B)
            K_decay = np.exp(-dt / (tau_K_eff + EPS))
            K = K * K_decay

            # Optional maintenance term (only if kK_maint > 0)
            if cfg.kK_maint > 0:
                K = K + dt * cfg.kK_maint * S * (cfg.K_target - K)

            K = clamp01(K)

        # === A dynamics with K-modulated pinning (v3) ===
        # Relaxation term: unchanged
        A_relax = -cfg.k_relax * (1 - S) * (A - cfg.A_floor)

        # Pinning term: now modulated by K (v3)
        if cfg.K_enabled:
            # K strengthens pinning: multiply by K
            A_pin = cfg.k_pin * S * K * (cfg.A0 - A)
        else:
            # Original behavior without K
            A_pin = cfg.k_pin * S * (cfg.A0 - A)

        A += dt * (A_relax + A_pin)

        # dm3→A coupling: disabled by default in v3 (K_mediated mode)
        if cfg.dm3_to_A_mode == "legacy":
            A += cfg.dm3_to_A * dm3_loss_norm

        A = clamp01(A)

        # === wlt2 dynamics ===
        wlt2 += cfg.dm3_to_lt2 * dm3_loss_norm

        # Compute base Gamma (peel-off rate)
        Gamma = cfg.gamma0 * (1 - A) ** cfg.pA * (1 - S) ** cfg.pR

        # v3: K suppresses peel-off: Gamma_new = Gamma * (1 - K)^pK
        if cfg.K_enabled:
            Gamma = Gamma * ((1 - K) ** cfg.pK)

        # Apply amplifier if enabled (legacy v2 feature)
        if cfg.amplifier_mode == "threshold":
            B_threshold = np.percentile(B, cfg.amp_B_percentile)
            low_B_mask = B < B_threshold
            Gamma = np.where(low_B_mask, Gamma * cfg.amp_gamma_boost, Gamma)

        wlt2 += dt * Gamma * (1 - wlt2)
        wlt2 = clamp01(wlt2)

        rho_dm3 = rho_dm3_next
        w2 = 1.0 - wlt2

        # === Diagnostics ===
        X_V = float(wlt2.mean())
        X_rho = float((rho_tot * wlt2).sum() / (rho_tot.sum() + EPS))
        mean_dm3 = float(rho_dm3.mean())
        V_EM = float(((1 - A) * (1 - S) * w2).mean())
        H_global = compute_H_global(cfg, X_V, mean_dm3, rho_tot.mean())

        # v3: K diagnostics
        mean_K = float(K.mean())
        K_p10 = float(np.percentile(K, 10))
        K_p50 = float(np.percentile(K, 50))
        K_p90 = float(np.percentile(K, 90))

        # Correlation between K and B
        K_flat = K.flatten()
        B_flat = B.flatten()
        if np.std(K_flat) > EPS and np.std(B_flat) > EPS:
            corr_K_B = float(np.corrcoef(K_flat, B_flat)[0, 1])
        else:
            corr_K_B = 0.0

        # Track release and onset times
        if K_release_step is None and mean_K < 0.5:
            K_release_step = i
        if wlt2_onset_step is None and X_V > 0.1:
            wlt2_onset_step = i

        history["H_global"].append(H_global)
        history["X_V"].append(X_V)
        history["X_rho"].append(X_rho)
        history["mean_dm3"].append(mean_dm3)
        history["V_EM"].append(V_EM)
        history["mean_K"].append(mean_K)
        history["K_p10"].append(K_p10)
        history["K_p50"].append(K_p50)
        history["K_p90"].append(K_p90)
        history["corr_K_B"].append(corr_K_B)

        if i in snapshot_indices.values():
            key = [k for k, v in snapshot_indices.items() if v == i][0]
            snapshots[key] = {
                "rho_tot": rho_tot.copy(),
                "rho_dm3": rho_dm3.copy(),
                "A": A.copy(),
                "wlt2": wlt2.copy(),
                "K": K.copy(),  # v3: include K in snapshots
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
        "K": K,  # v3: include K in final fields
    }

    # v3: K diagnostics summary
    K_diagnostics = {
        "K_release_step": K_release_step,
        "K_release_time": K_release_step / (steps - 1) if K_release_step else None,
        "wlt2_onset_step": wlt2_onset_step,
        "wlt2_onset_time": wlt2_onset_step / (steps - 1) if wlt2_onset_step else None,
        "mean_K_initial": float(history["mean_K"][0]),
        "mean_K_final": float(history["mean_K"][-1]),
        "corr_K_B_early": float(np.mean(history["corr_K_B"][:steps // 5])),
        "corr_K_B_mid": float(np.mean(history["corr_K_B"][steps // 3: 2 * steps // 3])),
        "corr_K_B_late": float(np.mean(history["corr_K_B"][-steps // 5:])),
    }

    return SimulationResult(
        cfg=cfg, history=history, snapshots=snapshots,
        final_fields=final_fields, K_diagnostics=K_diagnostics
    )
