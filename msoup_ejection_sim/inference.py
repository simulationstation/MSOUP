"""Inference utilities for the expansion surrogates."""
from __future__ import annotations

import numpy as np
from typing import Dict, Optional

from .config import SimulationConfig
from .projection import robust_mean


def compute_observer_weights(cfg: SimulationConfig, final_fields: Dict[str, np.ndarray],
                             readout_mode: Optional[str] = None) -> np.ndarray:
    """Compute observer/tracer weights for late-time inference.

    Args:
        cfg: Simulation configuration
        final_fields: Dictionary of final field arrays
        readout_mode: Override for cfg.late_readout_mode ("volume" or "tracer")

    Returns:
        Normalized weight array for observer-weighted averaging
    """
    mode = readout_mode or cfg.late_readout_mode
    rho_mem = final_fields["rho_mem"]
    B = final_fields.get("B")
    rho_tot = final_fields["rho_b"] + cfg.g_dm * final_fields["rho_dm3"]

    if mode == "volume":
        # Volume-weighted (baseline): weight by density and binding
        w_obs = (rho_mem ** cfg.q_obs)
        if B is not None:
            w_obs = w_obs * (1.0 + cfg.b_B * B)
    elif mode == "tracer":
        # Tracer-weighted: represents where M2 observers (halos/galaxies) form
        # Higher weight in high-density, high-binding regions
        if B is not None:
            # Tracer formation requires both high density and sufficient binding
            tracer_prob = np.where(B > cfg.tracer_B_threshold,
                                   (rho_tot ** cfg.tracer_rho_power) * (B ** 0.5),
                                   1e-8)
        else:
            tracer_prob = rho_tot ** cfg.tracer_rho_power
        w_obs = tracer_prob
    else:
        raise ValueError(f"Unknown late_readout_mode: {mode}")

    w_obs = np.clip(w_obs, 1e-8, None)
    w_obs = w_obs / w_obs.sum()
    return w_obs


def infer_expansion(cfg: SimulationConfig, history: Dict[str, np.ndarray],
                    final_fields: Dict[str, np.ndarray],
                    readout_mode: Optional[str] = None) -> Dict[str, float]:
    """Infer expansion rate proxies from simulation results.

    Args:
        cfg: Simulation configuration
        history: Time history of global metrics
        final_fields: Final field states
        readout_mode: Optional override for late readout mode

    Returns:
        Dictionary with H0_early, H0_late, H0_late_global, Delta_H0
    """
    H_hist = history["H_global"]
    X_V_hist = history["X_V"]

    early_mask = _time_window_mask(len(H_hist), cfg.early_window)
    late_mask = _time_window_mask(len(H_hist), cfg.late_window)

    H0_early = robust_mean(H_hist[early_mask])
    H0_late_global = robust_mean(H_hist[late_mask])

    X_V_final = float(X_V_hist[-1])
    A = final_fields["A"]

    # Compute observer weights based on readout mode
    w_obs = compute_observer_weights(cfg, final_fields, readout_mode)

    H_local = cfg.H_base * (1.0 + cfg.beta * X_V_final) * (1.0 + cfg.beta_loc * (1.0 - A))
    H0_late = robust_mean(H_local.flatten(), weights=w_obs.flatten())
    delta = H0_late - H0_early

    diagnostics = {
        "H0_early": float(H0_early),
        "H0_late": float(H0_late),
        "H0_late_global": float(H0_late_global),
        "Delta_H0": float(delta),
    }
    return diagnostics


def infer_expansion_multireadout(cfg: SimulationConfig, history: Dict[str, np.ndarray],
                                  final_fields: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:
    """Infer expansion under multiple readout definitions.

    Returns diagnostics for both 'volume' and 'tracer' readouts.
    """
    results = {}
    for mode in ["volume", "tracer"]:
        results[mode] = infer_expansion(cfg, history, final_fields, readout_mode=mode)
    return results


def morphology_stats(final_fields: Dict[str, np.ndarray], cfg: SimulationConfig):
    rho_tot = final_fields["rho_b"] + cfg.g_dm * final_fields["rho_dm3"]
    void_fraction = float(np.mean(rho_tot < cfg.rho_void))
    high_density_fraction = float(np.mean(rho_tot > cfg.rho_halo))
    structure_amp = float(np.std(np.log(rho_tot + 1e-8)))
    return {
        "void_fraction": void_fraction,
        "high_density_fraction": high_density_fraction,
        "structure_amp": structure_amp,
    }


def _time_window_mask(n_steps: int, window: tuple) -> np.ndarray:
    t = np.linspace(0, 1, n_steps)
    return (t >= window[0]) & (t <= window[1])
