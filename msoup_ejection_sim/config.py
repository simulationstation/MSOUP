"""Configuration and parameter utilities for the Msoup ejection + decompression simulation."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Tuple
import json


@dataclass
class SimulationConfig:
    grid: int = 256
    steps: int = 400
    dt: float = 1.0 / 400
    seed: int = 123
    kd: float = 18.0
    spectral_index: float = -2.5
    sigma0: float = 0.2
    sigmaF: float = 1.4
    q_pocket: float = 0.97
    pocket_alpha: float = 1.2
    pocket_extra: int = 4
    pocket_smooth: float = 1.5
    f_dm3_0: float = 0.12
    g_dm: float = 1.4
    lam_mem: float = 0.25
    rho_c: float = 1.0
    s_rho: float = 0.8
    k_relax: float = 1.4
    k_pin: float = 1.3
    A0: float = 0.9
    A_floor: float = 0.05
    dm3_to_A: float = 0.25
    dm3_to_lt2: float = 0.35
    tau_dm3_0: float = 0.15
    chi_B: float = 0.6
    gamma0: float = 0.22
    pA: float = 1.5
    pR: float = 1.3
    H_base: float = 67.0
    beta: float = 0.11
    beta_loc: float = 0.08
    beta_dm: float = 0.03
    q_obs: float = 1.2
    b_B: float = 0.2
    rho_void: float = 0.7
    rho_halo: float = 3.5
    early_window: Tuple[float, float] = (0.05, 0.18)
    late_window: Tuple[float, float] = (0.82, 1.0)

    # Late-time readout mode: "volume" (baseline), "tracer" (halo-like weighting)
    late_readout_mode: str = "volume"
    # Tracer weighting parameters (for tracer mode)
    tracer_B_threshold: float = 0.5  # Binding threshold for tracer formation
    tracer_rho_power: float = 1.5    # Density weighting power for tracers

    # Amplifier settings (OFF by default)
    amplifier_mode: str = "none"  # "none", "threshold"
    # Threshold amplifier: sharp peel-off turn-on below binding percentile
    amp_B_percentile: float = 30.0   # Binding percentile below which peel-off amplifies
    amp_gamma_boost: float = 2.0     # Multiplicative boost to gamma0 in low-B regions

    # === M3 Container Constraint Field K (v3) ===
    # K ∈ [0,1] represents M3 constraint region that holds M2 alignment
    # and suppresses peel-off until container decays
    K_enabled: bool = True  # Master switch for K field

    # K initialization: K = clip(K_floor + (K_ceiling - K_floor)*S0, 0, 1)
    # where S0 = sigmoid((log(rho_mem0) - log(rho_c)) / s_rho) reuses existing params
    K_floor: float = 0.05      # Minimum K in low-density regions (FIXED, not scanned)
    K_ceiling: float = 0.95    # Maximum K in high-density regions (FIXED, not scanned)

    # K decay dynamics: tau_K_eff = tau_K0 * (1 + chi_K * B)
    # K(t+dt) = K(t) * exp(-dt / tau_K_eff)
    tau_K0: float = 0.30       # Base K decay timescale (SCANNED)
    chi_K: float = 1.5         # Binding-dependent slowdown of K decay (SCANNED)

    # K maintenance term (FIXED, not scanned) - optional stabilization
    kK_maint: float = 0.0      # Maintenance rate (0 = disabled)
    K_target: float = 1.0      # Target K for maintenance term

    # K coupling to peel-off: Gamma_new = Gamma * (1 - K)^pK
    pK: float = 2.0            # Peel-off suppression power (FIXED, not scanned)

    # dm3_to_A coupling mode: "legacy" uses direct dm3→A, "K_mediated" disables it
    dm3_to_A_mode: str = "K_mediated"  # "legacy" or "K_mediated"

    @property
    def dt_internal(self) -> float:
        return 1.0 / self.steps if self.dt is None else self.dt

    def to_dict(self) -> Dict:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "SimulationConfig":
        return cls(**data)

    def dump(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


@dataclass
class CalibrationConfig:
    max_evals: int = 4000
    n_keep: int = 10
    refine_steps: int = 30
    grid_calib: int = 128
    steps_calib: int = 250
    smoke: bool = False


PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "kd": (10.0, 24.0),
    "sigma0": (0.05, 0.4),
    "sigmaF": (0.9, 1.8),
    "q_pocket": (0.93, 0.985),
    "pocket_alpha": (1.0, 1.6),
    "f_dm3_0": (0.05, 0.4),
    "g_dm": (1.0, 2.2),
    "lam_mem": (0.1, 0.5),
    "rho_c": (0.6, 1.4),
    "s_rho": (0.5, 1.2),
    "k_relax": (0.8, 2.0),
    "k_pin": (0.6, 2.2),
    "A0": (0.75, 0.98),
    "A_floor": (0.0, 0.2),
    "dm3_to_A": (0.05, 0.5),
    "dm3_to_lt2": (0.1, 0.6),
    "tau_dm3_0": (0.05, 0.6),
    "chi_B": (0.0, 1.5),
    "gamma0": (0.05, 0.5),
    "pA": (1.0, 2.2),
    "pR": (1.0, 2.2),
    "H_base": (65.0, 69.0),
    "beta": (0.05, 0.28),
    "beta_loc": (0.02, 0.28),
    "beta_dm": (0.0, 0.08),
    "q_obs": (0.8, 1.8),
    "b_B": (0.0, 0.8),
    # K field parameters (v3) - only tau_K0 and chi_K are scanned
    "tau_K0": (0.05, 2.0),   # Base K decay timescale
    "chi_K": (0.0, 5.0),     # Binding-dependent slowdown
}


def clamp(value, low, high):
    return max(low, min(high, value))
