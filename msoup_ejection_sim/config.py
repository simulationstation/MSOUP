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
}


def clamp(value, low, high):
    return max(low, min(high, value))
