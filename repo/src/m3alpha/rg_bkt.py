"""BKT renormalization-group flows for compact U(1) vortices."""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List, Tuple

import numpy as np

from .mapping import alpha_eff_from_k


@dataclasses.dataclass(frozen=True)
class RGConfig:
    l_max: float = 6.0
    n_steps: int = 600
    k_min: float = 0.2
    k_max: float = 1.5
    y_min: float = 1e-3
    y_max: float = 0.5
    grid_k: int = 9
    grid_y: int = 7
    critical_y0: float = 0.02
    y_threshold: float = 0.1


@dataclasses.dataclass(frozen=True)
class RGResult:
    k_c: float
    alpha_eff_c: float
    sanity_pass: bool
    flow_grid: Dict[Tuple[float, float], np.ndarray]
    message: str


def _rk4_step(k: float, y: float, dl: float) -> Tuple[float, float]:
    def deriv(k_val: float, y_val: float) -> Tuple[float, float]:
        dk_inv = 4.0 * math.pi**3 * y_val**2
        dy = (2.0 - math.pi * k_val) * y_val
        # Convert dk_inv to dk.
        dk = -k_val * k_val * dk_inv
        return dk, dy

    dk1, dy1 = deriv(k, y)
    dk2, dy2 = deriv(k + 0.5 * dl * dk1, y + 0.5 * dl * dy1)
    dk3, dy3 = deriv(k + 0.5 * dl * dk2, y + 0.5 * dl * dy2)
    dk4, dy4 = deriv(k + dl * dk3, y + dl * dy3)

    k_next = k + (dl / 6.0) * (dk1 + 2 * dk2 + 2 * dk3 + dk4)
    y_next = y + (dl / 6.0) * (dy1 + 2 * dy2 + 2 * dy3 + dy4)
    return k_next, y_next


def integrate_flow(k0: float, y0: float, config: RGConfig) -> np.ndarray:
    dl = config.l_max / config.n_steps
    flow = np.zeros((config.n_steps + 1, 3), dtype=float)
    flow[0] = (0.0, k0, y0)
    k, y = k0, y0
    for idx in range(1, config.n_steps + 1):
        k, y = _rk4_step(k, y, dl)
        if k <= 0:
            k = 1e-6
        if not math.isfinite(y):
            y = config.y_threshold
        flow[idx] = (idx * dl, k, y)
    return flow


def _is_disordered(flow: np.ndarray, y0: float, y_threshold: float) -> bool:
    final_y = flow[-1, 2]
    return final_y >= max(y_threshold, y0 * 5.0)


def find_critical_k(y0: float, config: RGConfig) -> float:
    k_low = config.k_min
    k_high = config.k_max
    for _ in range(40):
        k_mid = 0.5 * (k_low + k_high)
        flow_mid = integrate_flow(k_mid, y0, config)
        if _is_disordered(flow_mid, y0, config.y_threshold):
            k_low = k_mid
        else:
            k_high = k_mid
    return 0.5 * (k_low + k_high)


def run_rg_scan(config: RGConfig) -> RGResult:
    k_vals = np.linspace(config.k_min, config.k_max, config.grid_k)
    y_vals = np.linspace(config.y_min, config.y_max, config.grid_y)
    flows: Dict[Tuple[float, float], np.ndarray] = {}

    for k0 in k_vals:
        for y0 in y_vals:
            flows[(float(k0), float(y0))] = integrate_flow(k0, y0, config)

    k_c = find_critical_k(config.critical_y0, config)
    alpha_eff_c = alpha_eff_from_k(k_c)
    sanity_target = 2.0 / math.pi
    sanity_pass = abs(k_c - sanity_target) <= 0.05
    message = (
        "BKT sanity check passed." if sanity_pass else "BKT sanity check failed."
    )

    return RGResult(
        k_c=k_c,
        alpha_eff_c=alpha_eff_c,
        sanity_pass=sanity_pass,
        flow_grid=flows,
        message=message,
    )
