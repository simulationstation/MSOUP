"""Modulation and coherence proxies."""

from __future__ import annotations


def coherence_proxy(
    omega: float,
    kappa: float,
    mode: str = "q_factor",
    q_ref: float = 50.0,
    kappa_ref: float = 0.2,
) -> float:
    if mode == "kappa":
        value = 1.0 - (kappa / max(kappa_ref, 1e-9))
    else:
        q_val = omega / max(kappa, 1e-9)
        value = (q_val / max(q_ref, 1e-9))
    return max(0.0, min(1.0, value))


def boundary_index(delta_omega: float, omega: float) -> float:
    return max(0.0, min(1.0, abs(delta_omega) / max(omega, 1e-9)))
