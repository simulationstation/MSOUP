"""Mapping from stiffness to effective coupling."""
from __future__ import annotations

import math


def k_from_helicity(helicity_modulus: float, temperature: float) -> float:
    """Return renormalized stiffness K_R from helicity modulus.

    We adopt K_R := Υ / T (with J=1), consistent with XY conventions where
    K = J/T at the bare level.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return helicity_modulus / temperature


def g_eff_sq_from_k(k_r: float) -> float:
    """Effective squared coupling g_eff^2 := 1 / K_R."""
    if k_r <= 0:
        return math.inf
    return 1.0 / k_r


def alpha_eff_from_k(k_r: float) -> float:
    """Return alpha_eff := g_eff^2 / (4*pi)."""
    g_sq = g_eff_sq_from_k(k_r)
    return g_sq / (4.0 * math.pi)
