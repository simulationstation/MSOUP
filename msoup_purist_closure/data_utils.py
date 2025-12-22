"""Shared data loading helpers for purist closure analyses.

These routines intentionally avoid heavy dependencies and keep memory usage
predictable so they can be reused by both the distance-mode runner and the
f0 inference pipeline without duplicating large covariance arrays.
"""
from __future__ import annotations

import pathlib
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from .config import MsoupConfig, ProbeConfig
from .observables import angular_diameter_distance, comoving_distance
from .residuals import load_covariance


def load_sn_observed_data(probe: ProbeConfig) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray], str]:
    """
    Load SN observed data: z, mu_obs (or m_b), sigma, and optionally covariance.

    Returns (DataFrame, Covariance or None, status_reason).
    """
    probe_path = pathlib.Path(probe.path)

    if not probe_path.exists():
        return None, None, f"file not found: {probe.path}"

    try:
        df = pd.read_csv(probe_path)
    except Exception as e:  # pragma: no cover - defensive
        return None, None, f"CSV read error: {e}"

    # Check for required columns
    z_col = probe.z_column or "z"
    if z_col not in df.columns:
        return None, None, f"missing column: {z_col}"

    # Look for mu_obs or m_b
    mu_col = None
    for col in ["mu_obs", "mu", "m_b", "mb", "MU_SH0ES"]:
        if col in df.columns:
            mu_col = col
            break
    if mu_col is None:
        return None, None, "missing observed column: mu_obs/mu/m_b"

    # Look for sigma column
    sigma_col = probe.sigma_column or "sigma"
    if sigma_col not in df.columns:
        for col in ["mu_err_diag", "sigma", "err", "uncertainty"]:
            if col in df.columns:
                sigma_col = col
                break

    if sigma_col not in df.columns:
        return None, None, "missing sigma column"

    # Standardize columns
    df = df.rename(columns={z_col: "z", mu_col: "mu_obs", sigma_col: "sigma"})

    # Load covariance if available
    cov = None
    cov_path = probe_path.parent / (probe_path.stem.replace("_mu", "_cov_stat_sys") + ".npz")
    if cov_path.exists():
        cov = load_covariance(cov_path, len(df))

    return df, cov, "OK"


def load_bao_observed_data(probe: ProbeConfig) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load BAO observed data: z, observable type, value, sigma.
    Returns (DataFrame, status_reason).
    """
    probe_path = pathlib.Path(probe.path)
    if not probe_path.exists():
        return None, f"file not found: {probe.path}"

    try:
        df = pd.read_csv(probe_path)
    except Exception as e:  # pragma: no cover - defensive
        return None, f"CSV read error: {e}"

    # Check for required columns
    required = ["z", "value", "sigma"]
    for col in required:
        if col not in df.columns:
            return None, f"missing column: {col}"

    # Check for observable type column
    if "observable" not in df.columns:
        return None, "missing column: observable"

    return df, "OK"


def load_td_observed_data(probe: ProbeConfig) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Load time-delay observed data: z_lens, z_source, D_dt, sigma.
    Returns (DataFrame, status_reason).
    """
    probe_path = pathlib.Path(probe.path)
    if not probe_path.exists():
        return None, f"file not found: {probe.path}"

    try:
        df = pd.read_csv(probe_path)
    except Exception as e:  # pragma: no cover - defensive
        return None, f"CSV read error: {e}"

    # Check for required columns
    z_lens_col = None
    for col in ["z_lens", "z", "zlens"]:
        if col in df.columns:
            z_lens_col = col
            break
    if z_lens_col is None:
        return None, "missing column: z_lens"

    z_source_col = None
    for col in ["z_source", "zs", "zsrc"]:
        if col in df.columns:
            z_source_col = col
            break
    if z_source_col is None:
        return None, "missing column: z_source"

    ddt_col = None
    for col in ["D_dt", "ddt", "Ddt"]:
        if col in df.columns:
            ddt_col = col
            break
    if ddt_col is None:
        return None, "missing column: D_dt"

    sigma_col = None
    for col in ["sigma", "sigma_D_dt", "err", "uncertainty"]:
        if col in df.columns:
            sigma_col = col
            break
    if sigma_col is None:
        return None, "missing column: sigma"

    df = df.rename(columns={
        z_lens_col: "z_lens",
        z_source_col: "z_source",
        ddt_col: "D_dt",
        sigma_col: "sigma",
    })
    return df, "OK"


def compute_ddt_prediction(z_lens: float, z_source: float, delta_m: float, cfg: MsoupConfig) -> float:
    """
    Compute predicted D_dt = (1+z_l) * D_l * D_ls / D_s (time-delay distance).
    Uses angular diameter distances.
    """
    # Get angular diameter distances
    d_l = angular_diameter_distance(
        np.array([z_lens]), delta_m=delta_m,
        h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0
    )[0]

    d_s = angular_diameter_distance(
        np.array([z_source]), delta_m=delta_m,
        h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0
    )[0]

    chi_l = comoving_distance(
        np.array([z_lens]), delta_m=delta_m,
        h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0
    )[0]
    chi_s = comoving_distance(
        np.array([z_source]), delta_m=delta_m,
        h_early=cfg.h_early, omega_m0=cfg.omega_m0, omega_L0=cfg.omega_L0
    )[0]
    d_ls = (chi_s - chi_l) / (1 + z_source)

    if d_ls <= 0:
        return np.nan
    return (1 + z_lens) * d_l * d_s / d_ls


__all__ = [
    "load_sn_observed_data",
    "load_bao_observed_data",
    "load_td_observed_data",
    "compute_ddt_prediction",
]
