"""BAO fitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .bao_template import bao_template


@dataclass
class FitResult:
    alpha: float
    sigma_alpha: float
    chi2: float
    meta: Dict[str, float]


def _nuisance_design(s: np.ndarray, terms: List[str]) -> np.ndarray:
    columns = []
    for term in terms:
        if term == "a0":
            columns.append(np.ones_like(s))
        elif term == "a1/s":
            columns.append(1.0 / s)
        elif term == "a2/s^2":
            columns.append(1.0 / s**2)
        else:
            raise ValueError(f"Unknown nuisance term: {term}")
    return np.vstack(columns).T


def fit_wedge(
    s: np.ndarray,
    xi: np.ndarray,
    covariance: np.ndarray,
    fit_range: Tuple[float, float],
    nuisance_terms: List[str],
    template_params: Dict[str, float],
    alpha_grid: np.ndarray | None = None,
) -> FitResult:
    mask = (s >= fit_range[0]) & (s <= fit_range[1])
    s_fit = s[mask]
    xi_fit = xi[mask]
    cov_fit = covariance[np.ix_(mask, mask)]
    inv_cov = np.linalg.inv(cov_fit)

    if alpha_grid is None:
        alpha_grid = np.linspace(0.8, 1.2, 81)

    best = {"chi2": np.inf}
    for alpha in alpha_grid:
        template = bao_template(
            alpha * s_fit,
            r_d=template_params["r_d"],
            sigma_nl=template_params["sigma_nl"],
            omega_m=template_params["omega_m"],
            omega_b=template_params["omega_b"],
            h=template_params["h"],
            n_s=template_params["n_s"],
        )
        nuisance = _nuisance_design(s_fit, nuisance_terms)
        design = np.column_stack([template, nuisance])
        lhs = design.T @ inv_cov @ design
        rhs = design.T @ inv_cov @ xi_fit
        coeffs = np.linalg.solve(lhs, rhs)
        model = design @ coeffs
        resid = xi_fit - model
        chi2 = float(resid.T @ inv_cov @ resid)
        if chi2 < best["chi2"]:
            best = {"chi2": chi2, "alpha": alpha}

    alpha_idx = np.argmin(np.abs(alpha_grid - best["alpha"]))
    if 0 < alpha_idx < len(alpha_grid) - 1:
        step = alpha_grid[1] - alpha_grid[0]
        sigma_alpha = step
    else:
        sigma_alpha = alpha_grid[1] - alpha_grid[0]

    return FitResult(alpha=best["alpha"], sigma_alpha=sigma_alpha, chi2=best["chi2"], meta={"n_bins": len(s_fit)})
