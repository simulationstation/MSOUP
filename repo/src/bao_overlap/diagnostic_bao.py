"""Diagnostic BAO fitting with nonlinear damping freedom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.optimize import minimize

from .bao_template import bao_template_damped


@dataclass
class DiagnosticFitResult:
    alpha: float
    sigma_nl: float
    chi2: float
    dof: int
    sigma_nl_hit_bounds: bool
    meta: Dict[str, Any]


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


def fit_wedge_diagnostic(
    s: np.ndarray,
    xi: np.ndarray,
    covariance: np.ndarray,
    fit_range: Tuple[float, float],
    nuisance_terms: List[str],
    template_params: Dict[str, float],
    alpha_bounds: Tuple[float, float] = (0.6, 1.4),
    sigma_nl_bounds: Tuple[float, float] = (0.0, 15.0),
    alpha_scan_points: int = 41,
    optimizer: str = "L-BFGS-B",
) -> DiagnosticFitResult:
    mask = (s >= fit_range[0]) & (s <= fit_range[1])
    s_fit = s[mask]
    xi_fit = xi[mask]
    cov_fit = covariance[np.ix_(mask, mask)]
    inv_cov = np.linalg.inv(cov_fit)

    nuisance = _nuisance_design(s_fit, nuisance_terms)

    def chi2_for_alpha_sigma(alpha: float, sigma_nl: float) -> Tuple[float, np.ndarray]:
        template = bao_template_damped(
            alpha * s_fit,
            r_d=template_params["r_d"],
            sigma_nl=sigma_nl,
            omega_m=template_params["omega_m"],
            omega_b=template_params["omega_b"],
            h=template_params["h"],
            n_s=template_params["n_s"],
            sigma8=template_params["sigma8"],
        )
        design = np.column_stack([template, nuisance])
        lhs = design.T @ inv_cov @ design
        rhs = design.T @ inv_cov @ xi_fit
        coeffs = np.linalg.solve(lhs, rhs)
        model = design @ coeffs
        resid = xi_fit - model
        chi2 = float(resid.T @ inv_cov @ resid)
        return chi2, coeffs

    alpha_grid = np.linspace(alpha_bounds[0], alpha_bounds[1], alpha_scan_points)
    chi2_grid = np.empty_like(alpha_grid)
    sigma_grid = np.empty_like(alpha_grid)
    coeff_grid: List[np.ndarray] = []

    sigma_initial = np.mean(sigma_nl_bounds)

    for idx, alpha_val in enumerate(alpha_grid):
        def objective(sigma_arr: np.ndarray) -> float:
            sigma_val = float(sigma_arr[0])
            chi2, _ = chi2_for_alpha_sigma(alpha_val, sigma_val)
            return chi2

        result = minimize(
            objective,
            x0=np.array([sigma_initial]),
            method=optimizer,
            bounds=[sigma_nl_bounds],
        )
        sigma_best = float(result.x[0])
        chi2_best, coeffs = chi2_for_alpha_sigma(alpha_val, sigma_best)
        chi2_grid[idx] = chi2_best
        sigma_grid[idx] = sigma_best
        coeff_grid.append(coeffs)

    best_idx = int(np.argmin(chi2_grid))
    alpha_hat = float(alpha_grid[best_idx])
    sigma_nl_hat = float(sigma_grid[best_idx])
    coeffs = coeff_grid[best_idx]
    chi2_min = float(chi2_grid[best_idx])

    n_params = 2 + len(nuisance_terms)
    dof = max(len(s_fit) - n_params, 1)

    sigma_tol = 1.0e-3
    sigma_hit = (
        abs(sigma_nl_hat - sigma_nl_bounds[0]) <= sigma_tol
        or abs(sigma_nl_hat - sigma_nl_bounds[1]) <= sigma_tol
    )

    meta = {
        "n_bins": len(s_fit),
        "nuisance_terms": nuisance_terms,
        "nuisance_coeffs": coeffs[1:].tolist(),
        "bias_coeff": float(coeffs[0]),
        "optimizer": optimizer,
        "alpha_bounds": list(alpha_bounds),
        "sigma_nl_bounds": list(sigma_nl_bounds),
        "alpha_scan_points": alpha_scan_points,
        "alpha_grid": alpha_grid.tolist(),
        "chi2_grid": chi2_grid.tolist(),
        "sigma_nl_grid": sigma_grid.tolist(),
        "multipoles": ["xi0"],
    }

    return DiagnosticFitResult(
        alpha=alpha_hat,
        sigma_nl=sigma_nl_hat,
        chi2=chi2_min,
        dof=dof,
        sigma_nl_hit_bounds=sigma_hit,
        meta=meta,
    )
