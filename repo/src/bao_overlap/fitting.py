"""BAO fitting utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
from scipy.optimize import minimize

from .bao_template import bao_template


@dataclass
class FitResult:
    alpha: float
    sigma_alpha: float
    chi2: float
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


def fit_wedge(
    s: np.ndarray,
    xi: np.ndarray,
    covariance: np.ndarray,
    fit_range: Tuple[float, float],
    nuisance_terms: List[str],
    template_params: Dict[str, float],
    alpha_bounds: Tuple[float, float] = (0.8, 1.2),
    optimizer: str = "L-BFGS-B",
) -> FitResult:
    mask = (s >= fit_range[0]) & (s <= fit_range[1])
    s_fit = s[mask]
    xi_fit = xi[mask]
    cov_fit = covariance[np.ix_(mask, mask)]
    inv_cov = np.linalg.inv(cov_fit)

    nuisance = _nuisance_design(s_fit, nuisance_terms)

    def chi2_for_alpha(alpha: float) -> Tuple[float, np.ndarray]:
        template = bao_template(
            alpha * s_fit,
            r_d=template_params["r_d"],
            sigma_nl=template_params["sigma_nl"],
            omega_m=template_params["omega_m"],
            omega_b=template_params["omega_b"],
            h=template_params["h"],
            n_s=template_params["n_s"],
        )
        design = np.column_stack([template, nuisance])
        lhs = design.T @ inv_cov @ design
        rhs = design.T @ inv_cov @ xi_fit
        coeffs = np.linalg.solve(lhs, rhs)
        model = design @ coeffs
        resid = xi_fit - model
        chi2 = float(resid.T @ inv_cov @ resid)
        return chi2, coeffs

    def objective(alpha_arr: np.ndarray) -> float:
        alpha_val = float(alpha_arr[0])
        chi2, _ = chi2_for_alpha(alpha_val)
        return chi2

    initial = np.array([np.mean(alpha_bounds)])
    result = minimize(
        objective,
        x0=initial,
        method=optimizer,
        bounds=[alpha_bounds],
    )
    alpha_hat = float(result.x[0])
    chi2_min, coeffs = chi2_for_alpha(alpha_hat)

    delta = 1.0e-3
    chi2_plus, _ = chi2_for_alpha(min(alpha_bounds[1], alpha_hat + delta))
    chi2_minus, _ = chi2_for_alpha(max(alpha_bounds[0], alpha_hat - delta))
    second_deriv = (chi2_plus - 2.0 * chi2_min + chi2_minus) / (delta**2)
    sigma_alpha = float(np.sqrt(2.0 / second_deriv)) if second_deriv > 0 else float(delta)

    n_params = 1 + len(nuisance_terms)
    dof = max(len(s_fit) - n_params, 1)

    meta = {
        "n_bins": len(s_fit),
        "dof": dof,
        "nuisance_terms": nuisance_terms,
        "nuisance_coeffs": coeffs[1:].tolist(),
        "bias_coeff": float(coeffs[0]),
        "optimizer": optimizer,
    }
    return FitResult(alpha=alpha_hat, sigma_alpha=sigma_alpha, chi2=chi2_min, meta=meta)
