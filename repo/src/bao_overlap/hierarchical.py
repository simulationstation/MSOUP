"""Hierarchical beta inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class BetaResult:
    beta: float
    sigma_beta: float
    meta: Dict[str, float]


def two_step_beta(e_bins: np.ndarray, alpha_bins: np.ndarray, covariance: np.ndarray) -> BetaResult:
    inv_cov = np.linalg.inv(covariance)
    design = np.vstack([np.ones_like(e_bins), e_bins]).T
    lhs = design.T @ inv_cov @ design
    rhs = design.T @ inv_cov @ alpha_bins
    coeffs = np.linalg.solve(lhs, rhs)
    cov_params = np.linalg.inv(lhs)
    beta = float(coeffs[1])
    sigma_beta = float(np.sqrt(cov_params[1, 1]))
    return BetaResult(beta=beta, sigma_beta=sigma_beta, meta={"n_bins": len(e_bins)})


def bayesian_beta(e_bins: np.ndarray, alpha_bins: np.ndarray, covariance: np.ndarray, prior_sigma: float) -> BetaResult:
    inv_cov = np.linalg.inv(covariance)
    design = np.vstack([np.ones_like(e_bins), e_bins]).T
    prior = np.diag([0.0, 1.0 / prior_sigma**2])
    lhs = design.T @ inv_cov @ design + prior
    rhs = design.T @ inv_cov @ alpha_bins
    coeffs = np.linalg.solve(lhs, rhs)
    cov_params = np.linalg.inv(lhs)
    beta = float(coeffs[1])
    sigma_beta = float(np.sqrt(cov_params[1, 1]))
    return BetaResult(beta=beta, sigma_beta=sigma_beta, meta={"n_bins": len(e_bins), "prior_sigma": prior_sigma})
