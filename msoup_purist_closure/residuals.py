"""
Residual and chi-square computation for distance-mode analysis.

Supports:
- Diagonal chi-square (always available)
- Full covariance chi-square (if covariance file exists)
- SN offset marginalization (analytic)
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
from scipy import linalg


@dataclass
class ProbeResidualResult:
    """Result of residual computation for a single probe."""

    status: str  # "OK", "NOT_TESTABLE", "ERROR"
    reason: Optional[str] = None
    n: int = 0
    chi2: float = np.nan
    dof: int = 0
    chi2_dof: float = np.nan
    method: str = "none"  # "diagonal", "full_cov", "diag_fallback"
    marginalized_offset: bool = False
    b_hat: float = 0.0
    mean_resid: float = np.nan
    std_resid: float = np.nan
    z_min: float = np.nan
    z_max: float = np.nan
    z_median: float = np.nan
    obs_column: str = ""


def load_covariance(cov_path: pathlib.Path, n_expected: int) -> Optional[np.ndarray]:
    """Load covariance matrix from NPZ file if it exists and is valid."""
    if not cov_path.exists():
        return None

    try:
        data = np.load(cov_path)
        C = data['C'].astype(np.float64)

        # Sanity checks
        if C.shape[0] != n_expected or C.shape[1] != n_expected:
            return None

        # Make symmetric
        C = (C + C.T) / 2.0

        return C
    except Exception:
        return None


def compute_chi2_diagonal(resid: np.ndarray, sigma: np.ndarray) -> Tuple[float, int]:
    """Compute chi-square using diagonal uncertainties."""
    valid = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(valid):
        return np.nan, 0

    r = resid[valid]
    s = sigma[valid]
    chi2 = float(np.sum((r / s) ** 2))
    dof = len(r)
    return chi2, dof


def compute_chi2_covariance(resid: np.ndarray, C: np.ndarray,
                             eigenvalue_floor: float = 1e-10) -> Tuple[float, str]:
    """
    Compute chi-square using full covariance matrix.

    Uses Cholesky factorization if possible, falls back to eigendecomposition.
    Returns (chi2, method) where method is 'full_cov' or 'diag_fallback'.
    """
    n = len(resid)
    if C.shape[0] != n:
        return np.nan, "diag_fallback"

    # Make symmetric
    C = (C + C.T) / 2.0

    try:
        # Try Cholesky first (most numerically stable)
        L = linalg.cholesky(C, lower=True)
        # Solve L @ y = resid, then chi2 = y.T @ y
        y = linalg.solve_triangular(L, resid, lower=True)
        chi2 = float(np.dot(y, y))
        return chi2, "full_cov"
    except linalg.LinAlgError:
        pass

    try:
        # Fall back to eigendecomposition with floor
        eigvals, eigvecs = linalg.eigh(C)

        # Apply eigenvalue floor
        eigvals = np.maximum(eigvals, eigenvalue_floor)

        # chi2 = r.T @ C^{-1} @ r = sum((V.T @ r)^2 / eigvals)
        v_t_r = eigvecs.T @ resid
        chi2 = float(np.sum(v_t_r ** 2 / eigvals))
        return chi2, "full_cov"
    except Exception:
        # Complete failure - fall back to diagonal
        return np.nan, "diag_fallback"


def marginalize_offset_diagonal(resid: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Analytically marginalize SN offset using diagonal uncertainties.

    b_hat = (sum w_i r_i) / (sum w_i) where w_i = 1/sigma_i^2
    Returns (adjusted_resid, b_hat).
    """
    valid = np.isfinite(resid) & np.isfinite(sigma) & (sigma > 0)
    if not np.any(valid):
        return resid, 0.0

    w = 1.0 / (sigma ** 2)
    w[~valid] = 0.0

    sum_w = np.sum(w)
    if sum_w <= 0:
        return resid, 0.0

    b_hat = float(np.sum(w * resid) / sum_w)
    adjusted = resid - b_hat
    return adjusted, b_hat


def marginalize_offset_covariance(resid: np.ndarray, C: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Analytically marginalize SN offset using full covariance.

    b_hat = (1^T C^{-1} r) / (1^T C^{-1} 1)
    Returns (adjusted_resid, b_hat).
    """
    n = len(resid)
    ones = np.ones(n)

    try:
        # Solve C @ x = resid and C @ y = ones
        C_inv_r = linalg.solve(C, resid, assume_a='pos')
        C_inv_1 = linalg.solve(C, ones, assume_a='pos')

        denom = np.dot(ones, C_inv_1)
        if abs(denom) < 1e-15:
            return resid, 0.0

        b_hat = float(np.dot(ones, C_inv_r) / denom)
        adjusted = resid - b_hat
        return adjusted, b_hat
    except Exception:
        # Fall back to diagonal marginalization using sqrt(diag(C))
        sigma = np.sqrt(np.diag(C))
        return marginalize_offset_diagonal(resid, sigma)


def compute_sn_residuals(
    z_obs: np.ndarray,
    mu_obs: np.ndarray,
    mu_pred: np.ndarray,
    sigma: np.ndarray,
    C: Optional[np.ndarray] = None,
    marginalize_offset: bool = True,
) -> ProbeResidualResult:
    """
    Compute SN chi-square with optional offset marginalization.
    """
    n = len(z_obs)
    resid = mu_obs - mu_pred

    result = ProbeResidualResult(
        status="OK",
        n=n,
        z_min=float(np.min(z_obs)),
        z_max=float(np.max(z_obs)),
        z_median=float(np.median(z_obs)),
        obs_column="mu_obs",
    )

    # Marginalize offset if requested
    b_hat = 0.0
    if marginalize_offset:
        if C is not None:
            resid, b_hat = marginalize_offset_covariance(resid, C)
        else:
            resid, b_hat = marginalize_offset_diagonal(resid, sigma)
        result.marginalized_offset = True
        result.b_hat = b_hat

    result.mean_resid = float(np.mean(resid))
    result.std_resid = float(np.std(resid))

    # Compute chi-square
    if C is not None:
        chi2, method = compute_chi2_covariance(resid, C)
        if method == "diag_fallback" or np.isnan(chi2):
            chi2, dof = compute_chi2_diagonal(resid, sigma)
            method = "diag_fallback"
        else:
            dof = n
        result.method = method
    else:
        chi2, dof = compute_chi2_diagonal(resid, sigma)
        result.method = "diagonal"
        dof = n

    # Subtract 1 DOF for marginalized offset
    if marginalize_offset and dof > 1:
        dof -= 1

    result.chi2 = chi2
    result.dof = dof
    result.chi2_dof = chi2 / dof if dof > 0 else np.nan

    return result


def compute_bao_residuals(
    z_obs: np.ndarray,
    obs_values: np.ndarray,
    pred_values: np.ndarray,
    sigma: np.ndarray,
    observable: str,
) -> ProbeResidualResult:
    """
    Compute BAO chi-square (no offset marginalization for BAO).
    """
    n = len(z_obs)
    resid = obs_values - pred_values

    result = ProbeResidualResult(
        status="OK",
        n=n,
        z_min=float(np.min(z_obs)),
        z_max=float(np.max(z_obs)),
        z_median=float(np.median(z_obs)),
        obs_column=observable,
        marginalized_offset=False,
    )

    result.mean_resid = float(np.mean(resid))
    result.std_resid = float(np.std(resid))

    # BAO uses diagonal chi-square (measurements at different z are approximately independent)
    chi2, dof = compute_chi2_diagonal(resid, sigma)
    result.method = "diagonal"
    result.chi2 = chi2
    result.dof = dof
    result.chi2_dof = chi2 / dof if dof > 0 else np.nan

    return result


def compute_td_residuals(
    z_lens: np.ndarray,
    ddt_obs: np.ndarray,
    ddt_pred: np.ndarray,
    sigma: np.ndarray,
) -> ProbeResidualResult:
    """
    Compute time-delay chi-square (no offset marginalization).
    """
    n = len(z_lens)
    resid = ddt_obs - ddt_pred

    result = ProbeResidualResult(
        status="OK",
        n=n,
        z_min=float(np.min(z_lens)),
        z_max=float(np.max(z_lens)),
        z_median=float(np.median(z_lens)),
        obs_column="D_dt",
        marginalized_offset=False,
    )

    result.mean_resid = float(np.mean(resid))
    result.std_resid = float(np.std(resid))

    chi2, dof = compute_chi2_diagonal(resid, sigma)
    result.method = "diagonal"
    result.chi2 = chi2
    result.dof = dof
    result.chi2_dof = chi2 / dof if dof > 0 else np.nan

    return result
