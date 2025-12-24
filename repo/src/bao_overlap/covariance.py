"""Covariance estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CovarianceResult:
    covariance: np.ndarray
    meta: Dict[str, float]


def covariance_from_mocks(data: np.ndarray) -> CovarianceResult:
    """Compute covariance from mock measurements (n_mock, n_bins)."""
    cov = np.cov(data, rowvar=False, ddof=1)
    return CovarianceResult(covariance=cov, meta={"n_mocks": data.shape[0]})


def covariance_from_jackknife(data: np.ndarray) -> CovarianceResult:
    """Compute jackknife covariance from leave-one-out samples."""
    n_samples = data.shape[0]
    mean = np.mean(data, axis=0)
    diffs = data - mean
    cov = (n_samples - 1) / n_samples * diffs.T @ diffs
    return CovarianceResult(covariance=cov, meta={"n_jackknife": n_samples})


def assign_jackknife_regions(
    ra: np.ndarray,
    dec: np.ndarray,
    n_regions: int,
    scheme: str = "healpix",
    nside: int | None = None,
) -> np.ndarray:
    """Assign jackknife region IDs for RA/DEC positions."""
    ra = np.asarray(ra)
    dec = np.asarray(dec)
    if scheme == "healpix":
        try:
            import healpy as hp
        except ImportError:
            scheme = "ra_bins"
        else:
            if nside is None:
                nside = 4
            theta = np.radians(90.0 - dec)
            phi = np.radians(ra)
            pix = hp.ang2pix(nside, theta, phi)
            return pix % n_regions
    # Fallback: bin by RA quantiles
    edges = np.quantile(ra, np.linspace(0.0, 1.0, n_regions + 1))
    return np.clip(np.digitize(ra, edges[1:-1]), 0, n_regions - 1)
