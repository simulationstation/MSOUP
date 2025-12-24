"""Covariance estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class CovarianceResult:
    covariance: np.ndarray
    meta: Dict[str, float]


class StreamingJackknifeCovariance:
    """Streaming accumulator for jackknife covariance.

    Computes jackknife covariance without storing all sample vectors.
    Uses streaming sums to compute: cov = (n-1)/n * Σ(xi - mean)(xi - mean)^T

    Memory fix: Instead of storing all 100 xi vectors (which caused memory
    accumulation), we maintain only running sums:
    - sum_x (M,): Σ xi
    - sum_xx (M,M): Σ outer(xi, xi)

    The covariance is recovered as:
    cov = (n-1)/n * (sum_xx/n - outer(mean, mean))
        = (n-1)/n * (sum_xx - outer(sum_x, sum_x)/n) / n

    This is mathematically equivalent to the batch formula.
    """

    def __init__(self, vector_size: int):
        """Initialize accumulator.

        Parameters
        ----------
        vector_size : int
            Dimension of each xi vector (e.g., n_bins * n_s = 104)
        """
        self.m = vector_size
        self.n = 0
        self.sum_x = np.zeros(vector_size, dtype=np.float64)
        self.sum_xx = np.zeros((vector_size, vector_size), dtype=np.float64)

    def update(self, xi: np.ndarray) -> None:
        """Add a jackknife sample.

        Parameters
        ----------
        xi : np.ndarray
            1D vector of length m
        """
        xi = np.asarray(xi, dtype=np.float64).ravel()
        if xi.shape[0] != self.m:
            raise ValueError(f"Expected vector of length {self.m}, got {xi.shape[0]}")

        self.sum_x += xi
        self.sum_xx += np.outer(xi, xi)
        self.n += 1

    def finalize(self) -> CovarianceResult:
        """Compute final jackknife covariance.

        Returns
        -------
        CovarianceResult
            Covariance matrix with jackknife scaling (n-1)/n
        """
        if self.n < 2:
            raise ValueError(f"Need at least 2 samples, got {self.n}")

        # mean = sum_x / n
        # cov = (n-1)/n * [sum_xx/n - outer(mean, mean)]
        #     = (n-1)/n * [sum_xx/n - outer(sum_x, sum_x)/n^2]
        #     = (n-1)/n^2 * [sum_xx - outer(sum_x, sum_x)/n]
        n = self.n
        mean = self.sum_x / n
        cov = (n - 1) / n * (self.sum_xx / n - np.outer(mean, mean))

        return CovarianceResult(
            covariance=cov,
            meta={"n_jackknife": n, "method": "streaming"}
        )

    def memory_bytes(self) -> int:
        """Return approximate memory usage in bytes."""
        return self.sum_x.nbytes + self.sum_xx.nbytes


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
