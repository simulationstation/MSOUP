from __future__ import annotations

import math
from typing import Optional

import numpy as np


def _silverman_bandwidth(samples: np.ndarray) -> float:
    n = max(1, samples.size)
    std = float(np.std(samples))
    return 1.06 * std * n ** (-1 / 5)


def histogram_logpdf_from_samples(samples: np.ndarray, grid: np.ndarray, bandwidth: Optional[float] = None) -> np.ndarray:
    """
    Approximate log-pdf over a grid using a smoothed histogram of samples.

    A lightweight KDE is used to avoid memory blow-ups on large arrays.
    """
    samples = np.asarray(samples, dtype=float).reshape(-1)
    grid = np.asarray(grid, dtype=float)
    if samples.size == 0 or grid.size == 0:
        return np.full_like(grid, fill_value=-np.inf, dtype=float)

    bw = bandwidth or _silverman_bandwidth(samples)
    bw = max(bw, 1e-6)

    # Histogram with adaptive bin count
    bins = max(20, min(400, int(np.sqrt(samples.size))))
    hist, edges = np.histogram(samples, bins=bins, density=True)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_width = max(np.diff(edges).min(), 1e-12)

    # Gaussian smoothing via convolution
    offsets = (grid[:, None] - centers[None, :]) / bw
    kernel = np.exp(-0.5 * offsets ** 2)
    kernel_sum = kernel.sum(axis=1)
    density = (kernel * hist[None, :]).sum(axis=1) / (np.sqrt(2 * np.pi) * bw)
    density = np.where(kernel_sum > 0, density, 0.0)
    density = np.clip(density, 1e-300, None)
    return np.log(density)


def robust_residual_loglike(residual: np.ndarray, sigma: np.ndarray, df: float = 6.0) -> np.ndarray:
    """
    Student-t log likelihood for residuals.

    Args:
        residual: Difference obs - pred
        sigma: Scale (per datum)
        df: Degrees of freedom
    """
    residual = np.asarray(residual, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    sigma = np.clip(sigma, 1e-12, None)
    df = float(df)
    z = residual / sigma
    log_norm = math.lgamma((df + 1) / 2) - math.lgamma(df / 2) - 0.5 * (math.log(df) + math.log(np.pi)) - np.log(sigma)
    return log_norm - 0.5 * (df + 1) * np.log1p((z ** 2) / df)
