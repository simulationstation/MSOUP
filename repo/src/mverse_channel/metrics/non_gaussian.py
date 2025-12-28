"""Non-Gaussianity metrics."""

from __future__ import annotations

import numpy as np
from scipy import stats


def excess_kurtosis(x: np.ndarray) -> float:
    return float(stats.kurtosis(x, fisher=True, bias=False))


def mardia_kurtosis(samples: np.ndarray) -> float:
    """Compute Mardia kurtosis for multivariate samples."""
    n, d = samples.shape
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    cov = np.cov(centered, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    mahal = np.sum(centered @ cov_inv * centered, axis=1)
    return float(np.mean(mahal**2))


def mardia_skewness(samples: np.ndarray) -> float:
    n, d = samples.shape
    mean = np.mean(samples, axis=0)
    centered = samples - mean
    cov = np.cov(centered, rowvar=False)
    cov_inv = np.linalg.pinv(cov)
    skew = 0.0
    for i in range(n):
        for j in range(n):
            skew += (centered[i] @ cov_inv @ centered[j]) ** 3
    return float(skew / (n**2))
