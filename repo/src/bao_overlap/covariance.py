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
