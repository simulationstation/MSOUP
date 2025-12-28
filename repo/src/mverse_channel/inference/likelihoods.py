"""Likelihood utilities."""

from __future__ import annotations

from typing import Dict

import numpy as np


def gaussian_summary_loglike(
    observed: Dict[str, float],
    predicted: Dict[str, float],
    sigma: float,
) -> float:
    keys = sorted(set(observed) & set(predicted))
    if not keys:
        return 0.0
    residuals = np.array([observed[k] - predicted[k] for k in keys])
    return float(-0.5 * np.sum((residuals / (sigma + 1e-12)) ** 2))
