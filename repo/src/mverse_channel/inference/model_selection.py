"""Model comparison metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ModelComparison:
    aic: float
    bic: float


def aic(loglike: float, k_params: int) -> float:
    return 2 * k_params - 2 * loglike


def bic(loglike: float, k_params: int, n: int) -> float:
    n = max(n, 2)
    return k_params * float(np.log(n)) - 2 * loglike
