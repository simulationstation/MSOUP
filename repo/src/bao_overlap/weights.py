"""Weight utilities."""

from __future__ import annotations

import numpy as np


def fkp_weight(nbar: np.ndarray, p0: float) -> np.ndarray:
    """Compute FKP weights: 1 / (1 + nbar * p0)."""
    return 1.0 / (1.0 + nbar * p0)


def combine_weights(*weights: np.ndarray) -> np.ndarray:
    if not weights:
        raise ValueError("No weights provided")
    total = np.ones_like(weights[0], dtype="f8")
    for w in weights:
        total *= w
    return total
