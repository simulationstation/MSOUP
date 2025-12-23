"""BAO template models."""

from __future__ import annotations

import numpy as np


def bao_template(s: np.ndarray, scale: float, damping: float) -> np.ndarray:
    """Simple damped sinusoid placeholder for BAO template."""
    return np.sin(s / scale) * np.exp(-(s / damping) ** 2)


def model_xi(s: np.ndarray, alpha: float, bias: float, scale: float, damping: float, nuisance: np.ndarray) -> np.ndarray:
    template = bao_template(alpha * s, scale, damping)
    return bias ** 2 * template + nuisance
