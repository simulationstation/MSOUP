"""Memory metrics."""

from __future__ import annotations

import numpy as np
from scipy import signal


def autocorrelation(x: np.ndarray) -> np.ndarray:
    x = x - np.mean(x)
    corr = signal.correlate(x, x, mode="full")
    corr = corr[corr.size // 2 :]
    return corr / (corr[0] + 1e-12)


def fit_ou_timescale(x: np.ndarray, dt: float) -> float:
    """Fit exponential decay to the autocorrelation tail."""
    ac = autocorrelation(x)
    lag = np.arange(ac.size) * dt
    tail = ac[1:]
    lag_tail = lag[1:]
    mask = tail > 0
    if np.sum(mask) < 3:
        return 0.0
    y = np.log(tail[mask])
    x_vals = lag_tail[mask]
    slope, _ = np.polyfit(x_vals, y, 1)
    if slope >= 0:
        return 0.0
    return float(-1.0 / slope)
