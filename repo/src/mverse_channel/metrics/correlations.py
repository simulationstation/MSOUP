"""Correlation metrics."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy import signal


def cross_spectral_density(
    xa: np.ndarray,
    xb: np.ndarray,
    fs: float,
    nperseg: int = 128,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    freqs, pxx = signal.welch(xa, fs=fs, nperseg=nperseg)
    _, pyy = signal.welch(xb, fs=fs, nperseg=nperseg)
    _, pxy = signal.csd(xa, xb, fs=fs, nperseg=nperseg)
    coherence = np.abs(pxy) ** 2 / (pxx * pyy + 1e-12)
    return freqs, pxx, pyy, coherence


def cross_correlation(xa: np.ndarray, xb: np.ndarray) -> Dict[str, np.ndarray]:
    xa = xa - np.mean(xa)
    xb = xb - np.mean(xb)
    corr = signal.correlate(xa, xb, mode="full")
    lags = signal.correlation_lags(xa.size, xb.size, mode="full")
    corr = corr / (np.std(xa) * np.std(xb) * xa.size + 1e-12)
    return {"lags": lags, "corr": corr}
