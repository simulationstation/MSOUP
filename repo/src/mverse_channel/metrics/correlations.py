"""Correlation metrics.

This module provides phase-robust spectral coherence metrics for detecting
cross-channel correlations. Primary detection uses coherence magnitude
(phase-invariant), not time-domain correlation sign.
"""

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
    """Compute cross-spectral density and coherence.

    Returns:
        freqs: Frequency array
        pxx: Power spectral density of xa
        pyy: Power spectral density of xb
        coherence: Magnitude-squared coherence |Cxy|^2 / (Cxx * Cyy)
    """
    freqs, pxx = signal.welch(xa, fs=fs, nperseg=nperseg)
    _, pyy = signal.welch(xb, fs=fs, nperseg=nperseg)
    _, pxy = signal.csd(xa, xb, fs=fs, nperseg=nperseg)
    coherence = np.abs(pxy) ** 2 / (pxx * pyy + 1e-12)
    return freqs, pxx, pyy, coherence


def cross_spectrum_complex(
    xa: np.ndarray,
    xb: np.ndarray,
    fs: float,
    nperseg: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute complex cross-spectrum Sxy(f).

    Returns:
        freqs: Frequency array
        sxy: Complex cross-spectral density
    """
    freqs, sxy = signal.csd(xa, xb, fs=fs, nperseg=nperseg)
    return freqs, sxy


def coherence_magnitude(
    xa: np.ndarray,
    xb: np.ndarray,
    fs: float,
    nperseg: int = 128,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute magnitude coherence |Coh_AB(f)| (phase-robust).

    This is the primary detection metric: sqrt(|Sxy|^2 / (Sxx * Syy)).
    It is invariant to the phase relationship between signals.

    Returns:
        freqs: Frequency array
        coh_mag: Coherence magnitude at each frequency (0 to 1)
    """
    freqs, pxx = signal.welch(xa, fs=fs, nperseg=nperseg)
    _, pyy = signal.welch(xb, fs=fs, nperseg=nperseg)
    _, pxy = signal.csd(xa, xb, fs=fs, nperseg=nperseg)
    # Coherence magnitude = |Sxy| / sqrt(Sxx * Syy)
    coh_mag = np.abs(pxy) / np.sqrt(pxx * pyy + 1e-12)
    return freqs, coh_mag


def coherence_band_integrated(
    xa: np.ndarray,
    xb: np.ndarray,
    fs: float,
    f_low: float = 0.0,
    f_high: float = None,
    nperseg: int = 128,
) -> float:
    """Compute band-integrated coherence magnitude.

    Integrates |Coh_AB(f)| over frequency band [f_low, f_high].

    Returns:
        Integrated coherence (scalar)
    """
    if f_high is None:
        f_high = fs / 2.0
    freqs, coh_mag = coherence_magnitude(xa, xb, fs, nperseg)
    mask = (freqs >= f_low) & (freqs <= f_high)
    if not np.any(mask):
        return 0.0
    df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
    return float(np.sum(coh_mag[mask]) * df)


def coherence_peak_magnitude(
    xa: np.ndarray,
    xb: np.ndarray,
    fs: float,
    nperseg: int = 128,
) -> float:
    """Peak coherence magnitude across all frequencies.

    Primary detection statistic: max_f |Coh_AB(f)|
    """
    _, coh_mag = coherence_magnitude(xa, xb, fs, nperseg)
    return float(np.max(coh_mag))


def cross_correlation(xa: np.ndarray, xb: np.ndarray) -> Dict[str, np.ndarray]:
    """Time-domain cross-correlation (secondary diagnostic only).

    Note: Do NOT use the sign of this for primary detection.
    Use coherence_magnitude for phase-robust detection.
    """
    xa = xa - np.mean(xa)
    xb = xb - np.mean(xb)
    corr = signal.correlate(xa, xb, mode="full")
    lags = signal.correlation_lags(xa.size, xb.size, mode="full")
    corr = corr / (np.std(xa) * np.std(xb) * xa.size + 1e-12)
    return {"lags": lags, "corr": corr}
