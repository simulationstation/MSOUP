"""Compute the arc-visibility window S(theta)."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


def pixels_to_theta(
    mask: np.ndarray, center: tuple[float, float], theta_bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert mask pixels to theta bin indices and weights."""
    y, x = np.nonzero(mask)
    if y.size == 0:
        return np.array([], dtype=int), np.array([])
    theta = np.rad2deg(np.arctan2(y - center[1], x - center[0])) % 360.0
    bin_idx = np.digitize(theta, theta_bins) - 1
    return bin_idx, theta


def build_window(
    arc_mask: np.ndarray,
    data: np.ndarray,
    noise: np.ndarray,
    center: tuple[float, float],
    theta_bins: np.ndarray,
    smooth_sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the visibility window S(theta) based on arc brightness.

    Weights use positive SNR and are smoothed circularly, with mean normalized
    to unity.
    """
    theta_bins = np.asarray(theta_bins, dtype=float)
    bin_idx, theta = pixels_to_theta(arc_mask, center, theta_bins)
    if theta.size == 0:
        return np.ones(theta_bins.size - 1), theta_bins

    weights = np.clip(data / (noise + 1e-6), 0, None)
    weights = weights[arc_mask]
    n_bins = theta_bins.size - 1
    s_theta = np.zeros(n_bins, dtype=float)
    for idx, w in zip(bin_idx, weights):
        if 0 <= idx < n_bins:
            s_theta[idx] += w

    s_theta = gaussian_filter1d(s_theta, sigma=smooth_sigma, mode="wrap")
    mean_val = np.nanmean(s_theta) if np.any(np.isfinite(s_theta)) else 1.0
    if mean_val <= 0:
        mean_val = 1.0
    s_theta = s_theta / mean_val
    return s_theta, theta_bins
