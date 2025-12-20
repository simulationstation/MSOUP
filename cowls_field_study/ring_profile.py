"""Compute 1D residual profiles along theta."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class RingProfile:
    theta_edges: np.ndarray
    r_theta: np.ndarray
    uncertainty: np.ndarray
    coverage: np.ndarray
    weights: np.ndarray


def _huber_mean(values: np.ndarray, delta: float = 1.5) -> float:
    if values.size == 0:
        return np.nan
    med = np.nanmedian(values)
    diff = values - med
    mask = np.abs(diff) < delta * np.nanmedian(np.abs(diff) + 1e-9)
    if not np.any(mask):
        return float(med)
    return float(np.nanmean(values[mask]))


def _bin_reduce(values: np.ndarray, reducer: str = "median") -> float:
    if reducer == "median":
        return float(np.nanmedian(values))
    if reducer == "huber":
        return _huber_mean(values)
    raise ValueError(f"Unknown reducer {reducer}")


def compute_ring_profile(
    residual: np.ndarray,
    noise: np.ndarray,
    arc_mask: np.ndarray,
    center: tuple[float, float],
    theta_bins: np.ndarray,
    reducer: str = "huber",
    window_weights: np.ndarray | None = None,
) -> RingProfile:
    """
    Aggregate residual/noise into azimuthal bins.

    Parameters
    ----------
    residual : np.ndarray
        Residual map.
    noise : np.ndarray
        Noise map.
    arc_mask : np.ndarray
        Boolean mask selecting arc pixels.
    center : tuple
        Lens center (x, y).
    theta_bins : np.ndarray
        Bin edges in degrees.
    reducer : {"median", "huber"}
        Robust statistic to summarize each bin.
    """
    y, x = np.indices(residual.shape)
    theta = np.rad2deg(np.arctan2(y - center[1], x - center[0])) % 360.0
    vals = residual / (noise + 1e-6)

    n_bins = theta_bins.size - 1
    r_theta = np.full(n_bins, np.nan, dtype=float)
    unc = np.full(n_bins, np.nan, dtype=float)
    coverage = np.zeros(n_bins, dtype=float)
    weights = np.zeros(n_bins, dtype=float)

    for i in range(n_bins):
        bin_mask = (theta >= theta_bins[i]) & (theta < theta_bins[i + 1]) & arc_mask
        bin_vals = vals[bin_mask]
        if bin_vals.size == 0:
            continue
        r_theta[i] = _bin_reduce(bin_vals, reducer=reducer)
        mad = np.nanmedian(np.abs(bin_vals - np.nanmedian(bin_vals)))
        unc[i] = 1.4826 * mad / np.sqrt(bin_vals.size + 1e-9)
        coverage[i] = bin_vals.size
        weights[i] = bin_vals.size

    if window_weights is not None:
        ww = np.asarray(window_weights, dtype=float)
        ww = ww[:n_bins]
        weights = weights * ww
    if np.nanmean(weights) > 0:
        weights = weights / (np.nanmean(weights))
    weights = np.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
    return RingProfile(theta_bins, r_theta, unc, coverage, weights)
