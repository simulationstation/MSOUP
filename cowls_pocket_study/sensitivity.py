"""
Sensitivity estimation from real COWLS arcs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .io import cache_path


@dataclass
class SensitivityWindowReal:
    """Lightweight window compatible with simulator sampling."""

    theta_grid: np.ndarray
    s_grid: np.ndarray
    lens_id: str

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        """Circular interpolation of S(theta)."""
        theta_wrap = np.mod(theta, 2 * np.pi)
        # np.interp assumes ascending grid; ensure closure by appending 2π
        grid = np.concatenate([self.theta_grid, [2 * np.pi]])
        vals = np.concatenate([self.s_grid, [self.s_grid[0]]])
        return np.interp(theta_wrap, grid, vals)

    def save(self, path: Path) -> None:
        np.savez_compressed(path, theta_grid=self.theta_grid, s_grid=self.s_grid)

    @classmethod
    def load(cls, path: Path, lens_id: str) -> "SensitivityWindowReal":
        data = np.load(path)
        return cls(theta_grid=data["theta_grid"], s_grid=data["s_grid"], lens_id=lens_id)


def _theta_bins(n_bins: int) -> np.ndarray:
    return np.linspace(0, 2 * np.pi, n_bins, endpoint=False)


def compute_sensitivity(
    snr_map: np.ndarray,
    mask: np.ndarray,
    center: Tuple[float, float],
    n_bins: int = 180,
    smooth_sigma: float = 1.5,
    lens_id: str = "",
) -> SensitivityWindowReal:
    """
    Compute S(θ) by binning arc SNR in angular wedges.
    """
    y, x = np.indices(snr_map.shape)
    theta = np.arctan2(y - center[1], x - center[0]) % (2 * np.pi)

    theta_grid = _theta_bins(n_bins)
    bin_indices = np.digitize(theta[mask], bins=np.concatenate([theta_grid, [2 * np.pi]])) - 1
    sums = np.bincount(bin_indices, weights=snr_map[mask], minlength=n_bins)
    counts = np.bincount(bin_indices, minlength=n_bins)
    with np.errstate(invalid="ignore"):
        s_theta = sums / (counts + 1e-6)
    s_theta = gaussian_filter1d(s_theta, sigma=smooth_sigma, mode="wrap")
    s_theta = np.clip(s_theta, 0, None)
    mean_s = np.nanmean(s_theta + 1e-9)
    if mean_s == 0 or not np.isfinite(mean_s):
        s_theta = np.ones_like(theta_grid)
    else:
        s_theta /= mean_s
    return SensitivityWindowReal(theta_grid=theta_grid, s_grid=s_theta, lens_id=lens_id)


def load_or_compute_sensitivity(
    snr_map: np.ndarray,
    mask: np.ndarray,
    center: Tuple[float, float],
    cache_root: Path,
    lens_id: str,
    n_bins: int = 180,
) -> SensitivityWindowReal:
    """Cache wrapper for sensitivity estimation."""
    path = cache_path(cache_root, lens_id, "sensitivity")
    if path.exists():
        return SensitivityWindowReal.load(path, lens_id=lens_id)
    window = compute_sensitivity(
        snr_map=snr_map,
        mask=mask,
        center=center,
        n_bins=n_bins,
        lens_id=lens_id,
    )
    window.save(path)
    return window
