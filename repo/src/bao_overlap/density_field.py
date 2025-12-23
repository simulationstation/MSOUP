"""Density field construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class DensityField:
    grid: np.ndarray
    origin: np.ndarray
    cell_size: float


def build_density_field(
    data_xyz: np.ndarray,
    random_xyz: np.ndarray,
    random_weight: np.ndarray,
    grid_size: int,
    cell_size: float,
) -> DensityField:
    """Build overdensity field delta = (ng - alpha nr) / (alpha nr)."""
    if grid_size <= 0:
        raise ValueError("grid_size must be positive")

    mins = np.min(np.vstack([data_xyz, random_xyz]), axis=0)
    origin = mins - cell_size
    maxs = np.max(np.vstack([data_xyz, random_xyz]), axis=0)
    edges = [np.linspace(origin[i], maxs[i] + cell_size, grid_size + 1) for i in range(3)]

    data_hist, _ = np.histogramdd(data_xyz, bins=edges)
    rand_hist, _ = np.histogramdd(random_xyz, bins=edges, weights=random_weight)

    alpha = data_hist.sum() / np.maximum(rand_hist.sum(), 1.0)
    denom = np.where(rand_hist > 0, alpha * rand_hist, 1.0)
    delta = (data_hist - alpha * rand_hist) / denom

    return DensityField(grid=delta.astype("f4"), origin=origin.astype("f4"), cell_size=cell_size)


def gaussian_smooth(field: DensityField, radius: float) -> DensityField:
    from scipy.ndimage import gaussian_filter

    sigma = radius / field.cell_size
    smoothed = gaussian_filter(field.grid, sigma=sigma, mode="nearest")
    return DensityField(grid=smoothed.astype("f4"), origin=field.origin, cell_size=field.cell_size)


def trilinear_sample(field: DensityField, points: np.ndarray) -> np.ndarray:
    """Sample grid values at points using trilinear interpolation."""
    rel = (points - field.origin) / field.cell_size
    idx0 = np.floor(rel).astype(int)
    frac = rel - idx0

    nx, ny, nz = field.grid.shape
    idx0 = np.clip(idx0, [0, 0, 0], [nx - 2, ny - 2, nz - 2])
    idx1 = idx0 + 1

    c000 = field.grid[idx0[:, 0], idx0[:, 1], idx0[:, 2]]
    c100 = field.grid[idx1[:, 0], idx0[:, 1], idx0[:, 2]]
    c010 = field.grid[idx0[:, 0], idx1[:, 1], idx0[:, 2]]
    c001 = field.grid[idx0[:, 0], idx0[:, 1], idx1[:, 2]]
    c110 = field.grid[idx1[:, 0], idx1[:, 1], idx0[:, 2]]
    c101 = field.grid[idx1[:, 0], idx0[:, 1], idx1[:, 2]]
    c011 = field.grid[idx0[:, 0], idx1[:, 1], idx1[:, 2]]
    c111 = field.grid[idx1[:, 0], idx1[:, 1], idx1[:, 2]]

    fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    c00 = c000 * (1 - fx) + c100 * fx
    c01 = c001 * (1 - fx) + c101 * fx
    c10 = c010 * (1 - fx) + c110 * fx
    c11 = c011 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def line_integral(field: DensityField, p0: np.ndarray, p1: np.ndarray, step: float) -> float:
    vec = p1 - p0
    length = np.linalg.norm(vec)
    if length == 0.0:
        return 0.0
    n_steps = max(int(np.ceil(length / step)), 1)
    ts = np.linspace(0, 1, n_steps)
    points = p0[None, :] + ts[:, None] * vec[None, :]
    values = trilinear_sample(field, points)
    return np.trapz(values, ts) / 1.0
