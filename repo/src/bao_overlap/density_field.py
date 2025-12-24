"""Density field construction."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    origin: np.ndarray
    maxs: np.ndarray
    padding: float
    grid_shape: Tuple[int, int, int]
    cell_sizes: np.ndarray
    edges: Tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class DensityField:
    grid: np.ndarray
    origin: np.ndarray
    cell_sizes: np.ndarray
    grid_shape: Tuple[int, int, int]


def build_grid_spec(
    data_xyz: np.ndarray,
    random_xyz: np.ndarray,
    target_cell_size: float,
    padding: float,
    max_n_per_axis: int = 512,
) -> GridSpec:
    if target_cell_size <= 0:
        raise ValueError("target_cell_size must be positive")
    if padding < 0:
        raise ValueError("padding must be non-negative")
    if max_n_per_axis <= 0:
        raise ValueError("max_n_per_axis must be positive")

    mins = np.min(np.vstack([data_xyz, random_xyz]), axis=0)
    maxs = np.max(np.vstack([data_xyz, random_xyz]), axis=0)
    span = maxs - mins
    grid_shape = np.ceil((span + 2.0 * padding) / target_cell_size).astype(int)
    if np.any(grid_shape <= 0):
        raise ValueError("grid_shape must be positive on all axes")
    if np.any(grid_shape > max_n_per_axis):
        raise ValueError(
            f"Requested grid_shape {tuple(grid_shape)} exceeds max_n_per_axis={max_n_per_axis}. "
            "Increase target_cell_size or reduce padding."
        )

    edges = tuple(
        np.linspace(mins[i] - padding, maxs[i] + padding, grid_shape[i] + 1) for i in range(3)
    )
    cell_sizes = np.array([edge[1] - edge[0] for edge in edges], dtype="f8")
    return GridSpec(
        origin=mins.astype("f8"),
        maxs=maxs.astype("f8"),
        padding=float(padding),
        grid_shape=(int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])),
        cell_sizes=cell_sizes,
        edges=edges,
    )


def build_density_field(
    data_xyz: np.ndarray,
    random_xyz: np.ndarray,
    data_weight: np.ndarray,
    random_weight: np.ndarray,
    grid_spec: GridSpec,
) -> DensityField:
    """Build overdensity field delta = (ng - alpha nr) / (alpha nr)."""
    edges = grid_spec.edges

    data_hist, _ = np.histogramdd(data_xyz, bins=edges, weights=data_weight)
    rand_hist, _ = np.histogramdd(random_xyz, bins=edges, weights=random_weight)

    alpha = data_hist.sum() / np.maximum(rand_hist.sum(), 1.0)
    denom = np.where(rand_hist > 0, alpha * rand_hist, 1.0)
    delta = (data_hist - alpha * rand_hist) / denom

    grid_origin = np.array([edge[0] for edge in edges], dtype="f4")
    return DensityField(
        grid=delta.astype("f4"),
        origin=grid_origin,
        cell_sizes=grid_spec.cell_sizes.astype("f4"),
        grid_shape=grid_spec.grid_shape,
    )


def gaussian_smooth(field: DensityField, radius: float) -> DensityField:
    from scipy.ndimage import gaussian_filter

    sigma = radius / field.cell_sizes
    smoothed = gaussian_filter(field.grid, sigma=sigma, mode="nearest")
    return DensityField(
        grid=smoothed.astype("f4"),
        origin=field.origin,
        cell_sizes=field.cell_sizes,
        grid_shape=field.grid_shape,
    )


def trilinear_sample(field: DensityField, points: np.ndarray) -> np.ndarray:
    """Sample grid values at points using trilinear interpolation."""
    rel = (points - field.origin) / field.cell_sizes
    idx0 = np.floor(rel).astype(int)
    shape = np.array(field.grid.shape)
    inside = np.all((rel >= 0) & (rel <= (shape - 1)), axis=1)

    idx0 = np.clip(idx0, [0, 0, 0], shape - 2)
    idx1 = idx0 + 1
    frac = np.clip(rel - idx0, 0.0, 1.0)

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
    values = c0 * (1 - fz) + c1 * fz
    values = values.astype("f4")
    values[~inside] = np.nan
    return values


def sample_with_mask(field: DensityField, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Sample grid values and return mask of points inside the grid."""
    rel = (points - field.origin) / field.cell_sizes
    shape = np.array(field.grid.shape)
    inside = np.all((rel >= 0) & (rel <= (shape - 1)), axis=1)
    values = trilinear_sample(field, points)
    return values, inside


def line_integral(field: DensityField, p0: np.ndarray, p1: np.ndarray, step: float) -> Tuple[float, float]:
    """Line integral of the density field between points.

    Returns (integral, outside_fraction) where outside_fraction is the
    fraction of sample points falling outside the grid.
    """
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length == 0.0:
        return 0.0, 1.0
    n_steps = max(int(np.ceil(length / step)) + 1, 2)
    s_vals = np.linspace(0.0, length, n_steps)
    points = p0[None, :] + (s_vals / length)[:, None] * vec[None, :]
    values, inside = sample_with_mask(field, points)
    values = np.where(inside, values, 0.0)
    integral = float(np.trapz(values, s_vals))
    outside_fraction = 1.0 - float(np.mean(inside))
    return integral, outside_fraction


def save_density_field(path: Path, field: DensityField, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        grid=field.grid,
        origin=field.origin,
        cell_sizes=field.cell_sizes,
        grid_shape=np.array(field.grid_shape, dtype=int),
        meta=np.array([json.dumps(meta)], dtype=object),
    )


def load_density_field(path: Path) -> DensityField:
    """Load a density field from disk."""
    data = np.load(path, allow_pickle=True)
    grid_shape = tuple(data["grid_shape"].tolist())
    return DensityField(
        grid=data["grid"],
        origin=data["origin"],
        cell_sizes=data["cell_sizes"],
        grid_shape=grid_shape,
    )
