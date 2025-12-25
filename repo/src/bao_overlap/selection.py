"""Selection-function utilities for per-bin random reweighting."""

from __future__ import annotations

from typing import Tuple

import numpy as np


class SelectionFunctionError(RuntimeError):
    """Raised when selection-function inputs are invalid."""


def _require_healpy() -> "module":
    try:
        import healpy as hp  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise SelectionFunctionError(
            "healpy is required for selection-function computations."
        ) from exc
    return hp


def compute_cell_keys(
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    nside: int,
    z_edges: np.ndarray,
) -> np.ndarray:
    hp = _require_healpy()
    ra = np.asarray(ra, dtype="f8")
    dec = np.asarray(dec, dtype="f8")
    z = np.asarray(z, dtype="f8")

    n_zbins = len(z_edges) - 1
    keys = np.full(len(z), -1, dtype=int)

    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z)
    z_min = z_edges[0]
    z_max = z_edges[-1]
    valid = finite & (z >= z_min) & (z <= z_max)
    if not np.any(valid):
        return keys

    theta = np.deg2rad(90.0 - dec[valid])
    phi = np.deg2rad(ra[valid])
    pixel = hp.ang2pix(nside, theta, phi)
    zbin = np.digitize(z[valid], z_edges[1:-1], right=False)
    keys[valid] = pixel * n_zbins + zbin
    return keys


def compute_selection_function_f_b(
    ra: np.ndarray,
    dec: np.ndarray,
    z: np.ndarray,
    e_bin: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
    *,
    nside: int = 32,
    dz: float = 0.05,
    z_min: float = 0.6,
    z_max: float = 1.0,
) -> dict:
    """Compute per-bin selection fraction f_b(theta, z)."""
    _require_healpy()
    ra = np.asarray(ra, dtype="f8")
    dec = np.asarray(dec, dtype="f8")
    z = np.asarray(z, dtype="f8")
    e_bin = np.asarray(e_bin, dtype=int)
    weights = np.asarray(weights, dtype="f8")

    z_edges = np.arange(z_min, z_max + dz, dz)
    n_zbins = len(z_edges) - 1

    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(z) & np.isfinite(weights)
    in_range = (z >= z_min) & (z <= z_max)
    valid = finite & in_range & (e_bin >= 0)

    if not np.any(valid):
        return {
            "pixel_ids": np.array([], dtype=int),
            "zbin_ids": np.array([], dtype=int),
            "f_b": np.zeros((n_bins, 0), dtype="f8"),
            "n_all": np.array([], dtype="f8"),
            "z_edges": z_edges,
            "nside": nside,
        }

    cell_keys = compute_cell_keys(ra[valid], dec[valid], z[valid], nside, z_edges)
    unique_keys, inv = np.unique(cell_keys, return_inverse=True)
    n_cells = len(unique_keys)

    n_all = np.bincount(inv, weights=weights[valid], minlength=n_cells)
    f_b = np.zeros((n_bins, n_cells), dtype="f8")
    for b in range(n_bins):
        mask_b = e_bin[valid] == b
        if not np.any(mask_b):
            continue
        n_b = np.bincount(inv[mask_b], weights=weights[valid][mask_b], minlength=n_cells)
        with np.errstate(divide="ignore", invalid="ignore"):
            f_b[b] = np.where(n_all > 0, n_b / n_all, 0.0)

    pixel_ids = unique_keys // n_zbins
    zbin_ids = unique_keys % n_zbins

    return {
        "pixel_ids": pixel_ids,
        "zbin_ids": zbin_ids,
        "f_b": f_b,
        "n_all": n_all,
        "z_edges": z_edges,
        "nside": nside,
    }


def build_f_b_grid(
    pixel_ids: np.ndarray,
    zbin_ids: np.ndarray,
    f_b: np.ndarray,
    nside: int,
    z_edges: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Expand f_b values into a grid indexed by (pixel, zbin)."""
    hp = _require_healpy()
    n_zbins = len(z_edges) - 1
    n_pix = hp.nside2npix(nside)
    n_bins = f_b.shape[0]
    grid = np.zeros((n_bins, n_pix * n_zbins), dtype="f8")
    cell_keys = pixel_ids * n_zbins + zbin_ids
    grid[:, cell_keys] = f_b
    return grid, n_zbins


def apply_f_b_weights(
    rand_weights: np.ndarray,
    rand_cell_keys: np.ndarray,
    f_b_grid: np.ndarray,
    bin_index: int,
) -> np.ndarray:
    """Apply f_b weights to random weights for a given environment bin."""
    rand_weights = np.asarray(rand_weights, dtype="f8")
    f_b_values = np.zeros_like(rand_weights, dtype="f8")
    valid = rand_cell_keys >= 0
    if np.any(valid):
        f_b_values[valid] = f_b_grid[bin_index, rand_cell_keys[valid]]
    return rand_weights * f_b_values


__all__ = [
    "SelectionFunctionError",
    "compute_selection_function_f_b",
    "build_f_b_grid",
    "apply_f_b_weights",
    "compute_cell_keys",
]
