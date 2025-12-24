"""Environment / overlap-strength metrics."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
from scipy.ndimage import label
from joblib import Parallel, delayed

from .density_field import DensityField, line_integral, trilinear_sample

# Number of parallel workers (default to CPU count - 1, minimum 1)
N_JOBS = max(1, int(os.environ.get("BAO_N_JOBS", os.cpu_count() or 4)))


@dataclass
class EnvironmentResult:
    per_galaxy: np.ndarray
    per_pair: np.ndarray
    meta: Dict[str, float]


@dataclass
class EnvironmentAssignment:
    per_galaxy_raw: np.ndarray
    per_galaxy_norm: np.ndarray
    meta: Dict[str, Any]


def compute_e1(
    field: DensityField,
    pairs: np.ndarray,
    step: float,
    rng: np.random.Generator,
    subsample: float,
) -> np.ndarray:
    n_pairs = pairs.shape[0]
    if subsample < 1.0:
        keep = rng.random(n_pairs) < subsample
        pairs = pairs[keep]
    e_values = np.zeros(len(pairs), dtype="f4")
    for idx, (p0, p1) in enumerate(pairs):
        integral, _ = line_integral(field, p0, p1, step)
        length = np.linalg.norm(p1 - p0)
        e_values[idx] = integral / length if length > 0 else 0.0
    return e_values


def compute_e2(field: DensityField, pairs: np.ndarray, delta_threshold: float, min_volume: int) -> np.ndarray:
    mask = field.grid > delta_threshold
    labeled, n_labels = label(mask)
    if n_labels == 0:
        return np.zeros(pairs.shape[0], dtype="f4")

    counts = np.bincount(labeled.ravel())
    valid = counts >= min_volume
    valid[0] = False
    valid_labels = np.where(valid)[0]
    if valid_labels.size == 0:
        return np.zeros(pairs.shape[0], dtype="f4")

    super_mask = np.isin(labeled, valid_labels)
    e_values = np.zeros(pairs.shape[0], dtype="f4")
    min_cell = float(np.min(field.cell_sizes))
    for idx, (p0, p1) in enumerate(pairs):
        length = np.linalg.norm(p1 - p0)
        if length == 0.0:
            e_values[idx] = 0.0
            continue
        n_steps = max(int(np.ceil(length / min_cell)), 1)
        ts = np.linspace(0, 1, n_steps)
        points = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        rel = (points - field.origin) / field.cell_sizes
        idxs = np.floor(rel).astype(int)
        idxs = np.clip(idxs, [0, 0, 0], np.array(field.grid.shape) - 1)
        inside = super_mask[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
        e_values[idx] = inside.mean()
    return e_values


def normalize(values: np.ndarray, method: str = "mean_std") -> Tuple[np.ndarray, Dict[str, float]]:
    if method == "median_mad":
        median = float(np.median(values))
        mad_raw = float(np.median(np.abs(values - median))) if np.any(values) else 1.0
        mad_raw = mad_raw if mad_raw > 0 else 1.0
        scale = 1.4826 * mad_raw
        return (values - median) / scale, {"median": median, "mad": mad_raw, "scale": scale}
    mean = float(np.mean(values))
    std = float(np.std(values)) if np.std(values) > 0 else 1.0
    return (values - mean) / std, {"mean": mean, "std": std}


def compute_environment(
    field: DensityField,
    galaxy_xyz: np.ndarray,
    pair_xyz: np.ndarray,
    step: float,
    rng: np.random.Generator,
    subsample: float,
    delta_threshold: float,
    min_volume: int,
    normalize_output: bool,
    normalization_method: str,
    primary: str,
) -> EnvironmentResult:
    per_galaxy = trilinear_sample(field, galaxy_xyz)
    e1 = compute_e1(field, pair_xyz, step, rng, subsample)
    e2 = compute_e2(field, pair_xyz, delta_threshold, min_volume)

    if primary == "E1":
        primary_values = e1
    else:
        primary_values = e2

    meta = {"primary": primary}
    if normalize_output:
        primary_values, stats = normalize(primary_values, method=normalization_method)
        if "mean" in stats:
            meta.update({f"{primary}_mean": stats["mean"], f"{primary}_std": stats["std"]})
        else:
            meta.update({f"{primary}_median": stats["median"], f"{primary}_mad": stats["mad"]})

    return EnvironmentResult(per_galaxy=per_galaxy, per_pair=primary_values, meta=meta)


def compute_pair_e1(
    field: DensityField,
    p0: np.ndarray,
    p1: np.ndarray,
    step: float,
    max_outside_fraction: float,
) -> Tuple[float, bool, float]:
    """Compute E1 for a single pair with mask handling."""
    integral, outside_fraction = line_integral(field, p0, p1, step)
    length = float(np.linalg.norm(p1 - p0))
    if length == 0.0:
        return 0.0, False, outside_fraction
    if outside_fraction > max_outside_fraction:
        return np.nan, False, outside_fraction
    return integral / length, True, outside_fraction


def _process_single_galaxy(
    idx: int,
    galaxy_xyz: np.ndarray,
    field_grid: np.ndarray,
    field_origin: np.ndarray,
    field_cell_sizes: np.ndarray,
    s_min: float,
    s_max: float,
    step: float,
    max_outside_fraction: float,
    max_pairs_per_galaxy: int,
    candidate_multiplier: int,
    seed: int,
) -> Tuple[int, float, int, int, int]:
    """Process a single galaxy's E1 computation. Returns (idx, e1_mean, attempted, valid, invalid)."""
    n_gal = len(galaxy_xyz)
    rng = np.random.default_rng(seed + idx)

    # Reconstruct a minimal field-like object for line_integral
    class FieldProxy:
        def __init__(self, grid, origin, cell_sizes):
            self.grid = grid
            self.origin = origin
            self.cell_sizes = cell_sizes

    field = FieldProxy(field_grid, field_origin, field_cell_sizes)

    n_candidates = min(n_gal - 1, max_pairs_per_galaxy * candidate_multiplier)
    raw_choices = rng.choice(n_gal - 1, size=n_candidates, replace=False)
    neighbor_idx = np.where(raw_choices >= idx, raw_choices + 1, raw_choices)
    offsets = galaxy_xyz[neighbor_idx] - galaxy_xyz[idx]
    distances = np.linalg.norm(offsets, axis=1)
    in_range = (distances >= s_min) & (distances <= s_max)
    neighbor_idx = neighbor_idx[in_range][:max_pairs_per_galaxy]

    if neighbor_idx.size == 0:
        return (idx, np.nan, 0, 0, 0)

    attempted = neighbor_idx.size
    valid_count = 0
    invalid_count = 0
    e_values = []

    for jdx in neighbor_idx:
        e_val, is_valid, _ = compute_pair_e1(
            field=field,
            p0=galaxy_xyz[idx],
            p1=galaxy_xyz[jdx],
            step=step,
            max_outside_fraction=max_outside_fraction,
        )
        if is_valid:
            e_values.append(e_val)
            valid_count += 1
        else:
            invalid_count += 1

    mean_e1 = float(np.mean(e_values)) if e_values else np.nan
    return (idx, mean_e1, attempted, valid_count, invalid_count)


def compute_per_galaxy_mean_e1(
    field: DensityField,
    galaxy_xyz: np.ndarray,
    s_min: float,
    s_max: float,
    step: float,
    rng: np.random.Generator,
    pair_subsample_fraction: float,
    max_outside_fraction: float,
    max_pairs_per_galaxy: int | None = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Compute per-galaxy mean E1 across BAO-range pairs (parallelized)."""
    n_gal = len(galaxy_xyz)
    per_galaxy = np.full(n_gal, np.nan, dtype="f8")
    attempted = np.zeros(n_gal, dtype=int)
    valid = np.zeros(n_gal, dtype=int)
    invalid = np.zeros(n_gal, dtype=int)

    if n_gal < 2:
        return per_galaxy, {
            "attempted_pairs": attempted,
            "valid_pairs": valid,
            "invalid_pairs": invalid,
        }

    if max_pairs_per_galaxy is None:
        max_pairs_per_galaxy = max(1, int(pair_subsample_fraction * (n_gal - 1)))
    max_pairs_per_galaxy = min(max_pairs_per_galaxy, n_gal - 1)
    candidate_multiplier = 5

    # Get base seed for reproducibility
    base_seed = rng.integers(0, 2**31)

    # Progress tracking
    n_jobs = N_JOBS
    print(f"    [E1 parallel] Starting {n_gal} galaxies on {n_jobs} workers...", flush=True)

    # Run in parallel with progress reporting
    batch_size = max(1000, n_gal // 100)  # Report every ~1%

    results = Parallel(n_jobs=n_jobs, verbose=0, batch_size="auto")(
        delayed(_process_single_galaxy)(
            idx=idx,
            galaxy_xyz=galaxy_xyz,
            field_grid=field.grid,
            field_origin=field.origin,
            field_cell_sizes=field.cell_sizes,
            s_min=s_min,
            s_max=s_max,
            step=step,
            max_outside_fraction=max_outside_fraction,
            max_pairs_per_galaxy=max_pairs_per_galaxy,
            candidate_multiplier=candidate_multiplier,
            seed=base_seed,
        )
        for idx in range(n_gal)
    )

    # Unpack results
    for idx, mean_e1, att, val, inv in results:
        per_galaxy[idx] = mean_e1
        attempted[idx] = att
        valid[idx] = val
        invalid[idx] = inv

    valid_count = np.sum(np.isfinite(per_galaxy))
    print(f"    [E1 parallel] Done: {valid_count}/{n_gal} valid", flush=True)

    meta = {
        "attempted_pairs": attempted,
        "valid_pairs": valid,
        "invalid_pairs": invalid,
        "max_pairs_per_galaxy": max_pairs_per_galaxy,
        "max_outside_fraction": max_outside_fraction,
        "n_parallel_jobs": n_jobs,
    }
    return per_galaxy, meta


def assign_environment(
    field: DensityField,
    galaxy_xyz: np.ndarray,
    s_min: float,
    s_max: float,
    step: float,
    rng: np.random.Generator,
    pair_subsample_fraction: float,
    max_outside_fraction: float,
    normalization_method: str,
    max_pairs_per_galaxy: int | None = None,
) -> EnvironmentAssignment:
    """Compute per-galaxy mean E1 and normalization."""
    per_galaxy_raw, meta = compute_per_galaxy_mean_e1(
        field=field,
        galaxy_xyz=galaxy_xyz,
        s_min=s_min,
        s_max=s_max,
        step=step,
        rng=rng,
        pair_subsample_fraction=pair_subsample_fraction,
        max_outside_fraction=max_outside_fraction,
        max_pairs_per_galaxy=max_pairs_per_galaxy,
    )

    valid_mask = np.isfinite(per_galaxy_raw)
    normed = np.full_like(per_galaxy_raw, np.nan, dtype="f8")
    stats = {}
    if np.any(valid_mask):
        normed_vals, stats = normalize(per_galaxy_raw[valid_mask], method=normalization_method)
        normed[valid_mask] = normed_vals
    meta.update(stats)
    meta["normalization_method"] = normalization_method
    return EnvironmentAssignment(per_galaxy_raw=per_galaxy_raw, per_galaxy_norm=normed, meta=meta)
