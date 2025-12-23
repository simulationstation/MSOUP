"""Environment / overlap-strength metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy.ndimage import label

from .density_field import DensityField, line_integral, trilinear_sample


@dataclass
class EnvironmentResult:
    per_galaxy: np.ndarray
    per_pair: np.ndarray
    meta: Dict[str, float]


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
        integral = line_integral(field, p0, p1, step)
        e_values[idx] = integral
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
    for idx, (p0, p1) in enumerate(pairs):
        length = np.linalg.norm(p1 - p0)
        if length == 0.0:
            e_values[idx] = 0.0
            continue
        n_steps = max(int(np.ceil(length / field.cell_size)), 1)
        ts = np.linspace(0, 1, n_steps)
        points = p0[None, :] + ts[:, None] * (p1 - p0)[None, :]
        rel = (points - field.origin) / field.cell_size
        idxs = np.floor(rel).astype(int)
        idxs = np.clip(idxs, [0, 0, 0], np.array(field.grid.shape) - 1)
        inside = super_mask[idxs[:, 0], idxs[:, 1], idxs[:, 2]]
        e_values[idx] = inside.mean()
    return e_values


def normalize(values: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
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
        primary_values, stats = normalize(primary_values)
        meta.update({f"{primary}_mean": stats["mean"], f"{primary}_std": stats["std"]})

    return EnvironmentResult(per_galaxy=per_galaxy, per_pair=primary_values, meta=meta)
