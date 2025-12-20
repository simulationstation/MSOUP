"""Preprocessing: centers, Einstein radii, and arc masks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter


@dataclass
class PreprocessProducts:
    """Cached preprocessing outputs."""

    center: Tuple[float, float]
    einstein_radius: float
    arc_mask: np.ndarray
    lens_mask: np.ndarray
    snr_map: np.ndarray
    mode: str


def estimate_center(image: np.ndarray, metadata: Optional[dict] = None, window: int = 15) -> Tuple[float, float]:
    """
    Estimate the lens center using metadata or light centroid.

    Parameters
    ----------
    image : np.ndarray
        Science image array.
    metadata : dict, optional
        Optional mapping containing ``x_center`` / ``y_center`` or ``xc`` / ``yc``.
    window : int
        Half-width of the central window to compute centroid if metadata absent.
    """
    if metadata:
        for x_key in ("x_center", "xc", "x0"):
            y_key = x_key.replace("x", "y")
            x_val = metadata.get(x_key)
            y_val = metadata.get(y_key)
            if x_val is not None and y_val is not None:
                return float(x_val), float(y_val)

    h, w = image.shape
    cx, cy = w // 2, h // 2
    x0, x1 = max(0, cx - window), min(w, cx + window)
    y0, y1 = max(0, cy - window), min(h, cy + window)
    sub = np.clip(image[y0:y1, x0:x1], 0, None)
    y_grid, x_grid = np.indices(sub.shape)
    total = sub.sum() + 1e-9
    cx_local = (x_grid * sub).sum() / total
    cy_local = (y_grid * sub).sum() / total
    return x0 + cx_local, y0 + cy_local


def estimate_einstein_radius(
    snr_map: np.ndarray,
    center: Tuple[float, float],
    metadata: Optional[dict] = None,
    min_radius: float = 6.0,
) -> float:
    """
    Estimate the Einstein radius using SNR structure.

    The method looks for the first peak in a radial SNR profile outside the
    deflector core. Catalogue hints in ``metadata`` take precedence when
    provided.
    """
    if metadata:
        for key in ("theta_e", "einstein_radius", "r_ein"):
            val = metadata.get(key)
            if val is not None:
                return float(val)

    y, x = np.indices(snr_map.shape)
    r = np.hypot(x - center[0], y - center[1])
    r_bins = np.linspace(min_radius, np.nanmax(r) * 0.8, 48)
    prof, _ = np.histogram(r, bins=r_bins, weights=snr_map)
    counts, _ = np.histogram(r, bins=r_bins)
    snr_prof = prof / (counts + 1e-6)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

    weighted_prof = snr_prof * (1.0 + 0.3 * (r_centers.max() - r_centers) / (r_centers.max() + 1e-9))
    peak_idx = int(np.nanargmax(weighted_prof))
    candidate = float(r_centers[peak_idx])
    if candidate < min_radius * 0.8 or not np.isfinite(candidate):
        return max(min_radius, float(np.nanmedian(r_centers)))
    return candidate


def _mask_central(core_shape: Tuple[int, int], center: Tuple[float, float], radius: float) -> np.ndarray:
    y, x = np.indices(core_shape)
    r = np.hypot(x - center[0], y - center[1])
    return r <= radius


def build_arc_mask(
    image: np.ndarray,
    noise: np.ndarray,
    mode: str,
    metadata: Optional[dict],
    snr_threshold: float,
    annulus_width: float,
    min_radius: float = 6.0,
) -> PreprocessProducts:
    """
    Build an annular arc mask around the Einstein radius.

    The mask is constructed from SNR (or |residual| / noise for residual mode)
    and cleaned morphologically.
    """
    center = estimate_center(image, metadata)
    metric_map = image / (noise + 1e-6)
    if mode.startswith("approx"):
        metric_map = np.abs(image) / (noise + 1e-6)

    metric_smooth = gaussian_filter(metric_map, sigma=1.5)
    einstein_radius = estimate_einstein_radius(metric_smooth, center, metadata, min_radius=min_radius)

    y, x = np.indices(image.shape)
    r = np.hypot(x - center[0], y - center[1])
    inner = einstein_radius * (1 - annulus_width)
    outer = einstein_radius * (1 + annulus_width)
    annulus = (r >= inner) & (r <= outer)
    candidate = annulus & (metric_smooth > snr_threshold)

    lens_mask = _mask_central(image.shape, center, radius=min_radius * 0.8)
    cleaned = binary_opening(candidate, iterations=1)
    cleaned = binary_closing(cleaned, iterations=1)
    cleaned &= ~lens_mask

    return PreprocessProducts(
        center=center,
        einstein_radius=einstein_radius,
        arc_mask=cleaned,
        lens_mask=lens_mask,
        snr_map=metric_smooth,
        mode=mode,
    )


def save_preprocess(cache_path: Path, products: PreprocessProducts) -> None:
    """Persist preprocessing outputs to an ``.npz`` bundle."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        center=np.asarray(products.center, dtype=float),
        einstein_radius=float(products.einstein_radius),
        arc_mask=products.arc_mask,
        lens_mask=products.lens_mask,
        snr_map=products.snr_map,
        mode=products.mode,
    )


def load_preprocess(cache_path: Path) -> Optional[PreprocessProducts]:
    """Load preprocessing outputs if a cache file exists."""
    if not cache_path.exists():
        return None
    data = np.load(cache_path, allow_pickle=True)
    return PreprocessProducts(
        center=tuple(map(float, data["center"].tolist())),
        einstein_radius=float(data["einstein_radius"]),
        arc_mask=data["arc_mask"],
        lens_mask=data["lens_mask"],
        snr_map=data["snr_map"],
        mode=str(data["mode"]),
    )
