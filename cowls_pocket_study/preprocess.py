"""
Preprocessing utilities for real COWLS images.

Builds masks, estimates centers and Einstein radii, and prepares arc regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import binary_opening, binary_closing, gaussian_filter

from .io import LensRecord


@dataclass
class PreprocessResult:
    """Container for arc mask products."""

    center: Tuple[float, float]
    einstein_radius: float
    arc_mask: np.ndarray
    lens_mask: np.ndarray
    snr_map: np.ndarray


def estimate_center(image: np.ndarray, lens: LensRecord, window: int = 15) -> Tuple[float, float]:
    """
    Estimate lens center using catalogue hints or light centroid.
    """
    if lens.catalogue_row is not None:
        for key in ("x_center", "y_center", "xc", "yc"):
            if key in lens.catalogue_row.index:
                return float(lens.catalogue_row[key]), float(lens.catalogue_row[key.replace("x", "y")])

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
    snr_map: np.ndarray, center: Tuple[float, float], lens: LensRecord,
    min_radius: float = 8.0,
) -> float:
    """
    Use catalogue radius if present, otherwise find arc peak SNR radius.

    Parameters
    ----------
    min_radius : float
        Minimum radius to search for arc (excludes central deflector light).
        Default 8 pixels (~0.5 arcsec at JWST NIRCam scale).
    """
    if lens.catalogue_row is not None:
        for key in ("theta_e", "einstein_radius", "r_ein"):
            if key in lens.catalogue_row.index:
                return float(lens.catalogue_row[key])

    y, x = np.indices(snr_map.shape)
    r = np.hypot(x - center[0], y - center[1])

    # Search for arc peak outside the central region
    r_bins = np.linspace(min_radius, r.max() * 0.6, 40)
    prof, _ = np.histogram(r, bins=r_bins, weights=snr_map)
    counts, _ = np.histogram(r, bins=r_bins)
    snr_prof = prof / (counts + 1e-6)

    # Find first significant local maximum (arc signature)
    r_centers = 0.5 * (r_bins[1:] + r_bins[:-1])

    # Look for peak or plateau in the profile
    # Use weighted approach: higher SNR at smaller radius is more significant
    weighted_prof = snr_prof * (1.0 + 0.5 * (r_centers.max() - r_centers) / r_centers.max())
    peak_idx = np.nanargmax(weighted_prof)

    # Fallback to a reasonable default if profile is flat
    if snr_prof[peak_idx] < 0.5:
        # Use 1/4 of image size as default Einstein radius
        return float(snr_map.shape[0] / 4)

    return float(r_centers[peak_idx])


def _mask_central_region(image: np.ndarray, center: Tuple[float, float], radius_pix: float = 6) -> np.ndarray:
    """Binary mask of central lens light to exclude."""
    y, x = np.indices(image.shape)
    r = np.hypot(x - center[0], y - center[1])
    return r <= radius_pix


def build_arc_mask(
    image: np.ndarray,
    noise: np.ndarray,
    lens: LensRecord,
    snr_threshold: float = 1.2,
    annulus_width: float = 0.50,
) -> PreprocessResult:
    """
    Construct an arc mask guided by SNR in an annulus around the Einstein radius.

    Parameters
    ----------
    snr_threshold : float
        Minimum SNR to include in arc mask. Default 1.2 (lowered to capture faint arcs).
    annulus_width : float
        Fractional width of annulus around Einstein radius. Default 0.50 (widened).
    """
    center = estimate_center(image, lens)
    snr_map = image / (noise + 1e-6)

    # Smooth SNR map before thresholding to reduce noise
    snr_smooth = gaussian_filter(snr_map, sigma=1.5)

    einstein_radius = estimate_einstein_radius(snr_smooth, center, lens)
    y, x = np.indices(image.shape)
    r = np.hypot(x - center[0], y - center[1])
    inner = einstein_radius * (1 - annulus_width)
    outer = einstein_radius * (1 + annulus_width)
    annulus = (r >= inner) & (r <= outer)

    lens_mask = _mask_central_region(image, center)
    candidate = annulus & (snr_smooth > snr_threshold)

    # Light morphological cleaning (1 iteration instead of 2)
    cleaned = binary_opening(candidate, iterations=1)
    cleaned = binary_closing(cleaned, iterations=1)
    cleaned &= ~lens_mask

    return PreprocessResult(
        center=center,
        einstein_radius=einstein_radius,
        arc_mask=cleaned,
        lens_mask=lens_mask,
        snr_map=gaussian_filter(snr_map, sigma=1.0),
    )
