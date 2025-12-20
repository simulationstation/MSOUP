"""
Perturbation candidate extraction from residual-like maps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter, maximum_filter

from .io import cache_path
from .preprocess import PreprocessResult


@dataclass
class CandidateResult:
    """Container for detected perturbation candidates."""

    theta: np.ndarray
    strengths: np.ndarray
    residual_map: np.ndarray
    method: str

    @property
    def n_candidates(self) -> int:
        """Number of detected candidates."""
        return len(self.theta)


def _load_residual_from_model(model_products: Dict[str, Path]) -> Optional[np.ndarray]:
    """Load residual map if provided by modeling outputs."""
    from .io import read_fits

    if "residual" in model_products:
        return read_fits(model_products["residual"])[0]
    if "model_image" in model_products and "science" in model_products:
        model = read_fits(model_products["model_image"])[0]
        science = read_fits(model_products["science"])[0]
        return science - model
    return None


def build_residual_map(
    image: np.ndarray,
    noise: np.ndarray,
    preprocess: PreprocessResult,
    model_products: Optional[Dict[str, Path]] = None,
) -> Tuple[np.ndarray, str]:
    """
    Prefer modeling residuals; otherwise build a conservative approximate residual.
    """
    if model_products:
        residual = _load_residual_from_model(model_products)
        if residual is not None:
            return residual, "model_residual"

    smooth = gaussian_filter(image, sigma=5.0)
    residual = (image - smooth) * preprocess.arc_mask
    return residual, "approx_residual"


def _find_peaks_above_threshold(
    residual: np.ndarray,
    noise: np.ndarray,
    preprocess: PreprocessResult,
    threshold_sigma: float = 3.0,
    min_separation: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Locate local maxima in residual map restricted to arc mask."""
    snr_resid = residual / (noise + 1e-6)
    snr_resid[~preprocess.arc_mask] = -np.inf
    footprint = np.ones((3, 3))
    local_max = snr_resid == maximum_filter(snr_resid, footprint=footprint)
    candidate_mask = local_max & (snr_resid > threshold_sigma)

    y_idxs, x_idxs = np.where(candidate_mask)
    strengths = snr_resid[y_idxs, x_idxs]
    theta = np.arctan2(y_idxs - preprocess.center[1], x_idxs - preprocess.center[0]) % (2 * np.pi)

    # Enforce minimum angular separation
    order = np.argsort(strengths)[::-1]
    keep_theta = []
    keep_strengths = []
    for idx in order:
        t = theta[idx]
        if all(np.abs(((t - k + np.pi) % (2 * np.pi)) - np.pi) > min_separation for k in keep_theta):
            keep_theta.append(t)
            keep_strengths.append(strengths[idx])

    return np.array(keep_theta), np.array(keep_strengths)


def detect_candidates(
    image: np.ndarray,
    noise: np.ndarray,
    preprocess: PreprocessResult,
    cache_root: Path,
    lens_id: str,
    model_products: Optional[Dict[str, Path]] = None,
    threshold_sigma: float = 3.0,
    min_separation: float = 0.1,
) -> CandidateResult:
    """Detect perturbation candidates with caching."""
    cache_file = cache_path(cache_root, lens_id, "candidates")
    if cache_file.exists():
        data = np.load(cache_file)
        return CandidateResult(
            theta=data["theta"],
            strengths=data["strengths"],
            residual_map=data["residual_map"],
            method=str(data["method"]),
        )

    residual, method = build_residual_map(
        image=image,
        noise=noise,
        preprocess=preprocess,
        model_products=model_products,
    )
    theta, strengths = _find_peaks_above_threshold(
        residual=residual,
        noise=noise,
        preprocess=preprocess,
        threshold_sigma=threshold_sigma,
        min_separation=min_separation,
    )
    np.savez_compressed(
        cache_file,
        theta=theta,
        strengths=strengths,
        residual_map=residual,
        method=method,
    )
    return CandidateResult(theta=theta, strengths=strengths, residual_map=residual, method=method)
