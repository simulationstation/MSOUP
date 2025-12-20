"""Product discovery utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Candidate substrings to look for when identifying model products
MODEL_KEYWORDS = [
    "residual",
    "model",
    "source",
    "reconstruction",
    "autolens",
    "pylens",
    "fit",
    "lens_model",
    "pixelized",
]


def list_band_products(lens_path: Path, band: str) -> Dict[str, Path]:
    """Discover science, noise, PSF, and model-like products under a band folder."""
    band_dir = lens_path / band
    products: Dict[str, Path] = {}
    if not band_dir.is_dir():
        return products

    for candidate in band_dir.rglob("*.fits"):
        name = candidate.name.lower()
        stem = candidate.stem.lower()
        if stem == "data":
            products.setdefault("data", candidate)
            continue
        if "noise" in name:
            products.setdefault("noise", candidate)
        if "psf" in name:
            products.setdefault("psf", candidate)
        for kw in MODEL_KEYWORDS:
            if kw in name:
                products.setdefault(f"model:{kw}", candidate)
                break
    return products


def choose_noise_psf(products: Dict[str, Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """Pick noise and PSF paths when present."""
    return products.get("noise"), products.get("psf")


def _prioritize_residual(keys: List[str]) -> Optional[str]:
    ordering = [
        "pixelized",
        "source_reconstruction",
        "reconstruction",
        "residual",
        "autolens",
        "pylens",
        "model",
        "fit",
        "lens_model",
        "source",
    ]
    for token in ordering:
        for key in keys:
            if token in key:
                return key
    return keys[0] if keys else None


def choose_residual_product(products: Dict[str, Path], prefer_model: bool = True) -> Tuple[Optional[Path], str]:
    """
    Select the residual path and mode label.

    Returns
    -------
    residual_path : Path or None
        Path to the best available residual product.
    mode_label : str
        ``"model_residual"`` when model products exist, otherwise
        ``"approx_residual"`` signalling downstream approximation.
    """
    model_like = [k for k in products if k.startswith("model:")]
    residual_like = [k for k in model_like if "residual" in k or "reconstruction" in k or "pixelized" in k]
    if prefer_model and residual_like:
        chosen = _prioritize_residual(residual_like)
        return products[chosen], "model_residual"

    if prefer_model and model_like:
        chosen = _prioritize_residual(model_like)
        return products[chosen], "model_residual"

    if "data" in products:
        model_keys = [k for k in model_like if "model" in k]
        if model_keys:
            return products[model_keys[0]], "model_residual"

    return None, "approx_residual"
