"""Product discovery utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Candidate substrings to look for when identifying model products
# NOTE: Match against stem (filename without extension) to avoid false positives
# Priority: prefer image-plane products (source_light, lens_light) over source-plane (reconstruction)
MODEL_KEYWORDS = [
    "source_light",
    "lens_light",
    "model_image",
    "residual",
    "autolens",
    "pylens",
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
        if stem == "noise_map" or stem.endswith("_noise_map"):
            products.setdefault("noise", candidate)
            continue
        if stem == "psf" or stem.startswith("psf_"):
            products.setdefault("psf", candidate)
            continue
        # Check for model products using stem (no .fits extension)
        for kw in MODEL_KEYWORDS:
            if kw in stem:
                products.setdefault(f"model:{kw}", candidate)
                break
    return products


def choose_noise_psf(products: Dict[str, Path]) -> Tuple[Optional[Path], Optional[Path]]:
    """Pick noise and PSF paths when present."""
    return products.get("noise"), products.get("psf")


def choose_residual_product(products: Dict[str, Path], prefer_model: bool = True) -> Tuple[Dict[str, Path], str]:
    """
    Select the best available products for computing residuals.

    Returns
    -------
    residual_products : Dict[str, Path]
        Dictionary with 'source_light' and/or 'lens_light' paths if available.
        If a direct 'residual' product exists, it's returned as {'residual': path}.
    mode_label : str
        ``"model_residual"`` when model products exist, otherwise
        ``"approx_residual"`` signalling downstream approximation.
    """
    model_like = [k for k in products if k.startswith("model:")]

    # Best case: we have both source_light and lens_light for proper model subtraction
    source_light_key = next((k for k in model_like if "source_light" in k), None)
    lens_light_key = next((k for k in model_like if "lens_light" in k), None)

    if prefer_model and source_light_key and lens_light_key:
        return {
            "source_light": products[source_light_key],
            "lens_light": products[lens_light_key],
        }, "model_residual"

    # Check for direct residual product
    residual_key = next((k for k in model_like if "residual" in k), None)
    if prefer_model and residual_key:
        return {"residual": products[residual_key]}, "model_residual"

    # Fallback: just source_light available
    if prefer_model and source_light_key:
        return {"source_light": products[source_light_key]}, "model_residual"

    return {}, "approx_residual"
