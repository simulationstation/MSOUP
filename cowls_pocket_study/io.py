"""
I/O utilities for the JWST COWLS pocket-domain study.

Responsibilities
----------------
- Map lens folders under ``data/jwst_cowls``.
- Read FITS science/noise/model products.
- Provide band selection (including automatic SNR-based choice).
- Persist lightweight metadata for downstream steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from astropy.io import fits

DEFAULT_DATA_ROOT = Path("data") / "jwst_cowls"


@dataclass
class LensRecord:
    """Container describing a COWLS lens directory."""

    lens_id: str
    path: Path
    available_bands: List[str]
    catalogue_row: Optional[pd.Series] = None
    model_products: Optional[Dict[str, Path]] = None


def _extract_band_from_name(path: Path) -> Optional[str]:
    """Heuristic band parser from filenames."""
    for part in path.name.split("_"):
        if part.upper().startswith("F") and part.upper().endswith("W"):
            return part.upper()
    return None


def discover_lenses(
    data_root: Path = DEFAULT_DATA_ROOT, subset: Optional[Iterable[str]] = None
) -> List[LensRecord]:
    """
    Discover lens folders beneath ``data_root``.

    Parameters
    ----------
    data_root:
        Base folder containing COWLS data.
    subset:
        Optional iterable of lens IDs to keep.
    """
    data_root = Path(data_root)
    lens_dirs = [p for p in data_root.iterdir() if p.is_dir()]
    if subset is not None:
        wanted = {s.strip() for s in subset}
        lens_dirs = [p for p in lens_dirs if p.name in wanted]

    catalogue = load_catalogue(data_root / "catalogue.csv")
    records: List[LensRecord] = []
    for lens_dir in sorted(lens_dirs):
        band_files = list(lens_dir.glob("*.fits"))
        bands = []
        for f in band_files:
            band = _extract_band_from_name(f)
            if band:
                bands.append(band)
        catalogue_row = catalogue.loc[lens_dir.name] if catalogue is not None and lens_dir.name in catalogue.index else None
        records.append(
            LensRecord(
                lens_id=lens_dir.name,
                path=lens_dir,
                available_bands=sorted(set(bands)),
                catalogue_row=catalogue_row,
                model_products=_discover_model_products(lens_dir),
            )
        )
    return records


def _discover_model_products(lens_dir: Path) -> Dict[str, Path]:
    """Identify model/residual FITS if present."""
    products: Dict[str, Path] = {}
    for candidate in lens_dir.glob("*residual*.fits"):
        products["residual"] = candidate
    for candidate in lens_dir.glob("*model*.fits"):
        products.setdefault("model_image", candidate)
    return products


def load_catalogue(path: Path) -> Optional[pd.DataFrame]:
    """Load optional catalogue CSV keyed by lens_id."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "lens_id" in df.columns:
        df = df.set_index("lens_id")
    return df


def read_fits(path: Path) -> Tuple[np.ndarray, fits.Header]:
    """Read a FITS image into (data, header)."""
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
    return data, header


def _robust_std(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> float:
    """Robust standard deviation using MAD."""
    if mask is not None:
        arr = arr[mask]
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    return 1.4826 * mad + 1e-6


def _estimate_noise(image: np.ndarray, noise: Optional[np.ndarray] = None) -> np.ndarray:
    """Fallback noise estimate when per-pixel noise is missing."""
    if noise is not None:
        return noise
    sigma = _robust_std(image)
    return np.full_like(image, sigma)


def _band_priority() -> List[str]:
    """Preference ordering if auto selection is ambiguous."""
    return ["F277W", "F444W"]


def _compute_band_snr(image: np.ndarray, noise: np.ndarray, mask: Optional[np.ndarray]) -> float:
    """Estimate arc SNR within a mask."""
    usable = np.isfinite(image) & np.isfinite(noise) if mask is None else mask & np.isfinite(image) & np.isfinite(noise)
    if not np.any(usable):
        return np.nan
    snr_map = image / (noise + 1e-6)
    return float(np.nanmean(snr_map[usable]))


def choose_band_for_lens(
    lens: LensRecord, band: str = "auto", mask: Optional[np.ndarray] = None
) -> Optional[str]:
    """
    Select a band for analysis.

    If ``band`` is ``"auto"``, choose the available band with the highest
    average SNR within ``mask``. When SNR values tie or cannot be computed,
    fall back to the default priority ordering.
    """
    if band != "auto":
        return band if band in lens.available_bands else None

    best_band = None
    best_snr = -np.inf
    for b in lens.available_bands:
        img_path, noise_path = locate_band_files(lens.path, b)
        if img_path is None:
            continue
        image, _ = read_fits(img_path)
        noise = read_fits(noise_path)[0] if noise_path else None
        noise = _estimate_noise(image, noise)
        snr = _compute_band_snr(image, noise, mask)
        if np.isnan(snr):
            continue
        if snr > best_snr:
            best_snr = snr
            best_band = b

    if best_band is None:
        for b in _band_priority():
            if b in lens.available_bands:
                best_band = b
                break
    return best_band


def locate_band_files(lens_dir: Path, band: str) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Locate science and noise files for a band.

    Heuristic: look for files containing the band name and either ``"noise"``
    or ``"ivar"`` for noise maps.
    """
    science = None
    noise = None
    for f in lens_dir.glob(f"*{band}*.fits"):
        name_lower = f.name.lower()
        if "noise" in name_lower or "ivar" in name_lower:
            noise = f
        else:
            science = f if science is None else science
    return science, noise


def load_band_data(
    lens: LensRecord, band: str
) -> Tuple[np.ndarray, np.ndarray, fits.Header]:
    """
    Load image + noise for a given band.

    Returns
    -------
    image : np.ndarray
    noise : np.ndarray
    header : fits.Header
    """
    img_path, noise_path = locate_band_files(lens.path, band)
    if img_path is None:
        raise FileNotFoundError(f"No science image for {lens.lens_id} in band {band}")
    image, header = read_fits(img_path)
    noise = read_fits(noise_path)[0] if noise_path else None
    noise = _estimate_noise(image, noise)
    return image, noise, header


def load_lens_data(
    lens: LensRecord, band: str = "auto", mask: Optional[np.ndarray] = None
) -> Tuple[str, np.ndarray, np.ndarray, fits.Header]:
    """
    Load the preferred band for a lens.

    Returns
    -------
    band_used, image, noise, header
    """
    band_used = choose_band_for_lens(lens, band, mask)
    if band_used is None:
        raise RuntimeError(f"Unable to choose band for {lens.lens_id}")
    image, noise, header = load_band_data(lens, band_used)
    return band_used, image, noise, header


def cache_path(root: Path, lens_id: str, stem: str, suffix: str = ".npz") -> Path:
    """Utility to build cache file paths."""
    cache_dir = root / "cache" / lens_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{stem}{suffix}"
