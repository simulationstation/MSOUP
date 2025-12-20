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

DEFAULT_DATA_ROOT = Path("data") / "jwst_cowls" / "repo"

# Score bins in priority order (M25 = highest quality)
SCORE_BINS = ["M25"] + [f"S{i:02d}" for i in range(12, -1, -1)]

# Expected NIRCam bands
NIRCAM_BANDS = ["F115W", "F150W", "F277W", "F444W"]


@dataclass
class LensRecord:
    """Container describing a COWLS lens directory."""

    lens_id: str
    path: Path
    score_bin: str
    available_bands: List[str]
    catalogue_row: Optional[pd.Series] = None
    model_products: Optional[Dict[str, Path]] = None


def _discover_bands(lens_dir: Path) -> List[str]:
    """Discover available bands by looking for band subdirectories with data.fits."""
    bands = []
    for band in NIRCAM_BANDS:
        band_dir = lens_dir / band
        if band_dir.is_dir() and (band_dir / "data.fits").exists():
            bands.append(band)
            continue
        # Fallback for flat layouts used in tests
        alt_data = lens_dir / f"{lens_dir.name}_{band}_science.fits"
        if alt_data.exists():
            bands.append(band)
    return bands


def discover_lenses(
    data_root: Path = DEFAULT_DATA_ROOT,
    subset: Optional[Iterable[str]] = None,
    score_bins: Optional[List[str]] = None,
) -> List[LensRecord]:
    """
    Discover lens folders beneath ``data_root``.

    Parameters
    ----------
    data_root:
        Base folder containing COWLS repo (with M25, S00, ... subdirs).
    subset:
        Optional iterable of lens IDs to keep.
    score_bins:
        Optional list of score bins to include (e.g., ["M25", "S12", "S11", "S10"]).
        If None, includes all bins.
    """
    data_root = Path(data_root)
    wanted = {s.strip() for s in subset} if subset is not None else None
    bins_to_scan = score_bins if score_bins else SCORE_BINS

    # Try to load catalogue from parent or current dir
    catalogue = load_catalogue(data_root / "catalogue.csv")
    if catalogue is None:
        catalogue = load_catalogue(data_root.parent / "catalogue.csv")

    records: List[LensRecord] = []
    for score_bin in bins_to_scan:
        bin_dir = data_root / score_bin
        if not bin_dir.is_dir():
            continue

        for lens_dir in sorted(bin_dir.iterdir()):
            if not lens_dir.is_dir():
                continue
            if wanted is not None and lens_dir.name not in wanted:
                continue

            bands = _discover_bands(lens_dir)
            if not bands:
                continue  # Skip lenses without any valid band data

            catalogue_row = None
            if catalogue is not None and lens_dir.name in catalogue.index:
                catalogue_row = catalogue.loc[lens_dir.name]

            records.append(
                LensRecord(
                    lens_id=lens_dir.name,
                    path=lens_dir,
                    score_bin=score_bin,
                    available_bands=bands,
                    catalogue_row=catalogue_row,
                    model_products=_discover_model_products(lens_dir),
                )
            )
    if records:
        return records

    # Fallback: treat each subdirectory as a lens when score-bin layout is absent
    for lens_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        bands = _discover_bands(lens_dir)
        if not bands:
            continue
        records.append(
            LensRecord(
                lens_id=lens_dir.name,
                path=lens_dir,
                score_bin="UNKNOWN",
                available_bands=bands,
                catalogue_row=None,
                model_products=_discover_model_products(lens_dir),
            )
        )
    return records


def _discover_model_products(lens_dir: Path) -> Dict[str, Path]:
    """Identify model/residual FITS if present in band result folders."""
    products: Dict[str, Path] = {}

    # Check each band's result folder for model products
    for band in NIRCAM_BANDS:
        result_dir = lens_dir / band / "result"
        if not result_dir.is_dir():
            continue

        # Look for lens model components
        lens_light = result_dir / "lens_light.fits"
        source_light = result_dir / "source_light.fits"
        source_recon = result_dir / "source_reconstruction.fits"

        if lens_light.exists():
            products.setdefault("lens_light", lens_light)
        if source_light.exists():
            products.setdefault("source_light", source_light)
        if source_recon.exists():
            products.setdefault("source_reconstruction", source_recon)

        # Mark that we have modeling results for this band
        if any([lens_light.exists(), source_light.exists()]):
            products[f"{band}_has_model"] = result_dir

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

    In COWLS structure, files are in subdirectories:
    - {lens_dir}/{band}/data.fits (science)
    - {lens_dir}/{band}/noise_map.fits (noise)
    """
    band_dir = lens_dir / band
    science = None
    noise = None

    if band_dir.is_dir():
        science = band_dir / "data.fits" if (band_dir / "data.fits").exists() else None
        noise = band_dir / "noise_map.fits" if (band_dir / "noise_map.fits").exists() else None

    # Flat layout fallback
    if science is None:
        alt_science = lens_dir / f"{lens_dir.name}_{band}_science.fits"
        if alt_science.exists():
            science = alt_science
            alt_noise = lens_dir / f"{lens_dir.name}_{band}_noise.fits"
            noise = alt_noise if alt_noise.exists() else noise

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
