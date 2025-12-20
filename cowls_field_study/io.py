"""I/O and discovery utilities for the field-level COWLS study."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from astropy.io import fits

from .config import DEFAULT_DATA_ROOT


@dataclass
class LensEntry:
    """Metadata about a lens directory."""

    lens_id: str
    score_bin: str
    path: Path
    bands: List[str]


def discover_lenses(
    data_root: Path = DEFAULT_DATA_ROOT, subset: Optional[Iterable[str]] = None, score_bins: Optional[Sequence[str]] = None
) -> List[LensEntry]:
    """
    Walk the COWLS repo layout and return discovered lenses.

    The repository is expected to contain folders like ``M25/LENS_ID/BAND``.
    """
    data_root = Path(data_root)
    wanted = {s.strip() for s in subset} if subset is not None else None
    bins = score_bins if score_bins is not None else sorted([p.name for p in data_root.iterdir() if p.is_dir()])
    lenses: List[LensEntry] = []
    for score_bin in bins:
        bin_dir = data_root / score_bin
        if not bin_dir.is_dir():
            continue
        for lens_dir in sorted(bin_dir.iterdir()):
            if not lens_dir.is_dir():
                continue
            if wanted is not None and lens_dir.name not in wanted:
                continue
            bands = [p.name for p in lens_dir.iterdir() if p.is_dir() and (p / "data.fits").exists()]
            if not bands:
                continue
            lenses.append(LensEntry(lens_dir.name, score_bin, lens_dir, bands))
    return lenses


def read_fits(path: Path) -> Tuple[np.ndarray, fits.Header]:
    """Load FITS array and header."""
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)
        header = hdul[0].header
    return data, header
