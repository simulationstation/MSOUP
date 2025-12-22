from __future__ import annotations

import math
import pathlib
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    h5py = None

PosteriorInput = Union[str, pathlib.Path, np.ndarray]


class _CSVReiterator:
    def __init__(self, path: pathlib.Path, sample_col: str, weight_col: Optional[str]):
        self.path = path
        self.sample_col = sample_col
        self.weight_col = weight_col

    def __iter__(self) -> Iterator[float]:
        cols = [self.sample_col] + ([self.weight_col] if self.weight_col else [])
        for chunk in pd.read_csv(self.path, usecols=cols, chunksize=20000):
            samples = chunk[self.sample_col].to_numpy()
            if self.weight_col and self.weight_col in chunk.columns:
                weights = chunk[self.weight_col].to_numpy()
                for s, w in zip(samples, weights):
                    if not np.isfinite(s) or not np.isfinite(w):
                        continue
                    count = int(max(1, round(float(w))))
                    for _ in range(count):
                        yield float(s)
            else:
                for s in samples:
                    if np.isfinite(s):
                        yield float(s)


@dataclass
class LensPosterior:
    lens_id: str
    z_lens: float
    z_source: Optional[float]
    samples: Iterable[float] | Iterator[float]
    units: str = "D_dt"
    meta: Dict[str, object] = field(default_factory=dict)
    approximate: bool = False

    def iter_samples(self) -> Iterator[float]:
        for s in self.samples:
            if isinstance(s, (float, int, np.floating, np.integer)):
                yield float(s)
            elif isinstance(s, (list, tuple, np.ndarray)):
                for v in np.asarray(s).ravel():
                    yield float(v)
            else:  # pragma: no cover - defensive
                try:
                    yield float(s)
                except Exception as exc:  # pragma: no cover - defensive
                    raise TypeError(f"Unsupported sample type: {type(s)}") from exc


@dataclass
class PosteriorHistogram:
    bin_edges: np.ndarray
    pmf: np.ndarray
    support: str
    sample_mean: float
    sample_std: float
    source: Optional[str] = None

    def log_prob(self, value: float, floor: float = 1e-15) -> float:
        idx = np.searchsorted(self.bin_edges, value, side="right") - 1
        if idx < 0 or idx >= len(self.pmf):
            return math.log(floor)
        prob_density = self.pmf[idx] / max(floor, self.bin_edges[idx + 1] - self.bin_edges[idx])
        return math.log(max(floor, prob_density))


def _ensure_units(meta: Dict[str, object], fallback: str = "D_dt") -> str:
    units = meta.get("units") if meta else None
    if isinstance(units, str) and units.strip():
        return units.strip()
    return fallback


def _load_npz(path: pathlib.Path, key: Optional[str]) -> Tuple[np.ndarray, str]:
    with np.load(path) as data:
        keys = list(data.keys())
        target_key = key or (keys[0] if keys else None)
        if target_key is None or target_key not in data:
            raise ValueError(f"No key '{target_key}' in {path}")  # pragma: no cover - guard
        arr = np.asarray(data[target_key]).astype(float)
        meta_key = "_meta"
        units = "D_dt"
        if meta_key in data:
            try:
                meta_dict = dict(data[meta_key].item())
                units = _ensure_units(meta_dict, fallback="D_dt")
            except Exception:
                pass
        return arr, units


def _load_npy(path: pathlib.Path) -> Tuple[np.ndarray, str]:
    arr = np.load(path)
    return np.asarray(arr).astype(float), "D_dt"


def _load_csv(path: pathlib.Path, key: Optional[str]) -> Tuple[Iterable[float], str]:
    sample_col = key
    weight_col = None
    preview = pd.read_csv(path, nrows=1)
    if sample_col is None:
        for cand in ("D_dt", "ddt", "H0", "h0", "value", "sample"):
            if cand in preview.columns:
                sample_col = cand
                break
    if sample_col is None:
        raise ValueError("No recognizable sample column in CSV posterior.")
    if "weight" in preview.columns:
        weight_col = "weight"
    elif "weights" in preview.columns:
        weight_col = "weights"

    units = "H0" if sample_col.lower().startswith("h0") else "D_dt"
    return _CSVReiterator(path, sample_col, weight_col), units


def _load_hdf5(path: pathlib.Path, key: Optional[str]) -> Tuple[np.ndarray, str]:
    if h5py is None:  # pragma: no cover - optional dependency
        raise ImportError("h5py is required for HDF5 posterior loading")
    with h5py.File(path, "r") as f:
        target_key = key or next(iter(f.keys()))
        if target_key not in f:
            raise ValueError(f"Key {target_key} not in {path}")
        arr = np.asarray(f[target_key]).astype(float)
        units = f[target_key].attrs.get("units", "D_dt")
        if isinstance(units, bytes):
            units = units.decode()
    return arr, _ensure_units({"units": units})


def load_posterior_samples(path: PosteriorInput, key: Optional[str] = None) -> Tuple[Iterable[float], str]:
    p = pathlib.Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    if suffix == ".npz":
        return _load_npz(p, key)
    if suffix == ".npy":
        return _load_npy(p)
    if suffix == ".csv":
        return _load_csv(p, key)
    if suffix in {".h5", ".hdf5"}:
        return _load_hdf5(p, key)
    raise ValueError(f"Unsupported posterior format: {suffix}")


def synthetic_normal_posterior(
    lens_id: str,
    z_lens: float,
    z_source: Optional[float],
    mean: float,
    sigma: float,
    units: str = "D_dt",
    size: int = 2000,
) -> LensPosterior:
    rng = np.random.default_rng(abs(hash(lens_id)) % (2**32))
    samples = rng.normal(loc=mean, scale=sigma, size=size)
    return LensPosterior(
        lens_id=lens_id,
        z_lens=z_lens,
        z_source=z_source,
        samples=samples,
        units=units,
        meta={"approx": True, "sigma": sigma, "mean": mean},
        approximate=True,
    )


def ingest_td_posteriors(
    lens_rows: Iterable[Dict[str, object]],
    base_dir: pathlib.Path,
    sample_column: Optional[str],
    use_full_posteriors: bool,
    bins: int = 240,
) -> List[LensPosterior]:
    posters: List[LensPosterior] = []
    for row in lens_rows:
        lens_id = str(row.get("lens_id", len(posters)))
        zl = float(row.get("z_lens"))
        zs = row.get("z_source")
        zs_val = float(zs) if zs is not None and np.isfinite(zs) else None
        if use_full_posteriors and isinstance(row.get("file_ref"), str) and row["file_ref"]:
            posterior_path = (base_dir / str(row["file_ref"])).expanduser()
            samples, units = load_posterior_samples(posterior_path, key=sample_column)
            posters.append(
                LensPosterior(
                    lens_id=lens_id,
                    z_lens=zl,
                    z_source=zs_val,
                    samples=samples,
                    units=units,
                    meta={"source": str(posterior_path)},
                    approximate=False,
                )
            )
            continue

        # Fallback to summary stats if provided
        value = row.get("value")
        sigma = row.get("sigma")
        if value is None or sigma is None or not (np.isfinite(value) and np.isfinite(sigma)):
            continue
        posters.append(
            synthetic_normal_posterior(
                lens_id=lens_id,
                z_lens=zl,
                z_source=zs_val,
                mean=float(value),
                sigma=float(sigma),
                units=str(row.get("observable_type", "D_dt")),
            )
        )
    return posters


def posterior_to_histogram(posterior: LensPosterior, bins: int = 240) -> PosteriorHistogram:
    # Two-pass streaming: first bounds, then histogram. The iterator is expected to be reusable.
    mins, maxs, count, sample_sum, sample_sq = math.inf, -math.inf, 0, 0.0, 0.0
    for s in posterior.iter_samples():
        if not np.isfinite(s):
            continue
        mins = min(mins, s)
        maxs = max(maxs, s)
        count += 1
        sample_sum += s
        sample_sq += s * s
    if not np.isfinite(mins) or not np.isfinite(maxs) or count == 0:
        raise ValueError(f"No finite samples for lens {posterior.lens_id}")

    span = maxs - mins
    lo = mins - 0.01 * span
    hi = maxs + 0.01 * span
    edges = np.linspace(lo, hi, bins + 1)
    counts = np.zeros(bins, dtype=float)
    for s in posterior.iter_samples():
        if not np.isfinite(s):
            continue
        idx = np.searchsorted(edges, s, side="right") - 1
        if 0 <= idx < bins:
            counts[idx] += 1.0
    pmf = counts / max(1.0, counts.sum())
    mean = sample_sum / count
    var = max(0.0, sample_sq / count - mean ** 2)
    std = math.sqrt(var)
    return PosteriorHistogram(
        bin_edges=edges,
        pmf=pmf,
        support=posterior.units,
        sample_mean=mean,
        sample_std=std,
        source=str(posterior.meta.get("source")) if posterior.meta else None,
    )
