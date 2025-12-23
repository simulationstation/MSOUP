from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .config import MaxIDConfig


@dataclass
class LensPosterior:
    lens_id: str
    z_lens: float
    z_source: float
    obs_type: str
    support: str
    value: Optional[float] = None
    sigma: Optional[float] = None
    samples: Optional[np.ndarray] = None
    sample_mean: Optional[float] = None
    sample_std: Optional[float] = None
    file_ref: Optional[Path] = None

    @property
    def has_samples(self) -> bool:
        return self.samples is not None and self.samples.size > 0


def _resolve_path(path: str | Path, base: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return base / p


def _load_samples(path: Path, column: str, chunk_size: int) -> Tuple[np.ndarray, str]:
    ext = path.suffix.lower()
    if ext in {".npy", ".npz"}:
        arr = np.load(path, mmap_mode="r")
        if isinstance(arr, np.lib.npyio.NpzFile):
            first_key = list(arr.keys())[0]
            data = np.asarray(arr[first_key])
        else:
            data = np.asarray(arr)
        return data.reshape(-1), column

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        values = np.asarray(data.get(column, data.get("samples", [])), dtype=float)
        return values.reshape(-1), column

    # Default: CSV
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunk_size):
        cols = [c for c in chunk.columns if c.lower() == column.lower() or c.lower() in {"ddt", "d_dt", "h0", "h_0"}]
        if not cols:
            continue
        chunks.append(np.asarray(chunk[cols[0]].values, dtype=float))
    if not chunks:
        return np.array([], dtype=float), column
    data = np.concatenate(chunks)
    return data.reshape(-1), column


def load_td_master(cfg: MaxIDConfig) -> List[LensPosterior]:
    td_path = Path(cfg.paths.td_master)
    if not td_path.exists():
        return []

    df = pd.read_csv(td_path)
    rows: List[LensPosterior] = []
    for _, row in df.iterrows():
        lens_id = str(row.get("lens_id", row.get("name", len(rows))))
        z_lens = float(row["z_lens"]) if "z_lens" in row else float(row.get("z", 0.0))
        z_source = float(row.get("z_source", row.get("zs", 0.0)))
        obs_type = str(row.get("observable_type", "D_dt"))
        support = str(row.get("support", obs_type))
        file_ref = row.get("file_ref")
        value = row.get("value")
        sigma = row.get("sigma") or row.get("sigma_D_dt") or row.get("uncertainty")

        samples: Optional[np.ndarray] = None
        sample_mean = None
        sample_std = None
        resolved_file: Optional[Path] = None
        if isinstance(file_ref, str):
            resolved_file = _resolve_path(file_ref, Path(cfg.paths.posterior_dir))
            if resolved_file.exists() and not cfg.summary_only:
                samples, _ = _load_samples(resolved_file, cfg.posterior_column, cfg.compute.chunk_size)
                if samples.size > 0:
                    sample_mean = float(np.mean(samples))
                    sample_std = float(np.std(samples))
        if (samples is None or samples.size == 0) and value is not None:
            samples = None

        rows.append(
            LensPosterior(
                lens_id=lens_id,
                z_lens=z_lens,
                z_source=z_source,
                obs_type=obs_type,
                support=support,
                value=float(value) if value is not None else None,
                sigma=float(sigma) if sigma is not None else None,
                samples=samples,
                sample_mean=sample_mean,
                sample_std=sample_std,
                file_ref=resolved_file,
            )
        )
    return rows
