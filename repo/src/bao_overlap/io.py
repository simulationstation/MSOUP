"""I/O utilities for catalogs and configs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits


@dataclass(frozen=True)
class Catalog:
    ra: np.ndarray
    dec: np.ndarray
    z: np.ndarray
    w: np.ndarray
    meta: Dict[str, Any]


def load_yaml(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_run_config(path: str | Path) -> Dict[str, Any]:
    cfg = load_yaml(path)
    prereg = load_yaml(cfg["preregistration"])
    datasets = load_yaml(cfg["datasets"])
    cfg["_preregistration"] = prereg
    cfg["_datasets"] = datasets
    return cfg


def _eval_weight_expression(df: pd.DataFrame, expression: str) -> np.ndarray:
    if not expression:
        return np.ones(len(df), dtype="f8")
    return df.eval(expression).astype("f8")


def _read_fits(path: Path) -> pd.DataFrame:
    with fits.open(path) as hdul:
        data = hdul[1].data
        return pd.DataFrame({name: data[name] for name in data.names})


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_catalog(
    datasets_cfg: Dict[str, Any],
    catalog_key: str,
    dry_run_fraction: float | None = None,
    seed: int | None = None,
) -> Tuple[Catalog, Catalog]:
    catalog_cfg = datasets_cfg["catalogs"][catalog_key]
    rng = np.random.default_rng(seed)

    def read_block(block: Dict[str, Any]) -> pd.DataFrame:
        path = Path(block["path"])
        fmt = block["format"].lower()
        if fmt == "fits":
            df = _read_fits(path)
        elif fmt == "parquet":
            df = _read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")
        if dry_run_fraction is not None and dry_run_fraction > 0.0:
            keep = rng.random(len(df)) < dry_run_fraction
            df = df.loc[keep].reset_index(drop=True)
        return df

    data_df = read_block(catalog_cfg["data"])
    rand_df = read_block(catalog_cfg["randoms"])

    weights_expr = catalog_cfg.get("weights", {}).get("total", "")
    data_w = _eval_weight_expression(data_df, weights_expr)
    rand_w = _eval_weight_expression(rand_df, weights_expr)

    def to_catalog(df: pd.DataFrame, weights: np.ndarray) -> Catalog:
        required = ["RA", "DEC", "Z"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        return Catalog(
            ra=np.asarray(df["RA"], dtype="f8"),
            dec=np.asarray(df["DEC"], dtype="f8"),
            z=np.asarray(df["Z"], dtype="f8"),
            w=weights,
            meta={"n": len(df)},
        )

    return to_catalog(data_df, data_w), to_catalog(rand_df, rand_w)


def save_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
