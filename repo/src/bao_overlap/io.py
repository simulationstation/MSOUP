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
    """Read FITS file and handle byte order conversion."""
    with fits.open(path) as hdul:
        data = hdul[1].data
        # Convert to native byte order for pandas compatibility
        df_dict = {}
        for name in data.names:
            col = data[name]
            # Convert big-endian to native if needed
            if col.dtype.byteorder == '>':
                col = col.byteswap().view(col.dtype.newbyteorder('='))
            df_dict[name] = col
        return pd.DataFrame(df_dict)


def _read_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def load_catalog(
    datasets_cfg: Dict[str, Any],
    catalog_key: str,
    region: str = "NGC",
    reconstructed: bool = False,
    dry_run_fraction: float | None = None,
    seed: int | None = None,
) -> Tuple[Catalog, Catalog]:
    """
    Load data and random catalogs for a given survey/region.

    Parameters
    ----------
    datasets_cfg : Dict
        Dataset configuration dictionary.
    catalog_key : str
        Key identifying the catalog (e.g., "eboss_lrgpcmass").
    region : str
        Region to load ("NGC", "SGC", etc.).
    reconstructed : bool
        If True, load reconstructed catalogs.
    dry_run_fraction : float, optional
        Fraction of data to keep for dry runs.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    Tuple[Catalog, Catalog]
        (data_catalog, random_catalog)
    """
    catalog_cfg = datasets_cfg["catalogs"][catalog_key]
    data_root = Path(datasets_cfg.get("data_root", "../data"))
    rng = np.random.default_rng(seed)

    # Get region-specific paths
    region_cfg = catalog_cfg["regions"][region]
    if reconstructed:
        data_path = data_root / region_cfg["data_rec"]
        rand_path = data_root / region_cfg["randoms_rec"]
    else:
        data_path = data_root / region_cfg["data"]
        rand_path = data_root / region_cfg["randoms"]

    fmt = catalog_cfg.get("format", "fits").lower()

    def read_file(path: Path) -> pd.DataFrame:
        if fmt == "fits":
            df = _read_fits(path)
        elif fmt == "parquet":
            df = _read_parquet(path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        # Apply redshift cut if specified
        z_range = catalog_cfg.get("z_range")
        if z_range and "Z" in df.columns:
            mask = (df["Z"] >= z_range[0]) & (df["Z"] <= z_range[1])
            df = df.loc[mask].reset_index(drop=True)

        # Apply dry run fraction
        if dry_run_fraction is not None and dry_run_fraction > 0.0:
            keep = rng.random(len(df)) < dry_run_fraction
            df = df.loc[keep].reset_index(drop=True)

        return df

    print(f"Loading {catalog_key} {region} {'(reconstructed)' if reconstructed else ''}")
    print(f"  Data: {data_path}")
    print(f"  Randoms: {rand_path}")

    data_df = read_file(data_path)
    rand_df = read_file(rand_path)

    print(f"  Loaded {len(data_df):,} data, {len(rand_df):,} randoms")

    # Compute weights
    weights_expr = catalog_cfg.get("weights", {}).get("total", "")

    # Handle column name mapping (WEIGHT_FKP_EBOSS vs WEIGHT_FKP)
    for df in [data_df, rand_df]:
        if "WEIGHT_FKP_EBOSS" in df.columns and "WEIGHT_FKP" not in df.columns:
            df["WEIGHT_FKP"] = df["WEIGHT_FKP_EBOSS"]

    data_w = _eval_weight_expression(data_df, weights_expr)
    rand_w = _eval_weight_expression(rand_df, weights_expr)

    def to_catalog(df: pd.DataFrame, weights: np.ndarray, name: str) -> Catalog:
        required = ["RA", "DEC", "Z"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise KeyError(f"Missing columns in {name}: {missing}")
        return Catalog(
            ra=np.asarray(df["RA"], dtype="f8"),
            dec=np.asarray(df["DEC"], dtype="f8"),
            z=np.asarray(df["Z"], dtype="f8"),
            w=weights,
            meta={"n": len(df), "region": region, "reconstructed": reconstructed},
        )

    return to_catalog(data_df, data_w, "data"), to_catalog(rand_df, rand_w, "randoms")


def save_metadata(path: Path, metadata: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)
