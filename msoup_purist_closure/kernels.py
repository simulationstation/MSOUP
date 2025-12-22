from __future__ import annotations

import numpy as np
import pandas as pd


class KernelWeights:
    """Kernel weights representation."""

    def __init__(self, z: np.ndarray, weights: np.ndarray, name: str):
        self.z = np.asarray(z, dtype=float)
        self.weights = np.asarray(weights, dtype=float)
        self.name = name
        if self.z.shape != self.weights.shape:
            raise ValueError("z and weights must have same shape")
        if np.any(self.weights < 0):
            raise ValueError("weights must be non-negative")
        norm = self.weights.sum()
        if norm <= 0:
            raise ValueError("weights must sum to positive value")
        self.weights = self.weights / norm

    @property
    def effective_redshift(self) -> float:
        return float(np.average(self.z, weights=self.weights))

    def histogram(self, bins: int = 30):
        return np.histogram(self.z, bins=bins, weights=self.weights, density=True)


def load_probe_csv(path: str, z_column: str = "z", sigma_column: str | None = "sigma") -> pd.DataFrame:
    """Load probe CSV streaming-friendly."""
    df_iter = pd.read_csv(path, usecols=lambda c: c in {z_column, sigma_column} if sigma_column else [z_column], chunksize=10000)
    frames = []
    for chunk in df_iter:
        frames.append(chunk.dropna(subset=[z_column]))
    if not frames:
        raise ValueError(f"No data found in {path}")
    return pd.concat(frames, ignore_index=True)


def kernel_weights(df: pd.DataFrame, kernel: str, z_column: str = "z", sigma_column: str | None = "sigma") -> KernelWeights:
    """Compute kernel weights."""
    z = df[z_column].to_numpy(dtype=float)
    if kernel == "K1":
        weights = np.ones_like(z) / len(z)
    elif kernel == "K2":
        if sigma_column is None or sigma_column not in df.columns:
            raise ValueError("sigma column required for K2 kernel")
        sigma = df[sigma_column].to_numpy(dtype=float)
        inv_var = 1.0 / np.maximum(sigma, 1e-12) ** 2
        weights = inv_var / inv_var.sum()
    elif kernel == "K3":
        raise NotImplementedError("K3 kernel not implemented; sensitivity weights unavailable")
    else:
        raise ValueError(f"Unknown kernel {kernel}")
    return KernelWeights(z=z, weights=weights, name=kernel)
