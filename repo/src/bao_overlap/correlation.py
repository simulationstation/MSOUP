"""Correlation function computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class PairCounts:
    s_edges: np.ndarray
    mu_edges: np.ndarray
    dd: np.ndarray
    dr: np.ndarray
    rr: np.ndarray
    meta: Dict[str, float]


def _pairwise(data_xyz: np.ndarray, rand_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pairwise separation and mu for all data-data, data-rand, rand-rand."""
    dd = _pairwise_blocks(data_xyz, data_xyz, autocorr=True)
    dr = _pairwise_blocks(data_xyz, rand_xyz, autocorr=False)
    rr = _pairwise_blocks(rand_xyz, rand_xyz, autocorr=True)
    return dd, dr, rr


def _pairwise_blocks(a: np.ndarray, b: np.ndarray, autocorr: bool) -> Tuple[np.ndarray, np.ndarray]:
    pairs = []
    mus = []
    n_a = len(a)
    n_b = len(b)
    for i in range(n_a):
        j_start = i + 1 if autocorr else 0
        for j in range(j_start, n_b):
            vec = b[j] - a[i]
            s = np.linalg.norm(vec)
            if s == 0.0:
                continue
            mu = abs(vec[2]) / s
            pairs.append(s)
            mus.append(mu)
    return np.asarray(pairs, dtype="f4"), np.asarray(mus, dtype="f4")


def _histogram_pairs(sep: np.ndarray, mu: np.ndarray, s_edges: np.ndarray, mu_edges: np.ndarray) -> np.ndarray:
    h, _ = np.histogramdd((sep, mu), bins=(s_edges, mu_edges))
    return h


def compute_pair_counts(data_xyz: np.ndarray, rand_xyz: np.ndarray, s_edges: np.ndarray, mu_edges: np.ndarray) -> PairCounts:
    dd_sep, dd_mu = _pairwise_blocks(data_xyz, data_xyz, autocorr=True)
    dr_sep, dr_mu = _pairwise_blocks(data_xyz, rand_xyz, autocorr=False)
    rr_sep, rr_mu = _pairwise_blocks(rand_xyz, rand_xyz, autocorr=True)

    dd = _histogram_pairs(dd_sep, dd_mu, s_edges, mu_edges)
    dr = _histogram_pairs(dr_sep, dr_mu, s_edges, mu_edges)
    rr = _histogram_pairs(rr_sep, rr_mu, s_edges, mu_edges)

    return PairCounts(s_edges=s_edges, mu_edges=mu_edges, dd=dd, dr=dr, rr=rr, meta={"n_data": len(data_xyz)})


def landy_szalay(counts: PairCounts) -> np.ndarray:
    rr = np.where(counts.rr == 0, 1.0, counts.rr)
    return (counts.dd - 2.0 * counts.dr + counts.rr) / rr


def wedge_xi(xi: np.ndarray, mu_edges: np.ndarray, wedge: Tuple[float, float]) -> np.ndarray:
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    mask = (mu_centers >= wedge[0]) & (mu_centers < wedge[1])
    if not np.any(mask):
        raise ValueError("No mu bins in wedge")
    return np.mean(xi[:, mask], axis=1)


def bin_by_environment(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    return np.digitize(values, quantiles[1:-1], right=False)
