"""Correlation function computation using TreeCorr for efficiency.

This module computes 2-point correlation functions ξ(s,μ) using the
Landy-Szalay estimator with TreeCorr for efficient pair counting.

Per preregistration: Uses TreeCorr (or Corrfunc) for pair counting
as specified in software requirements (Section 12.1).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, Tuple, Optional, Callable

import numpy as np
try:
    import treecorr
    HAS_TREECORR = True
except ImportError:  # pragma: no cover - handled via fallback
    HAS_TREECORR = False


# Resource limits to prevent system overload
MAX_MEMORY_GB = float(os.environ.get("BAO_MAX_MEMORY_GB", "8.0"))
TREECORR_NTHREADS = int(os.environ.get("BAO_NTHREADS", "4"))


def _check_memory():
    """Check available memory and warn if low."""
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    available_gb = int(line.split()[1]) / 1e6
                    if available_gb < 2.0:
                        raise MemoryError(
                            f"Only {available_gb:.1f}GB available. "
                            "Need at least 2GB for correlation computation."
                        )
                    return available_gb
    except FileNotFoundError:
        pass  # Non-Linux, assume OK
    return None


@dataclass
class PairCounts:
    """Container for pair count results."""
    s_edges: np.ndarray
    mu_edges: np.ndarray
    dd: np.ndarray  # Shape: (n_s_bins, n_mu_bins)
    dr: np.ndarray
    rr: np.ndarray
    meta: Dict[str, float]


def compute_pair_counts(
    data_xyz: np.ndarray,
    rand_xyz: np.ndarray,
    s_edges: np.ndarray,
    mu_edges: np.ndarray,
    data_weights: Optional[np.ndarray] = None,
    rand_weights: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> PairCounts:
    """
    Compute DD, DR, RR pair counts using TreeCorr.

    Parameters
    ----------
    data_xyz : np.ndarray
        Data positions in Cartesian coordinates, shape (N_data, 3).
    rand_xyz : np.ndarray
        Random positions in Cartesian coordinates, shape (N_rand, 3).
    s_edges : np.ndarray
        Separation bin edges in h^-1 Mpc.
    mu_edges : np.ndarray
        mu = |cos(theta)| bin edges, from 0 to 1.
    data_weights : np.ndarray, optional
        Weights for data points.
    rand_weights : np.ndarray, optional
        Weights for random points.
    verbose : bool
        Print progress information.

    Returns
    -------
    PairCounts
        Container with DD, DR, RR pair count arrays.
    """
    _check_memory()

    n_data = len(data_xyz)
    n_rand = len(rand_xyz)
    n_s_bins = len(s_edges) - 1
    n_mu_bins = len(mu_edges) - 1

    if verbose:
        print(f"Computing pair counts: {n_data:,} data × {n_rand:,} randoms")
        print(f"  s bins: {n_s_bins}, mu bins: {n_mu_bins}")

    # Default weights
    if data_weights is None:
        data_weights = np.ones(n_data)
    if rand_weights is None:
        rand_weights = np.ones(n_rand)

    data_weights = np.asarray(data_weights, dtype="f8")
    rand_weights = np.asarray(rand_weights, dtype="f8")

    if not HAS_TREECORR:
        return compute_pair_counts_bruteforce(
            data_xyz,
            rand_xyz,
            s_edges=s_edges,
            mu_edges=mu_edges,
            data_weights=data_weights,
            rand_weights=rand_weights,
        )

    # Create TreeCorr catalogs
    data_cat = treecorr.Catalog(
        x=data_xyz[:, 0],
        y=data_xyz[:, 1],
        z=data_xyz[:, 2],
        w=data_weights,
    )
    rand_cat = treecorr.Catalog(
        x=rand_xyz[:, 0],
        y=rand_xyz[:, 1],
        z=rand_xyz[:, 2],
        w=rand_weights,
    )

    # TreeCorr configuration for 3D correlation
    # We compute in (rp, pi) space and transform to (s, mu)
    s_min = s_edges[0]
    s_max = s_edges[-1]

    # For (s, mu) binning, we use NNCorrelation in 3D mode
    # TreeCorr's 3D mode gives r (separation), which is our s
    config = {
        'min_sep': s_min,
        'max_sep': s_max,
        'nbins': n_s_bins,
        'bin_type': 'Linear',
        'num_threads': TREECORR_NTHREADS,
        'verbose': 1 if verbose else 0,
    }

    # For proper (s, mu) binning, we need to use rp/pi decomposition
    # and then convert. TreeCorr's NNCorrelation gives scalar separation.
    # For 2D ξ(s,μ), we need to compute in (rp, pi) then transform.

    # Use 2D correlation in (rp, pi) space
    rp_bins = n_s_bins
    pi_max = s_max  # Maximum line-of-sight separation

    config_2d = {
        'min_rpar': -pi_max,
        'max_rpar': pi_max,
        'nbins_rpar': n_s_bins,  # pi bins (symmetric around 0)
        'min_sep': 0.1,  # Minimum rp
        'max_sep': s_max,
        'nbins': n_s_bins,  # rp bins
        'bin_type': 'Linear',
        'num_threads': TREECORR_NTHREADS,
        'verbose': 1 if verbose else 0,
    }

    if verbose:
        print("  Computing DD pairs...")
    dd_corr = treecorr.NNCorrelation(**config_2d)
    dd_corr.process(data_cat)

    if verbose:
        print("  Computing DR pairs...")
    dr_corr = treecorr.NNCorrelation(**config_2d)
    dr_corr.process(data_cat, rand_cat)

    if verbose:
        print("  Computing RR pairs...")
    rr_corr = treecorr.NNCorrelation(**config_2d)
    rr_corr.process(rand_cat)

    # Get pair counts from TreeCorr
    # TreeCorr stores weighted pair counts in npairs
    # The 2D correlation gives us rp/pi binning

    # For now, use 1D s-binning and create mu-binned version
    # by running separate correlations (simplified approach)

    # Actually, TreeCorr's 3D mode with rpar gives us (rp, pi) which we can
    # transform to (s, mu). Let's extract the raw counts.

    # TreeCorr stores results in 2D arrays when using rpar binning
    # Shape: (nbins, nbins_rpar) = (n_rp, n_pi)

    # Transform (rp, pi) to (s, mu):
    # s = sqrt(rp^2 + pi^2)
    # mu = |pi| / s

    # Get bin centers
    rp_centers = dd_corr.rnom  # rp bin centers

    # For rpar (pi) bins, we need to get the centers
    # TreeCorr uses linear spacing in rpar
    pi_edges = np.linspace(-pi_max, pi_max, n_s_bins + 1)
    pi_centers = 0.5 * (pi_edges[:-1] + pi_edges[1:])

    # Create meshgrid for transformation
    rp_grid, pi_grid = np.meshgrid(rp_centers, pi_centers, indexing='ij')
    s_grid = np.sqrt(rp_grid**2 + pi_grid**2)
    mu_grid = np.abs(pi_grid) / np.maximum(s_grid, 1e-10)

    # Get pair counts arrays
    # TreeCorr's 2D correlation stores in (rp, pi) shape
    if hasattr(dd_corr, 'npairs') and dd_corr.npairs.ndim == 2:
        dd_rp_pi = dd_corr.npairs
        dr_rp_pi = dr_corr.npairs
        rr_rp_pi = rr_corr.npairs
    else:
        # Fallback: 1D binning, tile to create 2D
        dd_rp_pi = np.tile(dd_corr.npairs[:, np.newaxis], (1, n_mu_bins))
        dr_rp_pi = np.tile(dr_corr.npairs[:, np.newaxis], (1, n_mu_bins))
        rr_rp_pi = np.tile(rr_corr.npairs[:, np.newaxis], (1, n_mu_bins))

    # Rebin from (rp, pi) to (s, mu)
    dd_s_mu = _rebin_rp_pi_to_s_mu(
        dd_rp_pi, rp_centers, pi_centers, s_edges, mu_edges
    )
    dr_s_mu = _rebin_rp_pi_to_s_mu(
        dr_rp_pi, rp_centers, pi_centers, s_edges, mu_edges
    )
    rr_s_mu = _rebin_rp_pi_to_s_mu(
        rr_rp_pi, rp_centers, pi_centers, s_edges, mu_edges
    )

    meta = {
        "n_data": n_data,
        "n_rand": n_rand,
        "sum_w_data": float(data_weights.sum()),
        "sum_w_rand": float(rand_weights.sum()),
        "sum_w_data_sq": float(np.square(data_weights).sum()),
        "sum_w_rand_sq": float(np.square(rand_weights).sum()),
    }

    if verbose:
        print(f"  Pair counting complete.")

    return PairCounts(
        s_edges=s_edges,
        mu_edges=mu_edges,
        dd=dd_s_mu,
        dr=dr_s_mu,
        rr=rr_s_mu,
        meta=meta,
    )


def _empty_pair_counts(s_edges: np.ndarray, mu_edges: np.ndarray) -> PairCounts:
    n_s = len(s_edges) - 1
    n_mu = len(mu_edges) - 1
    return PairCounts(
        s_edges=s_edges,
        mu_edges=mu_edges,
        dd=np.zeros((n_s, n_mu)),
        dr=np.zeros((n_s, n_mu)),
        rr=np.zeros((n_s, n_mu)),
        meta={
            "n_data": 0,
            "n_rand": 0,
            "sum_w_data": 0.0,
            "sum_w_rand": 0.0,
            "sum_w_data_sq": 0.0,
            "sum_w_rand_sq": 0.0,
        },
    )


def compute_pair_counts_by_environment(
    data_xyz: np.ndarray,
    rand_xyz: np.ndarray,
    data_bins: np.ndarray,
    rand_bins: np.ndarray,
    n_bins: int,
    s_edges: np.ndarray,
    mu_edges: np.ndarray,
    data_weights: Optional[np.ndarray] = None,
    rand_weights: Optional[np.ndarray] = None,
    verbose: bool = True,
    pair_counter: Callable[..., PairCounts] = compute_pair_counts,
) -> Dict[int, PairCounts]:
    """Compute pair counts per environment bin."""
    counts_by_bin: Dict[int, PairCounts] = {}
    for b in range(n_bins):
        data_mask = data_bins == b
        rand_mask = rand_bins == b
        if not np.any(data_mask) or not np.any(rand_mask):
            counts_by_bin[b] = _empty_pair_counts(s_edges, mu_edges)
            continue
        counts_by_bin[b] = pair_counter(
            data_xyz[data_mask],
            rand_xyz[rand_mask],
            s_edges=s_edges,
            mu_edges=mu_edges,
            data_weights=None if data_weights is None else data_weights[data_mask],
            rand_weights=None if rand_weights is None else rand_weights[rand_mask],
            verbose=verbose,
        )
    return counts_by_bin


def _rebin_rp_pi_to_s_mu(
    counts_rp_pi: np.ndarray,
    rp_centers: np.ndarray,
    pi_centers: np.ndarray,
    s_edges: np.ndarray,
    mu_edges: np.ndarray,
) -> np.ndarray:
    """
    Rebin pair counts from (rp, pi) grid to (s, mu) grid.

    Parameters
    ----------
    counts_rp_pi : np.ndarray
        Pair counts in (rp, pi) binning.
    rp_centers : np.ndarray
        Centers of rp bins.
    pi_centers : np.ndarray
        Centers of pi bins.
    s_edges : np.ndarray
        Target s bin edges.
    mu_edges : np.ndarray
        Target mu bin edges.

    Returns
    -------
    np.ndarray
        Pair counts rebinned to (s, mu), shape (n_s, n_mu).
    """
    n_s = len(s_edges) - 1
    n_mu = len(mu_edges) - 1

    counts_s_mu = np.zeros((n_s, n_mu))

    # Create grid of (rp, pi) -> (s, mu)
    for i, rp in enumerate(rp_centers):
        for j, pi in enumerate(pi_centers):
            s = np.sqrt(rp**2 + pi**2)
            mu = abs(pi) / max(s, 1e-10)

            # Find target bins
            s_idx = np.searchsorted(s_edges, s) - 1
            mu_idx = np.searchsorted(mu_edges, mu) - 1

            if 0 <= s_idx < n_s and 0 <= mu_idx < n_mu:
                if i < counts_rp_pi.shape[0] and j < counts_rp_pi.shape[1]:
                    counts_s_mu[s_idx, mu_idx] += counts_rp_pi[i, j]

    return counts_s_mu


def compute_pair_counts_simple(
    data_xyz: np.ndarray,
    rand_xyz: np.ndarray,
    s_edges: np.ndarray,
    mu_edges: np.ndarray,
    data_weights: Optional[np.ndarray] = None,
    rand_weights: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> PairCounts:
    """
    Simplified pair counting using 1D TreeCorr and uniform mu distribution.

    This is faster and suitable for initial analysis. For full 2D ξ(s,μ),
    use compute_pair_counts().
    """
    _check_memory()

    if not HAS_TREECORR:
        return compute_pair_counts_bruteforce(
            data_xyz,
            rand_xyz,
            s_edges=s_edges,
            mu_edges=mu_edges,
            data_weights=data_weights,
            rand_weights=rand_weights,
        )

    n_data = len(data_xyz)
    n_rand = len(rand_xyz)
    n_s_bins = len(s_edges) - 1
    n_mu_bins = len(mu_edges) - 1

    if verbose:
        print(f"Computing 1D pair counts: {n_data:,} data × {n_rand:,} randoms")

    if data_weights is None:
        data_weights = np.ones(n_data)
    if rand_weights is None:
        rand_weights = np.ones(n_rand)

    data_cat = treecorr.Catalog(
        x=data_xyz[:, 0], y=data_xyz[:, 1], z=data_xyz[:, 2], w=data_weights
    )
    rand_cat = treecorr.Catalog(
        x=rand_xyz[:, 0], y=rand_xyz[:, 1], z=rand_xyz[:, 2], w=rand_weights
    )

    config = {
        'min_sep': s_edges[0],
        'max_sep': s_edges[-1],
        'nbins': n_s_bins,
        'bin_type': 'Linear',
        'num_threads': TREECORR_NTHREADS,
        'verbose': 1 if verbose else 0,
    }

    if verbose:
        print("  DD...")
    dd = treecorr.NNCorrelation(**config)
    dd.process(data_cat)

    if verbose:
        print("  DR...")
    dr = treecorr.NNCorrelation(**config)
    dr.process(data_cat, rand_cat)

    if verbose:
        print("  RR...")
    rr = treecorr.NNCorrelation(**config)
    rr.process(rand_cat)

    # Create 2D arrays assuming uniform mu distribution
    dd_2d = np.tile(dd.npairs[:, np.newaxis], (1, n_mu_bins)) / n_mu_bins
    dr_2d = np.tile(dr.npairs[:, np.newaxis], (1, n_mu_bins)) / n_mu_bins
    rr_2d = np.tile(rr.npairs[:, np.newaxis], (1, n_mu_bins)) / n_mu_bins

    return PairCounts(
        s_edges=s_edges,
        mu_edges=mu_edges,
        dd=dd_2d,
        dr=dr_2d,
        rr=rr_2d,
        meta={
            "n_data": n_data,
            "n_rand": n_rand,
            "sum_w_data": float(data_weights.sum()),
            "sum_w_rand": float(rand_weights.sum()),
            "sum_w_data_sq": float(np.square(data_weights).sum()),
            "sum_w_rand_sq": float(np.square(rand_weights).sum()),
            "mode": "1D_uniform_mu",
        },
    )


def compute_pair_counts_bruteforce(
    data_xyz: np.ndarray,
    rand_xyz: np.ndarray,
    s_edges: np.ndarray,
    mu_edges: np.ndarray,
    data_weights: Optional[np.ndarray] = None,
    rand_weights: Optional[np.ndarray] = None,
) -> PairCounts:
    """Brute-force pair counting fallback for small samples."""
    n_data = len(data_xyz)
    n_rand = len(rand_xyz)
    n_s_bins = len(s_edges) - 1
    n_mu_bins = len(mu_edges) - 1

    if data_weights is None:
        data_weights = np.ones(n_data)
    if rand_weights is None:
        rand_weights = np.ones(n_rand)

    dd = np.zeros((n_s_bins, n_mu_bins))
    dr = np.zeros((n_s_bins, n_mu_bins))
    rr = np.zeros((n_s_bins, n_mu_bins))

    def _accumulate(xyz_a, xyz_b, w_a, w_b, counts, symmetric: bool = False) -> None:
        for i in range(len(xyz_a)):
            j_start = i + 1 if symmetric else 0
            for j in range(j_start, len(xyz_b)):
                diff = xyz_b[j] - xyz_a[i]
                s = np.linalg.norm(diff)
                if s == 0.0:
                    continue
                mu = abs(diff[2]) / s
                s_idx = np.searchsorted(s_edges, s) - 1
                mu_idx = np.searchsorted(mu_edges, mu) - 1
                if 0 <= s_idx < n_s_bins and 0 <= mu_idx < n_mu_bins:
                    counts[s_idx, mu_idx] += w_a[i] * w_b[j]

    _accumulate(data_xyz, data_xyz, data_weights, data_weights, dd, symmetric=True)
    _accumulate(data_xyz, rand_xyz, data_weights, rand_weights, dr, symmetric=False)
    _accumulate(rand_xyz, rand_xyz, rand_weights, rand_weights, rr, symmetric=True)

    meta = {
        "n_data": n_data,
        "n_rand": n_rand,
        "sum_w_data": float(np.sum(data_weights)),
        "sum_w_rand": float(np.sum(rand_weights)),
        "sum_w_data_sq": float(np.sum(np.square(data_weights))),
        "sum_w_rand_sq": float(np.sum(np.square(rand_weights))),
        "mode": "bruteforce",
    }

    return PairCounts(
        s_edges=s_edges,
        mu_edges=mu_edges,
        dd=dd,
        dr=dr,
        rr=rr,
        meta=meta,
    )


def landy_szalay(counts: PairCounts) -> np.ndarray:
    """
    Compute Landy-Szalay correlation function estimator.

    ξ = (DD - 2DR + RR) / RR

    Normalized by the total weighted pair counts.
    """
    # Get normalization factors (weighted pair counts)
    sum_w_data = counts.meta.get("sum_w_data", counts.meta.get("n_data", 1))
    sum_w_rand = counts.meta.get("sum_w_rand", counts.meta.get("n_rand", 1))
    sum_w_data_sq = counts.meta.get("sum_w_data_sq", sum_w_data)
    sum_w_rand_sq = counts.meta.get("sum_w_rand_sq", sum_w_rand)

    norm_dd = max((sum_w_data**2 - sum_w_data_sq) / 2.0, 1.0)
    norm_rr = max((sum_w_rand**2 - sum_w_rand_sq) / 2.0, 1.0)
    norm_dr = max(sum_w_data * sum_w_rand, 1.0)

    dd_norm = counts.dd / norm_dd
    dr_norm = counts.dr / norm_dr
    rr_norm = counts.rr / norm_rr

    # Landy-Szalay estimator
    rr_safe = np.where(rr_norm > 0, rr_norm, 1.0)
    xi = (dd_norm - 2.0 * dr_norm + rr_norm) / rr_safe
    xi = np.where(rr_norm > 0, xi, 0.0)

    return xi


def wedge_xi(xi: np.ndarray, mu_edges: np.ndarray, wedge: Tuple[float, float]) -> np.ndarray:
    """
    Extract wedge-averaged correlation function.

    Parameters
    ----------
    xi : np.ndarray
        2D correlation function ξ(s, μ), shape (n_s, n_mu).
    mu_edges : np.ndarray
        Edges of mu bins.
    wedge : Tuple[float, float]
        (mu_min, mu_max) defining the wedge.

    Returns
    -------
    np.ndarray
        Wedge-averaged ξ(s), shape (n_s,).
    """
    mu_centers = 0.5 * (mu_edges[:-1] + mu_edges[1:])
    mask = (mu_centers >= wedge[0]) & (mu_centers < wedge[1])

    if not np.any(mask):
        raise ValueError(f"No mu bins in wedge [{wedge[0]}, {wedge[1]}]")

    return np.mean(xi[:, mask], axis=1)


def parse_wedge_bounds(wedge_cfg: Dict[str, float]) -> Tuple[float, float]:
    """Return numeric wedge bounds from config mapping."""
    mu_min = wedge_cfg.get("mu_min")
    mu_max = wedge_cfg.get("mu_max")
    if not isinstance(mu_min, (int, float)) or not isinstance(mu_max, (int, float)):
        raise ValueError("Wedge bounds must be numeric.")
    return float(mu_min), float(mu_max)


def combine_pair_counts(counts_list: Iterable[PairCounts]) -> PairCounts:
    """Combine pair counts across regions by summing weighted pairs."""
    counts_list = list(counts_list)
    if not counts_list:
        raise ValueError("No pair counts provided for combination.")

    base = counts_list[0]
    dd = np.zeros_like(base.dd)
    dr = np.zeros_like(base.dr)
    rr = np.zeros_like(base.rr)
    meta = {"n_data": 0, "n_rand": 0, "sum_w_data": 0.0, "sum_w_rand": 0.0, "sum_w_data_sq": 0.0, "sum_w_rand_sq": 0.0}

    for counts in counts_list:
        if not np.allclose(counts.s_edges, base.s_edges) or not np.allclose(counts.mu_edges, base.mu_edges):
            raise ValueError("Cannot combine pair counts with different binning.")
        dd += counts.dd
        dr += counts.dr
        rr += counts.rr
        meta["n_data"] += int(counts.meta.get("n_data", 0))
        meta["n_rand"] += int(counts.meta.get("n_rand", 0))
        meta["sum_w_data"] += float(counts.meta.get("sum_w_data", 0.0))
        meta["sum_w_rand"] += float(counts.meta.get("sum_w_rand", 0.0))
        meta["sum_w_data_sq"] += float(counts.meta.get("sum_w_data_sq", 0.0))
        meta["sum_w_rand_sq"] += float(counts.meta.get("sum_w_rand_sq", 0.0))

    return PairCounts(
        s_edges=base.s_edges,
        mu_edges=base.mu_edges,
        dd=dd,
        dr=dr,
        rr=rr,
        meta=meta,
    )


def bin_by_environment(values: np.ndarray, quantiles: np.ndarray) -> np.ndarray:
    """
    Assign environment bin indices based on quantile edges.

    Parameters
    ----------
    values : np.ndarray
        Environment values per galaxy.
    quantiles : np.ndarray
        Quantile edges, e.g., [0, 0.25, 0.5, 0.75, 1.0] for quartiles.

    Returns
    -------
    np.ndarray
        Bin indices (0 to n_bins-1) for each galaxy.
    """
    edges = np.quantile(values, quantiles)
    return np.digitize(values, edges[1:-1], right=False)
