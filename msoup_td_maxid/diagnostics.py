from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np


def logsumexp(arr: np.ndarray) -> float:
    a_max = np.max(arr)
    if not np.isfinite(a_max):
        return -np.inf
    return float(a_max + np.log(np.sum(np.exp(arr - a_max))))


def _approx_pareto_k(weights: np.ndarray) -> float:
    finite = weights[np.isfinite(weights) & (weights > 0)]
    if finite.size < 5:
        return math.nan
    sorted_w = np.sort(finite)[::-1]
    m = max(5, int(0.2 * sorted_w.size))
    tail = sorted_w[:m]
    threshold = tail[-1]
    if threshold <= 0:
        return math.nan
    logs = np.log(tail / threshold)
    k_hat = float(np.mean(logs))
    return max(0.0, k_hat)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    order = np.argsort(x)
    x_sorted = x[order]
    w_sorted = w[order]
    cum = np.cumsum(w_sorted) / np.sum(w_sorted)
    idx = np.searchsorted(cum, 0.5)
    return float(x_sorted[min(idx, len(x_sorted) - 1)])


def info_fractions(loglike_matrix: np.ndarray) -> List[float]:
    total = np.sum(loglike_matrix, axis=0)
    full = logsumexp(total)
    deltas = []
    for i in range(loglike_matrix.shape[0]):
        loo = logsumexp(total - loglike_matrix[i])
        deltas.append(max(0.0, full - loo))
    denom = sum(deltas) if sum(deltas) > 0 else 1.0
    return [d / denom for d in deltas]


@dataclass
class LOOEntry:
    lens_id: str
    pareto_k: float
    delta_m_shift: float
    median_loo: float
    median_full: float


def loo_diagnostics(lens_ids: Sequence[str], delta_grid: np.ndarray, loglike_matrix: np.ndarray, weights: np.ndarray) -> List[LOOEntry]:
    total = np.sum(loglike_matrix, axis=0)
    full_med = weighted_median(delta_grid, weights)
    entries: List[LOOEntry] = []
    for i, lens_id in enumerate(lens_ids):
        loo_logw = total - loglike_matrix[i]
        norm = logsumexp(loo_logw)
        loo_weights = np.exp(loo_logw - norm)
        loo_weights /= loo_weights.sum()
        pareto = _approx_pareto_k(loo_weights)
        median_loo = weighted_median(delta_grid, loo_weights)
        entries.append(
            LOOEntry(
                lens_id=lens_id,
                pareto_k=pareto,
                delta_m_shift=median_loo - full_med,
                median_loo=median_loo,
                median_full=full_med,
            )
        )
    return entries


@dataclass
class PPCEntry:
    lens_id: str
    residual: float
    sigma: float
    pull: float
    support: str


def dominance_kill_table(info_fracs: List[float], loo_entries: List[LOOEntry], ppc: List[PPCEntry], sigma_ref: float) -> Dict[str, bool]:
    max_info = max(info_fracs) if info_fracs else 0.0
    pareto_ok = all((entry.pareto_k < 0.7) or math.isnan(entry.pareto_k) for entry in loo_entries)
    sigma_ref = sigma_ref if np.isfinite(sigma_ref) else 1.0
    loo_ok = all(abs(entry.delta_m_shift) < sigma_ref for entry in loo_entries)
    ppc_ok = all(abs(entry.pull) < 3.0 for entry in ppc)
    return {
        "K-DOM": max_info < 0.30,
        "K-PSIS": pareto_ok,
        "K-LOO": loo_ok,
        "K-PPC": ppc_ok,
    }
