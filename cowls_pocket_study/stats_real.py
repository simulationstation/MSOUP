"""
Debiased clustering statistics for real COWLS data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy import stats as scipy_stats

from msoup_substructure_sim.stats import compute_pairwise_clustering
from msoup_substructure_sim.windows import sample_from_sensitivity

from .sensitivity import SensitivityWindowReal


@dataclass
class LensClustering:
    lens_id: str
    n_candidates: int
    C_obs: float
    C_null: float
    C_excess: float
    Z: float


@dataclass
class AggregateStats:
    lens_stats: List[LensClustering]
    global_Z: float
    global_pvalue: float
    sign_test_pvalue: float
    bootstrap_mean: float


def _null_distribution(
    window: SensitivityWindowReal,
    n_obs: int,
    theta0: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    C_samples = []
    for _ in range(n_resamples):
        theta_sample = sample_from_sensitivity(window, n_obs, rng)
        C_sample = compute_pairwise_clustering(theta_sample, theta0)
        C_samples.append(C_sample)
    return np.array(C_samples)


def per_lens_clustering(
    lens_id: str,
    theta: np.ndarray,
    window: SensitivityWindowReal,
    theta0: float,
    n_resamples: int,
    rng: np.random.Generator,
) -> Optional[LensClustering]:
    """Compute C_obs, C_excess, Z for a single lens."""
    if len(theta) < 2:
        return None
    C_obs = compute_pairwise_clustering(theta, theta0)
    C_samples = _null_distribution(window, len(theta), theta0, n_resamples, rng)
    C_null = float(np.mean(C_samples))
    sigma = float(np.std(C_samples) + 1e-9)
    C_excess = C_obs - C_null
    Z = (C_obs - C_null) / sigma
    return LensClustering(
        lens_id=lens_id,
        n_candidates=len(theta),
        C_obs=C_obs,
        C_null=C_null,
        C_excess=C_excess,
        Z=Z,
    )


def aggregate_clustering(
    theta_by_lens: Dict[str, np.ndarray],
    windows: Dict[str, SensitivityWindowReal],
    theta0: float = 0.3,
    n_resamples: int = 300,
    rng: Optional[np.random.Generator] = None,
) -> AggregateStats:
    """Aggregate debiased clustering across lenses."""
    rng = rng or np.random.default_rng(123)
    per_lens: List[LensClustering] = []
    for lens_id, theta in theta_by_lens.items():
        window = windows.get(lens_id)
        if window is None:
            continue
        result = per_lens_clustering(lens_id, theta, window, theta0, n_resamples, rng)
        if result:
            per_lens.append(result)

    if not per_lens:
        return AggregateStats([], global_Z=0.0, global_pvalue=1.0, sign_test_pvalue=1.0, bootstrap_mean=0.0)

    Z_vals = np.array([p.Z for p in per_lens])
    C_excess_vals = np.array([p.C_excess for p in per_lens])
    global_Z = float(np.mean(Z_vals))
    p_global = 1 - scipy_stats.norm.cdf(global_Z * np.sqrt(len(Z_vals)))
    sign_test = scipy_stats.binomtest(np.sum(C_excess_vals > 0), len(C_excess_vals), 0.5, alternative="greater")

    boot_means = []
    for _ in range(500):
        resample = rng.choice(C_excess_vals, size=len(C_excess_vals), replace=True)
        boot_means.append(np.mean(resample))
    return AggregateStats(
        lens_stats=per_lens,
        global_Z=global_Z,
        global_pvalue=float(p_global),
        sign_test_pvalue=float(sign_test.pvalue),
        bootstrap_mean=float(np.mean(boot_means)),
    )


def shuffle_control(
    theta_by_lens: Dict[str, np.ndarray],
    windows: Dict[str, SensitivityWindowReal],
    theta0: float,
    rng: Optional[np.random.Generator] = None,
    n_resamples: int = 200,
) -> float:
    """Randomly shuffle theta within each lens to verify C_excess≈0."""
    rng = rng or np.random.default_rng(321)
    shuffled_stats = []
    for lens_id, theta in theta_by_lens.items():
        window = windows[lens_id]
        shuffled_theta = sample_from_sensitivity(window, len(theta), rng)
        result = per_lens_clustering(lens_id, shuffled_theta, window, theta0, n_resamples, rng)
        if result:
            shuffled_stats.append(result.C_excess)
    return float(np.mean(shuffled_stats)) if shuffled_stats else 0.0


def window_stress_test(
    theta_by_lens: Dict[str, np.ndarray],
    windows: Dict[str, SensitivityWindowReal],
    theta0: float,
    sharpen: float = 2.0,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Sharpen S(theta) to test robustness of C_excess against window mis-modeling.
    """
    rng = rng or np.random.default_rng(456)
    inflated = []
    for lens_id, window in windows.items():
        s_sharp = np.power(window.s_grid, sharpen)
        s_sharp /= np.mean(s_sharp)
        sharp_window = SensitivityWindowReal(theta_grid=window.theta_grid, s_grid=s_sharp, lens_id=lens_id)
        theta = theta_by_lens.get(lens_id, np.array([]))
        if len(theta) < 2:
            continue
        result = per_lens_clustering(lens_id, theta, sharp_window, theta0, 200, rng)
        if result:
            inflated.append(result.C_obs)
    return float(np.nanmean(inflated)) if inflated else 0.0


def cox_rate_control(theta_counts: Sequence[int]) -> float:
    """
    Cox-process-only control: variation in counts should not create C_excess.
    """
    return float(np.std(theta_counts) / (np.mean(theta_counts) + 1e-6))
