"""
Statistical Analysis for Msoup Substructure Simulation

Computes:
1. Fano factor (variance/mean) - tests over-dispersion
2. Tail probabilities - tests heavy tails
3. Residual dispersion after Poisson regression - tests if host explains variance
4. Spatial clustering metric - tests angular clustering on the ring
5. Simple classifier to discriminate models
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats as scipy_stats

from .simulate import SimulationResult


@dataclass
class StatsSummary:
    """Summary statistics for a simulation result."""
    model_name: str
    n_lenses: int

    # Count statistics
    mean_count: float
    var_count: float
    fano_factor: float
    fano_se: float  # Standard error of Fano factor

    # Tail probability
    tail_threshold: float
    tail_prob: float
    tail_prob_poisson: float  # Expected under Poisson

    # Residual dispersion after regression
    residual_dispersion: float

    # Spatial clustering metric
    clustering_C: float
    clustering_C_se: float

    # Per-lens clustering values (for distribution)
    clustering_values: np.ndarray


def circular_distance(theta1: float, theta2: float) -> float:
    """Compute circular distance on [0, 2π)."""
    diff = abs(theta1 - theta2)
    return min(diff, 2 * np.pi - diff)


def compute_pairwise_clustering(theta: np.ndarray, theta0: float) -> float:
    """
    Compute fraction of pairs with angular separation < theta0.

    Parameters
    ----------
    theta : array
        Angular positions of perturbers in [0, 2π)
    theta0 : float
        Threshold for "close" pairs (radians)

    Returns
    -------
    float
        Fraction of pairs that are close (clustering metric C)
    """
    n = len(theta)
    if n < 2:
        return np.nan

    n_pairs = n * (n - 1) // 2
    n_close = 0

    for i in range(n):
        for j in range(i + 1, n):
            dist = circular_distance(theta[i], theta[j])
            if dist < theta0:
                n_close += 1

    return n_close / n_pairs


def compute_clustering_metric(theta_list: List[np.ndarray],
                               theta0: float = 0.3) -> Tuple[float, float, np.ndarray]:
    """
    Compute spatial clustering metric across all lenses.

    Parameters
    ----------
    theta_list : list of arrays
        Angular positions for each lens
    theta0 : float
        Threshold for close pairs

    Returns
    -------
    C_mean : float
        Mean clustering across lenses (excluding N < 2)
    C_se : float
        Standard error of mean
    C_values : array
        Per-lens clustering values
    """
    C_values = []
    for theta in theta_list:
        if len(theta) >= 2:
            C = compute_pairwise_clustering(theta, theta0)
            C_values.append(C)

    C_values = np.array(C_values)

    if len(C_values) > 0:
        C_mean = np.mean(C_values)
        C_se = np.std(C_values) / np.sqrt(len(C_values))
    else:
        C_mean = np.nan
        C_se = np.nan

    return C_mean, C_se, C_values


def compute_poisson_regression_dispersion(counts: np.ndarray,
                                           H: np.ndarray,
                                           z: np.ndarray) -> float:
    """
    Fit a Poisson regression and compute residual dispersion.

    For true Poisson data, dispersion ~ 1.
    For over-dispersed data, dispersion > 1.

    Parameters
    ----------
    counts : array
        Perturber counts
    H : array
        Host mass proxy
    z : array
        Redshift proxy

    Returns
    -------
    float
        Dispersion parameter (quasi-Poisson)
    """
    # Simple approach: compute Pearson residuals under fitted Poisson model
    # Fit log-linear model: log E[N] = a + b*H + c*z
    # Use weighted least squares on log(N+0.5) as approximation

    # Avoid log(0)
    log_counts = np.log(counts + 0.5)

    # Design matrix
    X = np.column_stack([np.ones(len(counts)), H, z])

    try:
        # Fit linear model
        beta, residuals, rank, s = np.linalg.lstsq(X, log_counts, rcond=None)

        # Predicted means
        log_mu = X @ beta
        mu = np.exp(log_mu)
        mu = np.maximum(mu, 0.1)  # Floor to avoid division issues

        # Pearson residuals: (y - mu) / sqrt(mu)
        pearson_resid = (counts - mu) / np.sqrt(mu)

        # Dispersion = sum of squared Pearson residuals / (n - p)
        n = len(counts)
        p = 3  # number of parameters
        dispersion = np.sum(pearson_resid**2) / (n - p)

    except Exception:
        dispersion = np.nan

    return dispersion


def compute_tail_probability(counts: np.ndarray,
                              k: float = 3.0) -> Tuple[float, float, float]:
    """
    Compute tail probability P(N >= threshold).

    Parameters
    ----------
    counts : array
        Perturber counts
    k : float
        Number of standard deviations for threshold

    Returns
    -------
    threshold : float
    tail_prob : float
    tail_prob_poisson : float
        Expected tail probability under Poisson with same mean
    """
    mean = np.mean(counts)
    threshold = mean + k * np.sqrt(mean)

    tail_prob = np.mean(counts >= threshold)

    # Under Poisson with this mean, what's the expected tail prob?
    from scipy.stats import poisson
    tail_prob_poisson = 1 - poisson.cdf(threshold - 1, mean)

    return threshold, tail_prob, tail_prob_poisson


def compute_fano_se(counts: np.ndarray, n_bootstrap: int = 100,
                    rng: Optional[np.random.Generator] = None) -> float:
    """
    Compute standard error of Fano factor via bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    n = len(counts)
    fano_boot = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        counts_boot = counts[idx]
        mean = np.mean(counts_boot)
        var = np.var(counts_boot)
        if mean > 0:
            fano_boot.append(var / mean)

    return np.std(fano_boot) if fano_boot else np.nan


def analyze_result(result: SimulationResult,
                   theta0: float = 0.3,
                   tail_k: float = 3.0,
                   n_bootstrap: int = 100,
                   rng: Optional[np.random.Generator] = None) -> StatsSummary:
    """
    Compute all statistics for a simulation result.

    Parameters
    ----------
    result : SimulationResult
    theta0 : float
        Clustering threshold
    tail_k : float
        Tail threshold in sigma
    n_bootstrap : int
        Bootstrap samples for SE

    Returns
    -------
    StatsSummary
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Fano factor
    fano_se = compute_fano_se(result.counts, n_bootstrap, rng)

    # Tail probability
    tail_thresh, tail_prob, tail_prob_pois = compute_tail_probability(
        result.counts, tail_k)

    # Residual dispersion
    resid_disp = compute_poisson_regression_dispersion(
        result.counts, result.H, result.z)

    # Clustering
    C_mean, C_se, C_values = compute_clustering_metric(result.theta_list, theta0)

    return StatsSummary(
        model_name=result.model_name,
        n_lenses=result.n_lenses,
        mean_count=result.mean_count,
        var_count=result.var_count,
        fano_factor=result.fano_factor,
        fano_se=fano_se,
        tail_threshold=tail_thresh,
        tail_prob=tail_prob,
        tail_prob_poisson=tail_prob_pois,
        residual_dispersion=resid_disp,
        clustering_C=C_mean,
        clustering_C_se=C_se,
        clustering_values=C_values
    )


def discriminate_models(stats1: StatsSummary, stats2: StatsSummary,
                         use_clustering: bool = True) -> Dict:
    """
    Simple discrimination between two models using summary statistics.

    Returns
    -------
    Dict with discrimination metrics
    """
    results = {}

    # Fano factor comparison
    fano_diff = stats2.fano_factor - stats1.fano_factor
    fano_se_combined = np.sqrt(stats1.fano_se**2 + stats2.fano_se**2)
    fano_z = fano_diff / fano_se_combined if fano_se_combined > 0 else 0
    results['fano_diff'] = fano_diff
    results['fano_z'] = fano_z
    results['fano_pvalue'] = 2 * (1 - scipy_stats.norm.cdf(abs(fano_z)))

    # Tail probability comparison
    tail_diff = stats2.tail_prob - stats1.tail_prob
    results['tail_diff'] = tail_diff

    # Dispersion comparison
    disp_diff = stats2.residual_dispersion - stats1.residual_dispersion
    results['dispersion_diff'] = disp_diff

    # Clustering comparison
    if use_clustering and not np.isnan(stats1.clustering_C) and not np.isnan(stats2.clustering_C):
        C_diff = stats2.clustering_C - stats1.clustering_C
        C_se_combined = np.sqrt(stats1.clustering_C_se**2 + stats2.clustering_C_se**2)
        C_z = C_diff / C_se_combined if C_se_combined > 0 else 0
        results['clustering_diff'] = C_diff
        results['clustering_z'] = C_z
        results['clustering_pvalue'] = 2 * (1 - scipy_stats.norm.cdf(abs(C_z)))
    else:
        results['clustering_diff'] = np.nan
        results['clustering_z'] = np.nan
        results['clustering_pvalue'] = np.nan

    return results


def compute_detectability(poisson_stats: StatsSummary,
                           alt_stats: StatsSummary,
                           alpha: float = 0.05) -> Dict:
    """
    Compute whether the alternative model is detectable at significance level alpha.

    Returns
    -------
    Dict with detectability metrics for each statistic
    """
    disc = discriminate_models(poisson_stats, alt_stats)

    results = {
        'fano_detectable': disc['fano_pvalue'] < alpha if not np.isnan(disc['fano_pvalue']) else False,
        'clustering_detectable': disc['clustering_pvalue'] < alpha if not np.isnan(disc.get('clustering_pvalue', np.nan)) else False,
        'fano_pvalue': disc['fano_pvalue'],
        'clustering_pvalue': disc.get('clustering_pvalue', np.nan),
    }

    return results


def expected_uniform_clustering(theta0: float) -> float:
    """
    Expected clustering metric C for uniformly distributed points.

    For uniform distribution on [0, 2π), the probability that two points
    are within theta0 of each other (circular) is theta0 / π.
    """
    return theta0 / np.pi


def format_stats_table(stats_list: List[StatsSummary]) -> str:
    """Format statistics as a markdown table."""
    lines = [
        "| Model | N_lenses | Mean | Fano | Tail Prob | Dispersion | Clustering C |",
        "|-------|----------|------|------|-----------|------------|--------------|",
    ]

    for s in stats_list:
        lines.append(
            f"| {s.model_name} | {s.n_lenses} | {s.mean_count:.2f} | "
            f"{s.fano_factor:.3f}±{s.fano_se:.3f} | {s.tail_prob:.4f} | "
            f"{s.residual_dispersion:.3f} | {s.clustering_C:.4f}±{s.clustering_C_se:.4f} |"
        )

    return "\n".join(lines)


# ============================================================================
# DEBIASED CLUSTERING STATISTICS
# ============================================================================

@dataclass
class DebiasedClusteringStats:
    """Debiased clustering statistics that account for selection effects."""
    model_name: str
    n_lenses: int
    n_with_pairs: int  # Lenses with >= 2 perturbers

    # Raw clustering
    C_obs_mean: float
    C_obs_se: float

    # Debiased metrics
    C_excess_mean: float  # Mean of (C_obs - C_null)
    C_excess_se: float
    Z_mean: float  # Mean of (C_obs - C_null) / sigma_null
    Z_se: float

    # Per-lens values (for distribution plots)
    C_obs_values: np.ndarray
    C_null_values: np.ndarray
    C_excess_values: np.ndarray
    Z_values: np.ndarray

    # Global p-value under null
    global_pvalue: float


def compute_debiased_clustering(theta_list: List[np.ndarray],
                                 windows: List,
                                 theta0: float = 0.3,
                                 n_resamples: int = 200,
                                 rng: Optional[np.random.Generator] = None
                                 ) -> DebiasedClusteringStats:
    """
    Compute debiased clustering statistics.

    For each lens, we compute:
    - C_obs: observed clustering
    - C_null: expected clustering under null (uniform + selection)
    - C_excess = C_obs - C_null
    - Z = (C_obs - C_null) / sigma_null

    Parameters
    ----------
    theta_list : list of arrays
        Observed angular positions for each lens
    windows : list of SensitivityWindow
        Sensitivity functions for each lens
    theta0 : float
        Clustering threshold
    n_resamples : int
        Monte Carlo resamples for null distribution
    rng : Generator

    Returns
    -------
    DebiasedClusteringStats
    """
    if rng is None:
        rng = np.random.default_rng(42)

    from .windows import sample_from_sensitivity

    C_obs_list = []
    C_null_list = []
    C_excess_list = []
    Z_list = []

    for i, (theta, window) in enumerate(zip(theta_list, windows)):
        n_obs = len(theta)

        if n_obs < 2:
            continue

        # Observed clustering
        C_obs = compute_pairwise_clustering(theta, theta0)

        # Null distribution via Monte Carlo resampling
        C_samples = []
        for _ in range(n_resamples):
            # Sample from sensitivity distribution
            theta_sample = sample_from_sensitivity(window, n_obs, rng)
            C_sample = compute_pairwise_clustering(theta_sample, theta0)
            C_samples.append(C_sample)

        C_samples = np.array(C_samples)
        C_null_mean = np.mean(C_samples)
        C_null_std = np.std(C_samples)

        # Debiased metrics
        C_excess = C_obs - C_null_mean
        Z = (C_obs - C_null_mean) / (C_null_std + 1e-9)

        C_obs_list.append(C_obs)
        C_null_list.append(C_null_mean)
        C_excess_list.append(C_excess)
        Z_list.append(Z)

    # Convert to arrays
    C_obs_arr = np.array(C_obs_list)
    C_null_arr = np.array(C_null_list)
    C_excess_arr = np.array(C_excess_list)
    Z_arr = np.array(Z_list)

    n_with_pairs = len(C_obs_arr)

    if n_with_pairs == 0:
        return DebiasedClusteringStats(
            model_name="",
            n_lenses=len(theta_list),
            n_with_pairs=0,
            C_obs_mean=np.nan,
            C_obs_se=np.nan,
            C_excess_mean=np.nan,
            C_excess_se=np.nan,
            Z_mean=np.nan,
            Z_se=np.nan,
            C_obs_values=np.array([]),
            C_null_values=np.array([]),
            C_excess_values=np.array([]),
            Z_values=np.array([]),
            global_pvalue=1.0
        )

    # Aggregate statistics
    C_obs_mean = np.mean(C_obs_arr)
    C_obs_se = np.std(C_obs_arr) / np.sqrt(n_with_pairs)

    C_excess_mean = np.mean(C_excess_arr)
    C_excess_se = np.std(C_excess_arr) / np.sqrt(n_with_pairs)

    Z_mean = np.mean(Z_arr)
    Z_se = np.std(Z_arr) / np.sqrt(n_with_pairs)

    # Global p-value: test if mean Z is significantly > 0
    # Under null, Z ~ N(0, 1), so mean(Z) ~ N(0, 1/sqrt(n))
    global_z = Z_mean * np.sqrt(n_with_pairs)
    global_pvalue = 1 - scipy_stats.norm.cdf(global_z)  # One-sided

    return DebiasedClusteringStats(
        model_name="",
        n_lenses=len(theta_list),
        n_with_pairs=n_with_pairs,
        C_obs_mean=C_obs_mean,
        C_obs_se=C_obs_se,
        C_excess_mean=C_excess_mean,
        C_excess_se=C_excess_se,
        Z_mean=Z_mean,
        Z_se=Z_se,
        C_obs_values=C_obs_arr,
        C_null_values=C_null_arr,
        C_excess_values=C_excess_arr,
        Z_values=Z_arr,
        global_pvalue=global_pvalue
    )


def format_debiased_stats_table(stats_list: List[DebiasedClusteringStats]) -> str:
    """Format debiased statistics as a markdown table."""
    lines = [
        "| Model | N_pairs | C_obs | C_excess | Z_mean | p-value |",
        "|-------|---------|-------|----------|--------|---------|",
    ]

    for s in stats_list:
        lines.append(
            f"| {s.model_name} | {s.n_with_pairs} | "
            f"{s.C_obs_mean:.4f}±{s.C_obs_se:.4f} | "
            f"{s.C_excess_mean:.4f}±{s.C_excess_se:.4f} | "
            f"{s.Z_mean:.3f}±{s.Z_se:.3f} | {s.global_pvalue:.4f} |"
        )

    return "\n".join(lines)
