"""
Metrics and Kill-Test Calibration for Mverse Shadow Simulation

Computes K1 (scatter about R=1) and K2 (trend with environment) metrics
and recommends pre-registration thresholds based on simulation.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import stats


@dataclass
class K1Metrics:
    """K1 kill condition metrics: scatter about R=1."""
    # For R_X (lensing vs X-ray)
    R_X_mean: float
    R_X_median: float
    R_X_std: float
    R_X_mad: float  # median absolute deviation
    R_X_p16: float
    R_X_p84: float
    R_X_p05: float
    R_X_p95: float

    # For R_V (lensing vs velocity)
    R_V_mean: float
    R_V_median: float
    R_V_std: float
    R_V_mad: float
    R_V_p16: float
    R_V_p84: float
    R_V_p05: float
    R_V_p95: float

    # Violation fractions at various thresholds
    violation_fracs_X: Dict[float, float]  # eps -> P(|R_X - 1| > eps)
    violation_fracs_V: Dict[float, float]

    # Recommended K1 tolerance (95th percentile of |R-1|)
    K1_tolerance_X: float
    K1_tolerance_V: float


@dataclass
class K2Metrics:
    """K2 kill condition metrics: trends with environment."""
    # Trend with mass
    slope_mass_X: float
    slope_mass_X_err: float
    slope_mass_X_pval: float

    slope_mass_V: float
    slope_mass_V_err: float
    slope_mass_V_pval: float

    # Trend with redshift
    slope_z_X: float
    slope_z_X_err: float
    slope_z_X_pval: float

    slope_z_V: float
    slope_z_V_err: float
    slope_z_V_pval: float

    # Recommended K2 tolerance (3-sigma of null slopes)
    K2_tolerance_mass: float
    K2_tolerance_z: float

    # Fraction of simulations that would show "significant" trend
    false_positive_rate_mass: float
    false_positive_rate_z: float


@dataclass
class MetricsSummary:
    """Combined metrics summary."""
    k1: K1Metrics
    k2: K2Metrics
    n_clusters: int
    radius_label: str  # "r500" or "half_r500"

    # Overall recommendations
    recommended_K1_tolerance: float
    recommended_K2_tolerance_mass: float
    recommended_K2_tolerance_z: float


def compute_k1_metrics(R_X: np.ndarray, R_V: np.ndarray,
                        thresholds: List[float] = None) -> K1Metrics:
    """
    Compute K1 (scatter) metrics.

    Parameters
    ----------
    R_X : array
        R ratio from lensing vs X-ray
    R_V : array
        R ratio from lensing vs velocity
    thresholds : list of float
        Thresholds for violation fraction calculation

    Returns
    -------
    K1Metrics
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.20, 0.30, 0.50]

    # Clip extreme outliers for robust statistics
    R_X_clip = np.clip(R_X, 0.1, 10.0)
    R_V_clip = np.clip(R_V, 0.1, 10.0)

    def robust_stats(arr):
        return {
            'mean': np.mean(arr),
            'median': np.median(arr),
            'std': np.std(arr),
            'mad': np.median(np.abs(arr - np.median(arr))),
            'p16': np.percentile(arr, 16),
            'p84': np.percentile(arr, 84),
            'p05': np.percentile(arr, 5),
            'p95': np.percentile(arr, 95),
        }

    stats_X = robust_stats(R_X_clip)
    stats_V = robust_stats(R_V_clip)

    # Violation fractions
    def violation_frac(arr, eps):
        return np.mean(np.abs(arr - 1) > eps)

    vf_X = {eps: violation_frac(R_X_clip, eps) for eps in thresholds}
    vf_V = {eps: violation_frac(R_V_clip, eps) for eps in thresholds}

    # K1 tolerance = 95th percentile of |R-1|
    K1_tol_X = np.percentile(np.abs(R_X_clip - 1), 95)
    K1_tol_V = np.percentile(np.abs(R_V_clip - 1), 95)

    return K1Metrics(
        R_X_mean=stats_X['mean'],
        R_X_median=stats_X['median'],
        R_X_std=stats_X['std'],
        R_X_mad=stats_X['mad'],
        R_X_p16=stats_X['p16'],
        R_X_p84=stats_X['p84'],
        R_X_p05=stats_X['p05'],
        R_X_p95=stats_X['p95'],
        R_V_mean=stats_V['mean'],
        R_V_median=stats_V['median'],
        R_V_std=stats_V['std'],
        R_V_mad=stats_V['mad'],
        R_V_p16=stats_V['p16'],
        R_V_p84=stats_V['p84'],
        R_V_p05=stats_V['p05'],
        R_V_p95=stats_V['p95'],
        violation_fracs_X=vf_X,
        violation_fracs_V=vf_V,
        K1_tolerance_X=K1_tol_X,
        K1_tolerance_V=K1_tol_V,
    )


def compute_k2_metrics(R_X: np.ndarray, R_V: np.ndarray,
                        log_M: np.ndarray, z: np.ndarray,
                        n_bootstrap: int = 100,
                        rng: Optional[np.random.Generator] = None) -> K2Metrics:
    """
    Compute K2 (trend) metrics.

    Parameters
    ----------
    R_X, R_V : arrays
        R ratios
    log_M : array
        log10(M200) in M_sun
    z : array
        Redshifts
    n_bootstrap : int
        Number of bootstrap samples for error estimation
    rng : Generator
        Random number generator

    Returns
    -------
    K2Metrics
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Clip extreme outliers
    R_X_clip = np.clip(R_X, 0.1, 10.0)
    R_V_clip = np.clip(R_V, 0.1, 10.0)

    def fit_trend(R, x):
        """Fit linear trend (R-1) vs x, return slope, error, p-value."""
        y = R - 1
        # Remove NaN/inf
        mask = np.isfinite(y) & np.isfinite(x)
        y = y[mask]
        x = x[mask]

        if len(y) < 10:
            return 0.0, np.inf, 1.0

        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return slope, std_err, p_value

    # Mass trends
    slope_mass_X, err_mass_X, pval_mass_X = fit_trend(R_X_clip, log_M)
    slope_mass_V, err_mass_V, pval_mass_V = fit_trend(R_V_clip, log_M)

    # Redshift trends
    slope_z_X, err_z_X, pval_z_X = fit_trend(R_X_clip, z)
    slope_z_V, err_z_V, pval_z_V = fit_trend(R_V_clip, z)

    # Bootstrap to get null distribution of slopes
    slopes_mass_boot = []
    slopes_z_boot = []
    N = len(R_X)

    for _ in range(n_bootstrap):
        idx = rng.choice(N, N, replace=True)
        # Shuffle to break correlations (null hypothesis)
        R_shuffled = R_X_clip[rng.permutation(N)]
        s_m, _, _ = fit_trend(R_shuffled[idx], log_M[idx])
        s_z, _, _ = fit_trend(R_shuffled[idx], z[idx])
        slopes_mass_boot.append(s_m)
        slopes_z_boot.append(s_z)

    slopes_mass_boot = np.array(slopes_mass_boot)
    slopes_z_boot = np.array(slopes_z_boot)

    # K2 tolerance = 3-sigma of null slopes
    K2_tol_mass = 3 * np.std(slopes_mass_boot)
    K2_tol_z = 3 * np.std(slopes_z_boot)

    # False positive rate (how often would random data show "significant" trend)
    sig_threshold_mass = 0.05  # p-value threshold
    sig_threshold_z = 0.05

    fp_mass = np.mean(np.abs(slopes_mass_boot) > 2 * np.std(slopes_mass_boot))
    fp_z = np.mean(np.abs(slopes_z_boot) > 2 * np.std(slopes_z_boot))

    return K2Metrics(
        slope_mass_X=slope_mass_X,
        slope_mass_X_err=err_mass_X,
        slope_mass_X_pval=pval_mass_X,
        slope_mass_V=slope_mass_V,
        slope_mass_V_err=err_mass_V,
        slope_mass_V_pval=pval_mass_V,
        slope_z_X=slope_z_X,
        slope_z_X_err=err_z_X,
        slope_z_X_pval=pval_z_X,
        slope_z_V=slope_z_V,
        slope_z_V_err=err_z_V,
        slope_z_V_pval=pval_z_V,
        K2_tolerance_mass=K2_tol_mass,
        K2_tolerance_z=K2_tol_z,
        false_positive_rate_mass=fp_mass,
        false_positive_rate_z=fp_z,
    )


def compute_all_metrics(inference_results: Dict,
                         radius: str = "r500",
                         rng: Optional[np.random.Generator] = None) -> MetricsSummary:
    """
    Compute all K1 and K2 metrics for a given radius.

    Parameters
    ----------
    inference_results : Dict
        Output from run_inference_vectorized or run_inference_full
    radius : str
        "r500" or "half_r500"
    rng : Generator
        Random number generator

    Returns
    -------
    MetricsSummary
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Get R arrays for the specified radius
    if radius == "r500":
        R_X = inference_results['R_X_r500']
        R_V = inference_results['R_V_r500']
    else:
        R_X = inference_results['R_X_half_r500']
        R_V = inference_results['R_V_half_r500']

    log_M = inference_results['log_M200_true']
    z = inference_results['z']

    k1 = compute_k1_metrics(R_X, R_V)
    k2 = compute_k2_metrics(R_X, R_V, log_M, z, rng=rng)

    # Overall recommendations (conservative: max of X and V)
    rec_K1 = max(k1.K1_tolerance_X, k1.K1_tolerance_V)
    rec_K2_mass = k2.K2_tolerance_mass
    rec_K2_z = k2.K2_tolerance_z

    return MetricsSummary(
        k1=k1,
        k2=k2,
        n_clusters=len(R_X),
        radius_label=radius,
        recommended_K1_tolerance=rec_K1,
        recommended_K2_tolerance_mass=rec_K2_mass,
        recommended_K2_tolerance_z=rec_K2_z,
    )


def format_metrics_table(summary: MetricsSummary) -> str:
    """Format metrics as a markdown table."""
    k1 = summary.k1
    k2 = summary.k2

    lines = [
        f"## Metrics at {summary.radius_label} (N = {summary.n_clusters})",
        "",
        "### K1: Scatter about R = 1",
        "",
        "| Statistic | R_X (Lens/X-ray) | R_V (Lens/Velocity) |",
        "|-----------|------------------|---------------------|",
        f"| Mean | {k1.R_X_mean:.3f} | {k1.R_V_mean:.3f} |",
        f"| Median | {k1.R_X_median:.3f} | {k1.R_V_median:.3f} |",
        f"| Std | {k1.R_X_std:.3f} | {k1.R_V_std:.3f} |",
        f"| MAD | {k1.R_X_mad:.3f} | {k1.R_V_mad:.3f} |",
        f"| 16-84% | [{k1.R_X_p16:.3f}, {k1.R_X_p84:.3f}] | [{k1.R_V_p16:.3f}, {k1.R_V_p84:.3f}] |",
        f"| 5-95% | [{k1.R_X_p05:.3f}, {k1.R_X_p95:.3f}] | [{k1.R_V_p05:.3f}, {k1.R_V_p95:.3f}] |",
        "",
        "**Violation fractions P(|R-1| > eps):**",
        "",
        "| eps | R_X | R_V |",
        "|-----|-----|-----|",
    ]

    for eps in sorted(k1.violation_fracs_X.keys()):
        lines.append(f"| {eps:.2f} | {k1.violation_fracs_X[eps]:.1%} | {k1.violation_fracs_V[eps]:.1%} |")

    lines.extend([
        "",
        f"**Recommended K1 tolerance (95th pct of |R-1|):** X={k1.K1_tolerance_X:.3f}, V={k1.K1_tolerance_V:.3f}",
        "",
        "### K2: Trends with Environment",
        "",
        "| Trend | Slope | Std Err | p-value |",
        "|-------|-------|---------|---------|",
        f"| R_X vs log(M) | {k2.slope_mass_X:.4f} | {k2.slope_mass_X_err:.4f} | {k2.slope_mass_X_pval:.3f} |",
        f"| R_V vs log(M) | {k2.slope_mass_V:.4f} | {k2.slope_mass_V_err:.4f} | {k2.slope_mass_V_pval:.3f} |",
        f"| R_X vs z | {k2.slope_z_X:.4f} | {k2.slope_z_X_err:.4f} | {k2.slope_z_X_pval:.3f} |",
        f"| R_V vs z | {k2.slope_z_V:.4f} | {k2.slope_z_V_err:.4f} | {k2.slope_z_V_pval:.3f} |",
        "",
        f"**Recommended K2 tolerances (3-sigma of null):** mass={k2.K2_tolerance_mass:.4f}, z={k2.K2_tolerance_z:.4f}",
        "",
        f"**False positive rates under null:** mass={k2.false_positive_rate_mass:.1%}, z={k2.false_positive_rate_z:.1%}",
        "",
    ])

    return "\n".join(lines)


def check_kill_conditions(summary: MetricsSummary,
                           K1_threshold: Optional[float] = None,
                           K2_mass_threshold: Optional[float] = None,
                           K2_z_threshold: Optional[float] = None) -> Dict:
    """
    Check if kill conditions would be triggered.

    Parameters
    ----------
    summary : MetricsSummary
        Computed metrics
    K1_threshold : float, optional
        Override K1 tolerance (default: use recommended)
    K2_mass_threshold : float, optional
        Override K2 mass slope tolerance
    K2_z_threshold : float, optional
        Override K2 z slope tolerance

    Returns
    -------
    Dict with kill condition status
    """
    k1 = summary.k1
    k2 = summary.k2

    K1_tol = K1_threshold or summary.recommended_K1_tolerance
    K2_mass_tol = K2_mass_threshold or summary.recommended_K2_tolerance_mass
    K2_z_tol = K2_z_threshold or summary.recommended_K2_tolerance_z

    # K1: fraction of clusters with |R-1| > tolerance
    k1_viol_X = k1.violation_fracs_X.get(K1_tol, np.mean(np.abs(k1.R_X_median - 1) > K1_tol))
    k1_viol_V = k1.violation_fracs_V.get(K1_tol, np.mean(np.abs(k1.R_V_median - 1) > K1_tol))

    # K2: are slopes significant (> tolerance)?
    k2_mass_X_trig = np.abs(k2.slope_mass_X) > K2_mass_tol
    k2_mass_V_trig = np.abs(k2.slope_mass_V) > K2_mass_tol
    k2_z_X_trig = np.abs(k2.slope_z_X) > K2_z_tol
    k2_z_V_trig = np.abs(k2.slope_z_V) > K2_z_tol

    return {
        'K1_tolerance_used': K1_tol,
        'K2_mass_tolerance_used': K2_mass_tol,
        'K2_z_tolerance_used': K2_z_tol,
        'K1_triggered': (k1.R_X_median < 1 - K1_tol) or (k1.R_X_median > 1 + K1_tol) or
                        (k1.R_V_median < 1 - K1_tol) or (k1.R_V_median > 1 + K1_tol),
        'K2_mass_triggered': k2_mass_X_trig or k2_mass_V_trig,
        'K2_z_triggered': k2_z_X_trig or k2_z_V_trig,
        'overall_kill': False,  # Under universal alpha, should not trigger
    }
