"""
Plotting functions for Mverse Shadow Simulation

Generates diagnostic plots for K1/K2 analysis.
"""

from typing import Dict, Optional
import numpy as np
from pathlib import Path

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def ensure_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for plotting. Install with: pip install matplotlib")


def plot_r_histogram(R_X: np.ndarray, R_V: np.ndarray,
                      radius_label: str,
                      output_path: Path) -> None:
    """
    Plot histogram of R_X and R_V with vertical line at R=1.

    Parameters
    ----------
    R_X : array
        R ratio from lensing vs X-ray
    R_V : array
        R ratio from lensing vs velocity
    radius_label : str
        Label for radius (e.g., "r500")
    output_path : Path
        Output file path
    """
    ensure_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Clip for visualization
    R_X_clip = np.clip(R_X, 0.2, 3.0)
    R_V_clip = np.clip(R_V, 0.2, 3.0)

    bins = np.linspace(0.2, 3.0, 50)

    # R_X histogram
    ax = axes[0]
    ax.hist(R_X_clip, bins=bins, density=True, alpha=0.7, color='steelblue',
            edgecolor='navy', label=f'R_X at {radius_label}')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='R = 1 (universal)')
    ax.axvline(np.median(R_X_clip), color='orange', linestyle=':', linewidth=2,
               label=f'Median = {np.median(R_X_clip):.2f}')
    ax.set_xlabel('R_X = M_exc^lens / M_exc^dynX', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Lensing vs X-ray Hydrostatic ({radius_label})', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(0.2, 3.0)

    # R_V histogram
    ax = axes[1]
    ax.hist(R_V_clip, bins=bins, density=True, alpha=0.7, color='forestgreen',
            edgecolor='darkgreen', label=f'R_V at {radius_label}')
    ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='R = 1 (universal)')
    ax.axvline(np.median(R_V_clip), color='orange', linestyle=':', linewidth=2,
               label=f'Median = {np.median(R_V_clip):.2f}')
    ax.set_xlabel('R_V = M_exc^lens / M_exc^dynV', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'Lensing vs Velocity Dispersion ({radius_label})', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(0.2, 3.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_r_vs_mass(R_X: np.ndarray, R_V: np.ndarray,
                   log_M: np.ndarray,
                   radius_label: str,
                   output_path: Path) -> None:
    """
    Plot R vs log(M200) with trend line.

    Parameters
    ----------
    R_X, R_V : arrays
        R ratios
    log_M : array
        log10(M200)
    radius_label : str
        Label for radius
    output_path : Path
        Output file path
    """
    ensure_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Clip for visualization
    R_X_clip = np.clip(R_X, 0.2, 3.0)
    R_V_clip = np.clip(R_V, 0.2, 3.0)

    # R_X vs mass
    ax = axes[0]
    ax.scatter(log_M, R_X_clip, alpha=0.3, s=5, c='steelblue')

    # Fit and plot trend
    from scipy import stats
    mask = np.isfinite(R_X_clip) & np.isfinite(log_M)
    slope, intercept, _, p_value, _ = stats.linregress(log_M[mask], R_X_clip[mask])
    x_fit = np.linspace(log_M.min(), log_M.max(), 100)
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', linewidth=2,
            label=f'slope = {slope:.4f}, p = {p_value:.3f}')

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('log10(M200 / M_sun)', fontsize=12)
    ax.set_ylabel('R_X', fontsize=12)
    ax.set_title(f'R_X vs Mass ({radius_label})', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_ylim(0.2, 3.0)

    # R_V vs mass
    ax = axes[1]
    ax.scatter(log_M, R_V_clip, alpha=0.3, s=5, c='forestgreen')

    mask = np.isfinite(R_V_clip) & np.isfinite(log_M)
    slope, intercept, _, p_value, _ = stats.linregress(log_M[mask], R_V_clip[mask])
    y_fit = slope * x_fit + intercept
    ax.plot(x_fit, y_fit, 'r-', linewidth=2,
            label=f'slope = {slope:.4f}, p = {p_value:.3f}')

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('log10(M200 / M_sun)', fontsize=12)
    ax.set_ylabel('R_V', fontsize=12)
    ax.set_title(f'R_V vs Mass ({radius_label})', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_ylim(0.2, 3.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_r_vs_redshift(R_X: np.ndarray, R_V: np.ndarray,
                        z: np.ndarray,
                        radius_label: str,
                        output_path: Path) -> None:
    """
    Plot R vs redshift with trend line.
    """
    ensure_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    R_X_clip = np.clip(R_X, 0.2, 3.0)
    R_V_clip = np.clip(R_V, 0.2, 3.0)

    # R_X vs z
    ax = axes[0]
    ax.scatter(z, R_X_clip, alpha=0.3, s=5, c='steelblue')

    from scipy import stats
    mask = np.isfinite(R_X_clip) & np.isfinite(z)
    slope, intercept, _, p_value, _ = stats.linregress(z[mask], R_X_clip[mask])
    z_fit = np.linspace(z.min(), z.max(), 100)
    y_fit = slope * z_fit + intercept
    ax.plot(z_fit, y_fit, 'r-', linewidth=2,
            label=f'slope = {slope:.4f}, p = {p_value:.3f}')

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('R_X', fontsize=12)
    ax.set_title(f'R_X vs Redshift ({radius_label})', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_ylim(0.2, 3.0)

    # R_V vs z
    ax = axes[1]
    ax.scatter(z, R_V_clip, alpha=0.3, s=5, c='forestgreen')

    mask = np.isfinite(R_V_clip) & np.isfinite(z)
    slope, intercept, _, p_value, _ = stats.linregress(z[mask], R_V_clip[mask])
    y_fit = slope * z_fit + intercept
    ax.plot(z_fit, y_fit, 'r-', linewidth=2,
            label=f'slope = {slope:.4f}, p = {p_value:.3f}')

    ax.axhline(1.0, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Redshift z', fontsize=12)
    ax.set_ylabel('R_V', fontsize=12)
    ax.set_title(f'R_V vs Redshift ({radius_label})', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_ylim(0.2, 3.0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_quantile_bands(R_X: np.ndarray, R_V: np.ndarray,
                         log_M: np.ndarray,
                         radius_label: str,
                         output_path: Path,
                         n_bins: int = 8) -> None:
    """
    Plot quantile bands of R as a function of mass.
    """
    ensure_matplotlib()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bin by mass
    mass_bins = np.linspace(log_M.min(), log_M.max(), n_bins + 1)
    bin_centers = 0.5 * (mass_bins[:-1] + mass_bins[1:])

    for i_ax, (R, label, color) in enumerate([
        (R_X, 'R_X', 'steelblue'),
        (R_V, 'R_V', 'forestgreen')
    ]):
        ax = axes[i_ax]
        R_clip = np.clip(R, 0.2, 3.0)

        medians = []
        p16 = []
        p84 = []
        p05 = []
        p95 = []

        for j in range(n_bins):
            mask = (log_M >= mass_bins[j]) & (log_M < mass_bins[j+1])
            R_bin = R_clip[mask]
            if len(R_bin) > 5:
                medians.append(np.median(R_bin))
                p16.append(np.percentile(R_bin, 16))
                p84.append(np.percentile(R_bin, 84))
                p05.append(np.percentile(R_bin, 5))
                p95.append(np.percentile(R_bin, 95))
            else:
                medians.append(np.nan)
                p16.append(np.nan)
                p84.append(np.nan)
                p05.append(np.nan)
                p95.append(np.nan)

        medians = np.array(medians)
        p16 = np.array(p16)
        p84 = np.array(p84)
        p05 = np.array(p05)
        p95 = np.array(p95)

        ax.fill_between(bin_centers, p05, p95, alpha=0.2, color=color, label='5-95%')
        ax.fill_between(bin_centers, p16, p84, alpha=0.4, color=color, label='16-84%')
        ax.plot(bin_centers, medians, 'o-', color=color, linewidth=2, markersize=8, label='Median')

        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='R = 1')
        ax.set_xlabel('log10(M200 / M_sun)', fontsize=12)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f'{label} Quantile Bands ({radius_label})', fontsize=14)
        ax.legend(loc='upper right')
        ax.set_ylim(0.5, 2.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_bias_effects(inference_results: Dict,
                       radius: str,
                       output_path: Path) -> None:
    """
    Plot how R depends on the underlying biases.
    """
    ensure_matplotlib()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    if radius == "r500":
        R_X = inference_results['R_X_r500']
        R_V = inference_results['R_V_r500']
    else:
        R_X = inference_results['R_X_half_r500']
        R_V = inference_results['R_V_half_r500']

    b_hse = inference_results['b_hse_true']
    b_aniso = inference_results['b_aniso_true']
    proj_boost = inference_results['proj_boost']

    R_X_clip = np.clip(R_X, 0.2, 3.0)
    R_V_clip = np.clip(R_V, 0.2, 3.0)

    # R_X vs b_hse
    ax = axes[0, 0]
    ax.scatter(b_hse, R_X_clip, alpha=0.3, s=5, c='steelblue')
    ax.axhline(1.0, color='gray', linestyle='--')
    ax.set_xlabel('True hydrostatic bias b_hse', fontsize=12)
    ax.set_ylabel('R_X', fontsize=12)
    ax.set_title('R_X vs Hydrostatic Bias', fontsize=14)

    # R_X vs proj_boost
    ax = axes[0, 1]
    ax.scatter(proj_boost, R_X_clip, alpha=0.3, s=5, c='steelblue')
    ax.axhline(1.0, color='gray', linestyle='--')
    ax.set_xlabel('Projection boost factor', fontsize=12)
    ax.set_ylabel('R_X', fontsize=12)
    ax.set_title('R_X vs Projection Boost', fontsize=14)

    # R_V vs b_aniso
    ax = axes[1, 0]
    ax.scatter(b_aniso, R_V_clip, alpha=0.3, s=5, c='forestgreen')
    ax.axhline(1.0, color='gray', linestyle='--')
    ax.set_xlabel('True anisotropy bias b_aniso', fontsize=12)
    ax.set_ylabel('R_V', fontsize=12)
    ax.set_title('R_V vs Anisotropy Bias', fontsize=14)

    # R_V vs proj_boost
    ax = axes[1, 1]
    ax.scatter(proj_boost, R_V_clip, alpha=0.3, s=5, c='forestgreen')
    ax.axhline(1.0, color='gray', linestyle='--')
    ax.set_xlabel('Projection boost factor', fontsize=12)
    ax.set_ylabel('R_V', fontsize=12)
    ax.set_title('R_V vs Projection Boost', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_all_plots(inference_results: Dict,
                        output_dir: Path,
                        radius: str = "r500") -> None:
    """
    Generate all diagnostic plots.

    Parameters
    ----------
    inference_results : Dict
        Output from inference
    output_dir : Path
        Output directory
    radius : str
        "r500" or "half_r500"
    """
    ensure_matplotlib()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if radius == "r500":
        R_X = inference_results['R_X_r500']
        R_V = inference_results['R_V_r500']
    else:
        R_X = inference_results['R_X_half_r500']
        R_V = inference_results['R_V_half_r500']

    log_M = inference_results['log_M200_true']
    z = inference_results['z']

    suffix = radius.replace("_", "")

    # Generate plots
    plot_r_histogram(R_X, R_V, radius,
                     output_dir / f"R_histogram_{suffix}.png")

    plot_r_vs_mass(R_X, R_V, log_M, radius,
                   output_dir / f"R_vs_mass_{suffix}.png")

    plot_r_vs_redshift(R_X, R_V, z, radius,
                       output_dir / f"R_vs_redshift_{suffix}.png")

    plot_quantile_bands(R_X, R_V, log_M, radius,
                        output_dir / f"R_quantiles_{suffix}.png")

    plot_bias_effects(inference_results, radius,
                      output_dir / f"R_vs_biases_{suffix}.png")
