"""
Plotting functions for Msoup Substructure Simulation

Generates diagnostic plots for model comparison.
"""

from typing import Dict, List, Optional
import numpy as np
from pathlib import Path

from .simulate import SimulationResult
from .stats import StatsSummary, expected_uniform_clustering

# Check for matplotlib availability
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def ensure_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for plotting")


def plot_count_histograms(results: Dict[str, SimulationResult],
                           output_path: Path) -> None:
    """
    Plot histogram of perturber counts for each model.
    """
    ensure_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = {'poisson': 'steelblue', 'cox': 'darkorange', 'cluster': 'forestgreen'}
    titles = {
        'poisson': 'Poisson (CDM-like)',
        'cox': 'Cox (Over-dispersed)',
        'cluster': 'Cluster (Pockets)'
    }

    for i, (model, result) in enumerate(results.items()):
        ax = axes[i]
        counts = result.counts

        # Histogram
        max_count = int(np.percentile(counts, 99))
        bins = np.arange(0, max_count + 2) - 0.5

        ax.hist(counts, bins=bins, density=True, alpha=0.7,
                color=colors.get(model, 'gray'), edgecolor='black')

        # Poisson fit for comparison
        from scipy.stats import poisson
        mean = result.mean_count
        x = np.arange(0, max_count + 1)
        ax.plot(x, poisson.pmf(x, mean), 'r--', linewidth=2,
                label=f'Poisson(λ={mean:.2f})')

        ax.set_xlabel('Perturber count N', fontsize=12)
        ax.set_ylabel('Probability', fontsize=12)
        ax.set_title(f'{titles.get(model, model)}\nFano = {result.fano_factor:.3f}', fontsize=14)
        ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_fano_vs_n(sweep_results: Dict[str, List[SimulationResult]],
                   stats_list: Dict[str, List[StatsSummary]],
                   output_path: Path) -> None:
    """
    Plot Fano factor vs N_lenses for each model.
    """
    ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'poisson': 'steelblue', 'cox': 'darkorange', 'cluster': 'forestgreen'}
    markers = {'poisson': 'o', 'cox': 's', 'cluster': '^'}
    labels = {'poisson': 'Poisson', 'cox': 'Cox', 'cluster': 'Cluster'}

    for model in ['poisson', 'cox', 'cluster']:
        if model not in stats_list:
            continue

        n_vals = [s.n_lenses for s in stats_list[model]]
        fano_vals = [s.fano_factor for s in stats_list[model]]
        fano_se = [s.fano_se for s in stats_list[model]]

        ax.errorbar(n_vals, fano_vals, yerr=fano_se,
                    marker=markers[model], color=colors[model],
                    label=labels[model], linewidth=2, markersize=8, capsize=3)

    ax.axhline(1.0, color='gray', linestyle='--', linewidth=1, label='Poisson (F=1)')
    ax.set_xlabel('Number of lenses', fontsize=12)
    ax.set_ylabel('Fano factor (Var/Mean)', fontsize=12)
    ax.set_title('Fano Factor vs Sample Size', fontsize=14)
    ax.legend()
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_clustering_vs_n(stats_list: Dict[str, List[StatsSummary]],
                          theta0: float,
                          output_path: Path) -> None:
    """
    Plot clustering metric C vs N_lenses for each model.
    """
    ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'poisson': 'steelblue', 'cox': 'darkorange', 'cluster': 'forestgreen'}
    markers = {'poisson': 'o', 'cox': 's', 'cluster': '^'}
    labels = {'poisson': 'Poisson', 'cox': 'Cox', 'cluster': 'Cluster'}

    for model in ['poisson', 'cox', 'cluster']:
        if model not in stats_list:
            continue

        n_vals = [s.n_lenses for s in stats_list[model]]
        C_vals = [s.clustering_C for s in stats_list[model]]
        C_se = [s.clustering_C_se for s in stats_list[model]]

        ax.errorbar(n_vals, C_vals, yerr=C_se,
                    marker=markers[model], color=colors[model],
                    label=labels[model], linewidth=2, markersize=8, capsize=3)

    # Expected for uniform
    C_uniform = expected_uniform_clustering(theta0)
    ax.axhline(C_uniform, color='gray', linestyle='--', linewidth=1,
               label=f'Uniform (C={C_uniform:.3f})')

    ax.set_xlabel('Number of lenses', fontsize=12)
    ax.set_ylabel(f'Clustering metric C (θ₀={theta0:.2f} rad)', fontsize=12)
    ax.set_title('Spatial Clustering vs Sample Size', fontsize=14)
    ax.legend()
    ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_residual_scatter(results: Dict[str, SimulationResult],
                           output_path: Path) -> None:
    """
    Plot N_i vs host proxy H, showing residual scatter.
    """
    ensure_matplotlib()

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = {'poisson': 'steelblue', 'cox': 'darkorange', 'cluster': 'forestgreen'}
    titles = {'poisson': 'Poisson', 'cox': 'Cox', 'cluster': 'Cluster'}

    for i, (model, result) in enumerate(results.items()):
        ax = axes[i]

        # Subsample for visibility
        n_plot = min(2000, len(result.counts))
        idx = np.random.choice(len(result.counts), n_plot, replace=False)

        ax.scatter(result.H[idx], result.counts[idx], alpha=0.3, s=10,
                   c=colors.get(model, 'gray'))

        # Trend line
        z = np.polyfit(result.H[idx], result.counts[idx], 1)
        p = np.poly1d(z)
        H_range = np.linspace(result.H.min(), result.H.max(), 100)
        ax.plot(H_range, p(H_range), 'r-', linewidth=2)

        ax.set_xlabel('Host mass proxy H', fontsize=12)
        ax.set_ylabel('Perturber count N', fontsize=12)
        ax.set_title(f'{titles.get(model, model)} (Fano={result.fano_factor:.2f})', fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_clustering_distribution(stats_dict: Dict[str, StatsSummary],
                                  theta0: float,
                                  output_path: Path) -> None:
    """
    Plot distribution of per-lens clustering values.
    """
    ensure_matplotlib()

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'poisson': 'steelblue', 'cox': 'darkorange', 'cluster': 'forestgreen'}
    labels = {'poisson': 'Poisson', 'cox': 'Cox', 'cluster': 'Cluster'}

    bins = np.linspace(0, 1, 30)

    for model in ['poisson', 'cox', 'cluster']:
        if model not in stats_dict:
            continue

        C_values = stats_dict[model].clustering_values
        if len(C_values) > 0:
            ax.hist(C_values, bins=bins, density=True, alpha=0.5,
                    color=colors[model], label=labels[model], edgecolor='black')

    # Expected for uniform
    C_uniform = expected_uniform_clustering(theta0)
    ax.axvline(C_uniform, color='gray', linestyle='--', linewidth=2,
               label=f'Uniform expected (C={C_uniform:.3f})')

    ax.set_xlabel(f'Per-lens clustering C (θ₀={theta0:.2f} rad)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Distribution of Spatial Clustering Metric', fontsize=14)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_angular_examples(results: Dict[str, SimulationResult],
                           n_examples: int = 3,
                           output_path: Path = None) -> None:
    """
    Plot example lenses showing angular positions of perturbers.
    """
    ensure_matplotlib()

    fig, axes = plt.subplots(3, n_examples, figsize=(4*n_examples, 12),
                              subplot_kw=dict(projection='polar'))

    colors = {'poisson': 'steelblue', 'cox': 'darkorange', 'cluster': 'forestgreen'}
    titles = {'poisson': 'Poisson', 'cox': 'Cox', 'cluster': 'Cluster'}

    for i, model in enumerate(['poisson', 'cox', 'cluster']):
        if model not in results:
            continue

        result = results[model]

        # Find lenses with moderate counts for visualization
        counts = result.counts
        valid_idx = np.where((counts >= 3) & (counts <= 10))[0]

        if len(valid_idx) < n_examples:
            valid_idx = np.where(counts >= 2)[0]

        if len(valid_idx) >= n_examples:
            example_idx = np.random.choice(valid_idx, n_examples, replace=False)
        else:
            example_idx = valid_idx[:n_examples]

        for j, idx in enumerate(example_idx):
            ax = axes[i, j]
            theta = result.theta_list[idx]

            # Plot perturbers on the ring
            r = np.ones(len(theta))
            ax.scatter(theta, r, s=100, c=colors[model], alpha=0.7, edgecolor='black')

            # Draw the ring
            ring_theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(ring_theta, np.ones(100), 'k-', linewidth=1, alpha=0.3)

            ax.set_ylim(0, 1.5)
            ax.set_rticks([])

            if j == 0:
                ax.set_ylabel(titles[model], fontsize=14, labelpad=20)
            if i == 0:
                ax.set_title(f'N={len(theta)}', fontsize=12)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_all_plots(results: Dict[str, SimulationResult],
                        stats_dict: Dict[str, StatsSummary],
                        output_dir: Path,
                        theta0: float = 0.3) -> None:
    """
    Generate all diagnostic plots.
    """
    ensure_matplotlib()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Count histograms
    plot_count_histograms(results, output_dir / "count_histograms.png")

    # Residual scatter
    plot_residual_scatter(results, output_dir / "residual_scatter.png")

    # Clustering distribution
    plot_clustering_distribution(stats_dict, theta0, output_dir / "clustering_distribution.png")

    # Angular examples
    plot_angular_examples(results, n_examples=3, output_path=output_dir / "angular_examples.png")


def generate_sweep_plots(sweep_stats: Dict[str, List[StatsSummary]],
                          output_dir: Path,
                          theta0: float = 0.3) -> None:
    """
    Generate plots from sweep results.
    """
    ensure_matplotlib()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Fano vs N
    plot_fano_vs_n({}, sweep_stats, output_dir / "fano_vs_n.png")

    # Clustering vs N
    plot_clustering_vs_n(sweep_stats, theta0, output_dir / "clustering_vs_n.png")


# ============================================================================
# WINDOW-SPECIFIC PLOTS
# ============================================================================

def plot_sensitivity_windows(windows: List,
                             n_examples: int = 4,
                             output_path: Path = None) -> None:
    """
    Plot example sensitivity windows S(θ).
    """
    ensure_matplotlib()

    fig, axes = plt.subplots(1, n_examples, figsize=(4*n_examples, 4))

    for i, window in enumerate(windows[:n_examples]):
        ax = axes[i] if n_examples > 1 else axes

        # Evaluate on grid
        theta_grid, S_grid = window.evaluate_grid(n_grid=200)

        ax.fill_between(theta_grid, 0, S_grid, alpha=0.3, color='steelblue')
        ax.plot(theta_grid, S_grid, 'steelblue', linewidth=2)

        # Mark segment centers
        for center in window.theta_centers:
            ax.axvline(center, color='red', linestyle='--', alpha=0.5)

        ax.axhline(1.0, color='gray', linestyle=':', linewidth=1)

        ax.set_xlabel('θ (radians)', fontsize=11)
        ax.set_ylabel('S(θ)', fontsize=11)
        ax.set_title(f'Lens {i+1}: K={len(window.theta_centers)} segments', fontsize=12)
        ax.set_xlim(0, 2*np.pi)
        ax.set_ylim(0, None)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_sensitivity_polar(windows: List,
                           n_examples: int = 4,
                           output_path: Path = None) -> None:
    """
    Plot sensitivity windows in polar coordinates (ring visualization).
    """
    ensure_matplotlib()

    fig, axes = plt.subplots(1, n_examples, figsize=(4*n_examples, 4),
                              subplot_kw=dict(projection='polar'))

    for i, window in enumerate(windows[:n_examples]):
        ax = axes[i] if n_examples > 1 else axes

        theta_grid, S_grid = window.evaluate_grid(n_grid=200)

        # Plot as radial variation from 1
        r = 0.5 + 0.5 * S_grid / np.max(S_grid)

        ax.fill(theta_grid, r, alpha=0.3, color='steelblue')
        ax.plot(theta_grid, r, 'steelblue', linewidth=2)

        ax.set_rticks([])
        ax.set_title(f'K={len(window.theta_centers)}', fontsize=12, pad=10)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_debiased_clustering_distribution(debias_dict: Dict,
                                           theta0: float,
                                           output_path: Path = None) -> None:
    """
    Plot distributions of C_obs, C_excess, and Z for debiased analysis.
    """
    ensure_matplotlib()
    from .stats import DebiasedClusteringStats

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    colors = {
        'poisson_window': 'steelblue',
        'cox_window': 'darkorange',
        'cluster_window': 'forestgreen'
    }
    labels = {
        'poisson_window': 'Poisson+Window',
        'cox_window': 'Cox+Window',
        'cluster_window': 'Cluster+Window'
    }

    # C_obs distribution
    ax = axes[0]
    for model in ['poisson_window', 'cox_window', 'cluster_window']:
        if model in debias_dict:
            d = debias_dict[model]
            if len(d.C_obs_values) > 0:
                ax.hist(d.C_obs_values, bins=30, density=True, alpha=0.5,
                        color=colors[model], label=labels[model], edgecolor='black')

    C_uniform = expected_uniform_clustering(theta0)
    ax.axvline(C_uniform, color='gray', linestyle='--', linewidth=2,
               label=f'Uniform ({C_uniform:.3f})')
    ax.set_xlabel('C_obs', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Observed Clustering', fontsize=14)
    ax.legend(fontsize=9)

    # C_excess distribution
    ax = axes[1]
    for model in ['poisson_window', 'cox_window', 'cluster_window']:
        if model in debias_dict:
            d = debias_dict[model]
            if len(d.C_excess_values) > 0:
                ax.hist(d.C_excess_values, bins=30, density=True, alpha=0.5,
                        color=colors[model], label=labels[model], edgecolor='black')

    ax.axvline(0, color='gray', linestyle='--', linewidth=2)
    ax.set_xlabel('C_excess = C_obs - C_null', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Excess Clustering (Debiased)', fontsize=14)
    ax.legend(fontsize=9)

    # Z distribution
    ax = axes[2]
    for model in ['poisson_window', 'cox_window', 'cluster_window']:
        if model in debias_dict:
            d = debias_dict[model]
            if len(d.Z_values) > 0:
                ax.hist(d.Z_values, bins=30, density=True, alpha=0.5,
                        color=colors[model], label=labels[model], edgecolor='black')

    # Standard normal for reference
    z_range = np.linspace(-4, 4, 100)
    from scipy.stats import norm
    ax.plot(z_range, norm.pdf(z_range), 'k-', linewidth=2, label='N(0,1)')
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    ax.set_xlabel('Z = (C_obs - C_null) / σ_null', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('Standardized Excess', fontsize=14)
    ax.legend(fontsize=9)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_window_sweep_heatmap(sweep_results: Dict,
                               output_path: Path = None) -> None:
    """
    Plot heatmap of C_obs across window parameters.
    """
    ensure_matplotlib()

    if not sweep_results or 'params' not in sweep_results:
        return

    params = sweep_results['params']
    C_obs = np.array(sweep_results['C_obs'])

    # Extract unique parameter values
    kappas = sorted(set(p[0] for p in params))
    sigmas = sorted(set(p[1] for p in params))
    gammas = sorted(set(p[2] for p in params))

    # Create heatmaps for different gamma values
    n_gamma = len(gammas)
    fig, axes = plt.subplots(1, n_gamma, figsize=(5*n_gamma, 4))

    if n_gamma == 1:
        axes = [axes]

    for g_idx, gamma in enumerate(gammas):
        ax = axes[g_idx]

        # Build matrix
        matrix = np.full((len(sigmas), len(kappas)), np.nan)

        for i, (kappa, sigma, g) in enumerate(params):
            if g == gamma:
                k_idx = kappas.index(kappa)
                s_idx = sigmas.index(sigma)
                matrix[s_idx, k_idx] = C_obs[i]

        im = ax.imshow(matrix, cmap='viridis', aspect='auto',
                       origin='lower', vmin=0.05, vmax=0.35)

        ax.set_xticks(range(len(kappas)))
        ax.set_xticklabels([f'{k:.1f}' for k in kappas])
        ax.set_yticks(range(len(sigmas)))
        ax.set_yticklabels([f'{s:.2f}' for s in sigmas])

        ax.set_xlabel('κ_arc (num segments)', fontsize=11)
        ax.set_ylabel('σ_arc (segment width)', fontsize=11)
        ax.set_title(f'γ = {gamma}', fontsize=12)

        plt.colorbar(im, ax=ax, label='C_obs')

    plt.suptitle('Observed Clustering vs Window Parameters', fontsize=14, y=1.02)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_window_angular_examples(results: Dict[str, SimulationResult],
                                  n_examples: int = 3,
                                  output_path: Path = None) -> None:
    """
    Plot example lenses showing perturbers and sensitivity windows.
    """
    ensure_matplotlib()

    models = ['poisson_window', 'cox_window', 'cluster_window']
    n_models = sum(1 for m in models if m in results)

    if n_models == 0:
        return

    fig, axes = plt.subplots(n_models, n_examples, figsize=(4*n_examples, 4*n_models),
                              subplot_kw=dict(projection='polar'))

    if n_models == 1:
        axes = axes.reshape(1, -1)
    if n_examples == 1:
        axes = axes.reshape(-1, 1)

    colors = {
        'poisson_window': 'steelblue',
        'cox_window': 'darkorange',
        'cluster_window': 'forestgreen'
    }
    titles = {
        'poisson_window': 'Poisson+Window',
        'cox_window': 'Cox+Window',
        'cluster_window': 'Cluster+Window'
    }

    row = 0
    for model in models:
        if model not in results:
            continue

        result = results[model]
        windows = result.windows

        # Find lenses with moderate counts
        counts = result.counts
        valid_idx = np.where((counts >= 3) & (counts <= 10))[0]

        if len(valid_idx) < n_examples:
            valid_idx = np.where(counts >= 2)[0]

        if len(valid_idx) >= n_examples:
            example_idx = np.random.choice(valid_idx, n_examples, replace=False)
        else:
            example_idx = valid_idx[:n_examples]

        for col, idx in enumerate(example_idx):
            ax = axes[row, col]
            theta = result.theta_list[idx]

            # Plot sensitivity window as background
            if windows is not None and idx < len(windows):
                window = windows[idx]
                theta_grid, S_grid = window.evaluate_grid(n_grid=200)
                r_sensitivity = 0.3 + 0.7 * S_grid / np.max(S_grid)
                ax.fill(theta_grid, r_sensitivity, alpha=0.2, color='gray')

            # Plot perturbers
            r = np.ones(len(theta))
            ax.scatter(theta, r, s=100, c=colors[model], alpha=0.8, edgecolor='black', zorder=3)

            ax.set_ylim(0, 1.3)
            ax.set_rticks([])

            if col == 0:
                ax.set_ylabel(titles[model], fontsize=12, labelpad=20)
            if row == 0:
                ax.set_title(f'N={len(theta)}', fontsize=11)

        row += 1

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def generate_window_plots(results: Dict[str, SimulationResult],
                          debias_dict: Dict,
                          sweep_results: Dict,
                          output_dir: Path,
                          theta0: float = 0.3) -> None:
    """
    Generate all window-specific plots.
    """
    ensure_matplotlib()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sensitivity window examples
    for model in ['poisson_window', 'cox_window', 'cluster_window']:
        if model in results and results[model].windows is not None:
            plot_sensitivity_windows(
                results[model].windows[:8],
                n_examples=4,
                output_path=output_dir / f"sensitivity_windows_{model}.png"
            )
            plot_sensitivity_polar(
                results[model].windows[:4],
                n_examples=4,
                output_path=output_dir / f"sensitivity_polar_{model}.png"
            )
            break  # Only need one set

    # Debiased clustering distributions
    if debias_dict:
        plot_debiased_clustering_distribution(
            debias_dict, theta0,
            output_path=output_dir / "debiased_clustering_distribution.png"
        )

    # Window sweep heatmap
    if sweep_results and 'params' in sweep_results:
        plot_window_sweep_heatmap(
            sweep_results,
            output_path=output_dir / "window_sweep_heatmap.png"
        )

    # Angular examples with windows
    plot_window_angular_examples(
        results, n_examples=3,
        output_path=output_dir / "window_angular_examples.png"
    )
