"""
CLI Entrypoint for Msoup Substructure Simulation

Run simulations and generate reports comparing Poisson, Cox, and Cluster models.
"""

import argparse
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import numpy as np

from .config import SimConfig, calibrate_models_to_match_mean
from .simulate import run_simulation, run_comparison, run_sweep, SimulationResult
from .stats import (analyze_result, StatsSummary, format_stats_table,
                    discriminate_models, compute_detectability, expected_uniform_clustering,
                    compute_debiased_clustering, DebiasedClusteringStats, format_debiased_stats_table)

# Conditional import for plots
try:
    from .plots import (generate_all_plots, generate_sweep_plots,
                        generate_window_plots, HAS_MATPLOTLIB)
except ImportError:
    HAS_MATPLOTLIB = False


def generate_report(results: Dict[str, SimulationResult],
                    stats_dict: Dict[str, StatsSummary],
                    sweep_stats: Dict[str, List[StatsSummary]],
                    output_path: Path,
                    runtime: float) -> str:
    """Generate markdown report."""
    lines = [
        "# Msoup Substructure Simulation: Domain/Pocket Distinguishability",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This simulation tests whether 'domain/pocket' dark structure (Msoup case-2)",
        "produces distinguishable substructure statistics compared to a Poisson baseline.",
        "",
        "### Key Findings",
        "",
    ]

    # Extract key findings from stats
    if 'poisson' in stats_dict and 'cluster' in stats_dict:
        ps = stats_dict['poisson']
        cs = stats_dict['cluster']
        cx = stats_dict.get('cox', None)

        lines.extend([
            f"**At N = {ps.n_lenses} lenses:**",
            "",
            f"| Model | Fano Factor | Clustering C | Meaning |",
            f"|-------|-------------|--------------|---------|",
            f"| Poisson | {ps.fano_factor:.3f} | {ps.clustering_C:.4f} | CDM-like baseline |",
        ])

        if cx:
            lines.append(f"| Cox | {cx.fano_factor:.3f} | {cx.clustering_C:.4f} | Over-dispersion only |")

        lines.append(f"| Cluster | {cs.fano_factor:.3f} | {cs.clustering_C:.4f} | Pockets (both effects) |")

        # Compute discriminability
        disc_cox = discriminate_models(ps, cx) if cx else None
        disc_cluster = discriminate_models(ps, cs)

        lines.extend([
            "",
            "### Discriminability",
            "",
        ])

        if disc_cox:
            lines.extend([
                f"**Cox vs Poisson:**",
                f"- Fano difference: {disc_cox['fano_diff']:.3f} (z = {disc_cox['fano_z']:.2f}, p = {disc_cox['fano_pvalue']:.4f})",
                f"- Clustering difference: {disc_cox['clustering_diff']:.4f} (z = {disc_cox['clustering_z']:.2f}, p = {disc_cox['clustering_pvalue']:.4f})",
                "",
            ])

        lines.extend([
            f"**Cluster vs Poisson:**",
            f"- Fano difference: {disc_cluster['fano_diff']:.3f} (z = {disc_cluster['fano_z']:.2f}, p = {disc_cluster['fano_pvalue']:.4f})",
            f"- Clustering difference: {disc_cluster['clustering_diff']:.4f} (z = {disc_cluster['clustering_z']:.2f}, p = {disc_cluster['clustering_pvalue']:.4f})",
            "",
        ])

    # Model interpretation
    lines.extend([
        "## Model Interpretation (in M terms)",
        "",
        "### Poisson Model (CDM-like)",
        "- Independent M~2 subhalos scattered uniformly around the Einstein ring",
        "- Variance = Mean (Fano factor ≈ 1)",
        "- No spatial clustering beyond random",
        "",
        "### Cox Model (Lens-to-lens variation)",
        "- Each lens has a different 'domain state' affecting local subhalo abundance",
        "- Creates over-dispersion (Fano > 1) without spatial clustering",
        "- Represents lens-to-lens intermittency but not within-lens clustering",
        "",
        "### Cluster Model (Domain pockets)",
        "- Rare M>2 'domain pockets' spawn clusters of perturbers",
        "- Creates BOTH over-dispersion AND spatial clustering along the arc",
        "- **This is the true 'pocket smoking gun'**",
        "",
    ])

    # Key insight
    lines.extend([
        "## Critical Insight",
        "",
        "**The true domain/pocket signature is NOT just over-dispersion.**",
        "",
        "- Cox model shows Fano > 1 but clustering C ≈ uniform",
        "- Cluster model shows Fano > 1 AND clustering C > uniform",
        "",
        "If real data shows high Fano but normal clustering → could be explained by",
        "host-to-host variation (Cox-like), not pockets.",
        "",
        "If real data shows high Fano AND high clustering → stronger evidence for pockets.",
        "",
    ])

    # Sweep results if available
    if sweep_stats:
        lines.extend([
            "## Sample Size Requirements",
            "",
            "### Fano Factor Detection",
            "",
        ])

        for model in ['cox', 'cluster']:
            if model not in sweep_stats:
                continue
            lines.append(f"**{model.capitalize()} vs Poisson:**")
            for i, s in enumerate(sweep_stats[model]):
                ps = sweep_stats['poisson'][i]
                det = compute_detectability(ps, s)
                det_str = "DETECTABLE" if det['fano_detectable'] else "not detectable"
                lines.append(f"- N = {s.n_lenses}: Fano p = {det['fano_pvalue']:.4f} ({det_str})")
            lines.append("")

        lines.extend([
            "### Clustering Detection",
            "",
        ])

        for model in ['cox', 'cluster']:
            if model not in sweep_stats:
                continue
            lines.append(f"**{model.capitalize()} vs Poisson:**")
            for i, s in enumerate(sweep_stats[model]):
                ps = sweep_stats['poisson'][i]
                det = compute_detectability(ps, s)
                det_str = "DETECTABLE" if det['clustering_detectable'] else "not detectable"
                lines.append(f"- N = {s.n_lenses}: Clustering p = {det['clustering_pvalue']:.4f} ({det_str})")
            lines.append("")

    # Degeneracy warning
    lines.extend([
        "## Degeneracy Warning",
        "",
        "**Cox can mimic Fano but NOT clustering:**",
        "- If you only measure over-dispersion (Fano > 1), you cannot distinguish",
        "  Cox (host variation) from Cluster (pockets)",
        "",
        "**Host-to-host scatter can explain high Fano:**",
        "- Even after conditioning on observed host properties (H, z), unobserved",
        "  scatter can create Fano > 1",
        "- This is NOT evidence for pockets",
        "",
        "**Spatial clustering is the discriminating signal:**",
        "- Cluster model elevates C above uniform expectation",
        "- Cox model does NOT affect C",
        "- This is why clustering metric is essential",
        "",
    ])

    # Technical details
    lines.extend([
        "## Technical Details",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Runtime | {runtime:.2f} s |",
    ])

    if 'poisson' in results:
        r = results['poisson']
        lines.append(f"| Mean perturbers/lens | {r.mean_count:.2f} |")
        lines.append(f"| N_lenses | {r.n_lenses} |")

    if 'poisson' in stats_dict:
        theta0 = 0.3  # default
        C_uniform = expected_uniform_clustering(theta0)
        lines.append(f"| Clustering threshold θ₀ | {theta0} rad |")
        lines.append(f"| Expected C (uniform) | {C_uniform:.4f} |")

    lines.extend([
        "",
        "---",
        "",
        "*Report generated by msoup_substructure_sim*",
    ])

    content = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(content)

    return content


def run_full_analysis(n_lenses: int = 10000,
                       seed: int = 42,
                       output_dir: Path = None,
                       do_sweep: bool = False,
                       verbose: bool = True) -> Dict:
    """
    Run full analysis comparing all three models.
    """
    start_time = time.time()

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('results') / f'substructure_sim_{timestamp}'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Run comparison
    if verbose:
        print("=" * 60)
        print("MSOUP SUBSTRUCTURE SIMULATION")
        print("=" * 60)

    results = run_comparison(n_lenses, seed, verbose)

    # Compute statistics
    if verbose:
        print("\nComputing statistics...")

    stats_dict = {}
    for model, result in results.items():
        stats = analyze_result(result, theta0=0.3, tail_k=3.0, n_bootstrap=100, rng=rng)
        stats_dict[model] = stats

    # Sweep if requested
    sweep_stats = {}
    if do_sweep:
        if verbose:
            print("\nRunning sample size sweep...")

        n_values = [100, 300, 1000, 3000, 10000]
        sweep_results = run_sweep(n_values, seed, verbose)

        for model in ['poisson', 'cox', 'cluster']:
            sweep_stats[model] = []
            for result in sweep_results[model]:
                stats = analyze_result(result, theta0=0.3, rng=rng)
                sweep_stats[model].append(stats)

    # Generate plots
    if HAS_MATPLOTLIB:
        if verbose:
            print("\nGenerating plots...")
        generate_all_plots(results, stats_dict, output_dir, theta0=0.3)

        if do_sweep:
            generate_sweep_plots(sweep_stats, output_dir, theta0=0.3)
    else:
        if verbose:
            print("Skipping plots (matplotlib not available)")

    # Generate report
    total_runtime = time.time() - start_time
    report_path = output_dir / "report.md"
    report = generate_report(results, stats_dict, sweep_stats, report_path, total_runtime)

    # Save summary JSON
    summary = {
        'n_lenses': n_lenses,
        'seed': seed,
        'runtime_seconds': total_runtime,
        'models': {}
    }
    for model, s in stats_dict.items():
        summary['models'][model] = {
            'mean_count': float(s.mean_count),
            'fano_factor': float(s.fano_factor),
            'fano_se': float(s.fano_se),
            'clustering_C': float(s.clustering_C),
            'clustering_C_se': float(s.clustering_C_se),
            'residual_dispersion': float(s.residual_dispersion),
            'tail_prob': float(s.tail_prob),
        }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nTotal runtime: {total_runtime:.2f}s")
        print(f"Results saved to: {output_dir}")
        print("\n" + format_stats_table(list(stats_dict.values())))

    return {
        'results': results,
        'stats': stats_dict,
        'sweep_stats': sweep_stats,
        'report': report,
        'runtime': total_runtime,
    }


# ============================================================================
# WINDOW ANALYSIS
# ============================================================================

def run_window_comparison(n_lenses: int = 10000,
                          seed: int = 42,
                          detection_mode: str = "thin",
                          verbose: bool = True) -> Dict[str, SimulationResult]:
    """
    Run all three window models and return results for comparison.

    Parameters
    ----------
    n_lenses : int
        Number of lenses per model
    seed : int
        Random seed
    detection_mode : str
        "thin" for probabilistic thinning, "threshold" for threshold modulation
    verbose : bool
        Print progress

    Returns
    -------
    Dict mapping model name to SimulationResult
    """
    results = {}

    for base_model in ["poisson", "cox", "cluster"]:
        model_name = f"{base_model}_window"

        if verbose:
            print(f"Running {model_name} model ({detection_mode} mode) with {n_lenses} lenses...")

        config = SimConfig(n_lenses=n_lenses, seed=seed, model=model_name)
        config.window.detection_mode = detection_mode
        config = calibrate_models_to_match_mean(config)
        result = run_simulation(config, calibrate=False)
        results[model_name] = result

        if verbose:
            print(f"  Mean count: {result.mean_count:.3f}")
            print(f"  Fano factor: {result.fano_factor:.3f}")

    return results


def run_window_sweep(kappa_arc_values: List[float] = None,
                     sigma_arc_values: List[float] = None,
                     gamma_values: List[float] = None,
                     n_lenses: int = 5000,
                     seed: int = 42,
                     detection_mode: str = "thin",
                     verbose: bool = True) -> Dict:
    """
    Run parameter sweep over window configurations.

    Returns dict with structure:
    {
        'params': [(kappa, sigma, gamma), ...],
        'poisson_window': [result, ...],
        'C_obs': [...],
        'C_excess': [...],
        'Z_mean': [...]
    }
    """
    if kappa_arc_values is None:
        kappa_arc_values = [1.0, 2.0, 3.0, 5.0]
    if sigma_arc_values is None:
        sigma_arc_values = [0.2, 0.4, 0.6]
    if gamma_values is None:
        gamma_values = [1.0, 1.5, 2.0, 3.0]

    results = {
        'params': [],
        'poisson_window': [],
        'C_obs': [],
        'C_excess': [],
        'Z_mean': [],
        'global_pvalue': []
    }

    total = len(kappa_arc_values) * len(sigma_arc_values) * len(gamma_values)
    count = 0

    for kappa in kappa_arc_values:
        for sigma in sigma_arc_values:
            for gamma in gamma_values:
                count += 1
                if verbose:
                    print(f"[{count}/{total}] kappa={kappa}, sigma={sigma}, gamma={gamma}")

                config = SimConfig(
                    n_lenses=n_lenses,
                    seed=seed,
                    model="poisson_window"
                )
                config.window.kappa_arc = kappa
                config.window.sigma_arc = sigma
                config.window.gamma = gamma
                config.window.detection_mode = detection_mode
                config = calibrate_models_to_match_mean(config)

                result = run_simulation(config, calibrate=False)

                # Compute debiased clustering
                rng = np.random.default_rng(seed + count)
                if result.windows is not None:
                    debias = compute_debiased_clustering(
                        result.theta_list,
                        result.windows,
                        theta0=0.3,
                        n_resamples=100,
                        rng=rng
                    )
                    C_obs = debias.C_obs_mean
                    C_excess = debias.C_excess_mean
                    Z_mean = debias.Z_mean
                    pvalue = debias.global_pvalue
                else:
                    C_obs = np.nan
                    C_excess = np.nan
                    Z_mean = np.nan
                    pvalue = np.nan

                results['params'].append((kappa, sigma, gamma))
                results['poisson_window'].append(result)
                results['C_obs'].append(C_obs)
                results['C_excess'].append(C_excess)
                results['Z_mean'].append(Z_mean)
                results['global_pvalue'].append(pvalue)

                if verbose:
                    print(f"    C_obs={C_obs:.4f}, C_excess={C_excess:.4f}, Z={Z_mean:.3f}")

    return results


def generate_window_report(results: Dict[str, SimulationResult],
                           stats_dict: Dict[str, StatsSummary],
                           debias_dict: Dict[str, DebiasedClusteringStats],
                           sweep_results: Dict,
                           output_path: Path,
                           runtime: float,
                           detection_mode: str) -> str:
    """Generate markdown report for window analysis."""
    lines = [
        "# Msoup Substructure Simulation: Arc Sensitivity Window Analysis",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This simulation tests whether unequal arc sensitivity along the Einstein ring",
        "can create spurious clustering C even if true perturbers are uniform (Poisson)",
        "or only over-dispersed (Cox).",
        "",
        f"**Detection mode:** {detection_mode}",
        "",
    ]

    # Key findings table
    if debias_dict:
        lines.extend([
            "### Key Findings",
            "",
            "| Model | C_obs | C_excess | Z_mean | p-value |",
            "|-------|-------|----------|--------|---------|",
        ])

        for model in ['poisson_window', 'cox_window', 'cluster_window']:
            if model in debias_dict:
                d = debias_dict[model]
                lines.append(
                    f"| {model} | {d.C_obs_mean:.4f} | "
                    f"{d.C_excess_mean:.4f} | {d.Z_mean:.3f} | {d.global_pvalue:.4f} |"
                )

        lines.append("")

    # Interpretation
    lines.extend([
        "## Interpretation",
        "",
        "### Selection Effect on Clustering",
        "",
        "- **C_obs**: Raw observed clustering (affected by selection)",
        "- **C_excess**: Clustering above null (C_obs - E[C|selection])",
        "- **Z_mean**: Standardized excess ((C_obs - E[C|selection]) / σ_null)",
        "",
        "**Key insight:**",
        "- If Poisson_window has C_excess ≈ 0 and Z ≈ 0, selection doesn't create spurious clustering",
        "- If Poisson_window has C_excess > 0 and Z > 2, selection CAN create spurious clustering",
        "- Cluster_window should still show elevated C_excess (real pockets detectable)",
        "",
    ])

    # Sweep results if available
    if sweep_results and 'params' in sweep_results:
        lines.extend([
            "## Parameter Sweep Results",
            "",
            "### Maximum Spurious Clustering from Selection",
            "",
        ])

        # Find max C_obs from Poisson_window sweep
        C_obs_arr = np.array(sweep_results['C_obs'])
        Z_arr = np.array(sweep_results['Z_mean'])
        params = sweep_results['params']

        if len(C_obs_arr) > 0 and not np.all(np.isnan(C_obs_arr)):
            max_idx = np.nanargmax(C_obs_arr)
            max_C = C_obs_arr[max_idx]
            max_params = params[max_idx]
            max_Z = Z_arr[max_idx]

            lines.extend([
                f"**Maximum C_obs achieved: {max_C:.4f}**",
                f"- At kappa_arc={max_params[0]}, sigma_arc={max_params[1]}, gamma={max_params[2]}",
                f"- Z score: {max_Z:.3f}",
                "",
            ])

            # Compare to cluster baseline (~0.32)
            cluster_target = 0.32
            if max_C >= 0.25:
                lines.append(f"⚠️ **WARNING**: Selection can achieve C ≥ 0.25 (cluster target ~{cluster_target})")
                lines.append("Selection effects could potentially mimic pocket clustering!")
            else:
                lines.append(f"✓ Selection alone cannot reach cluster-level clustering (~{cluster_target})")
                lines.append("Pocket signal remains distinguishable from selection artifacts.")

            lines.append("")

            # Show top 5 parameter combinations
            lines.extend([
                "### Top 5 Configurations by C_obs",
                "",
                "| Rank | kappa_arc | sigma_arc | gamma | C_obs | C_excess | Z |",
                "|------|-----------|-----------|-------|-------|----------|---|",
            ])

            sorted_idx = np.argsort(C_obs_arr)[::-1][:5]
            for rank, idx in enumerate(sorted_idx, 1):
                if not np.isnan(C_obs_arr[idx]):
                    p = params[idx]
                    lines.append(
                        f"| {rank} | {p[0]} | {p[1]} | {p[2]} | "
                        f"{C_obs_arr[idx]:.4f} | {sweep_results['C_excess'][idx]:.4f} | "
                        f"{Z_arr[idx]:.3f} |"
                    )

            lines.append("")

    # Technical details
    lines.extend([
        "## Technical Details",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| Runtime | {runtime:.2f} s |",
        f"| Detection mode | {detection_mode} |",
    ])

    if 'poisson_window' in results:
        r = results['poisson_window']
        lines.append(f"| Mean perturbers/lens | {r.mean_count:.2f} |")
        lines.append(f"| N_lenses | {r.n_lenses} |")

    lines.extend([
        f"| Clustering threshold θ₀ | 0.3 rad |",
        f"| Expected C (uniform) | {expected_uniform_clustering(0.3):.4f} |",
        "",
        "---",
        "",
        "*Report generated by msoup_substructure_sim (window analysis)*",
    ])

    content = "\n".join(lines)

    with open(output_path, 'w') as f:
        f.write(content)

    return content


def run_window_analysis(n_lenses: int = 10000,
                        seed: int = 42,
                        detection_mode: str = "thin",
                        output_dir: Path = None,
                        do_sweep: bool = False,
                        debias: bool = True,
                        verbose: bool = True) -> Dict:
    """
    Run full window analysis comparing selection effects across models.
    """
    start_time = time.time()

    if output_dir is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('results') / f'substructure_sim_window_{timestamp}'
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)

    # Run comparison
    if verbose:
        print("=" * 60)
        print("MSOUP SUBSTRUCTURE SIMULATION - WINDOW ANALYSIS")
        print("=" * 60)
        print(f"Detection mode: {detection_mode}")
        print("")

    results = run_window_comparison(n_lenses, seed, detection_mode, verbose)

    # Compute statistics
    if verbose:
        print("\nComputing statistics...")

    stats_dict = {}
    debias_dict = {}

    for model, result in results.items():
        stats = analyze_result(result, theta0=0.3, tail_k=3.0, n_bootstrap=100, rng=rng)
        stats_dict[model] = stats

        # Compute debiased clustering if windows available
        if debias and result.windows is not None:
            if verbose:
                print(f"  Computing debiased clustering for {model}...")

            debias_stats = compute_debiased_clustering(
                result.theta_list,
                result.windows,
                theta0=0.3,
                n_resamples=200,
                rng=rng
            )
            debias_stats = DebiasedClusteringStats(
                model_name=model,
                n_lenses=debias_stats.n_lenses,
                n_with_pairs=debias_stats.n_with_pairs,
                C_obs_mean=debias_stats.C_obs_mean,
                C_obs_se=debias_stats.C_obs_se,
                C_excess_mean=debias_stats.C_excess_mean,
                C_excess_se=debias_stats.C_excess_se,
                Z_mean=debias_stats.Z_mean,
                Z_se=debias_stats.Z_se,
                C_obs_values=debias_stats.C_obs_values,
                C_null_values=debias_stats.C_null_values,
                C_excess_values=debias_stats.C_excess_values,
                Z_values=debias_stats.Z_values,
                global_pvalue=debias_stats.global_pvalue
            )
            debias_dict[model] = debias_stats

    # Sweep if requested
    sweep_results = {}
    if do_sweep:
        if verbose:
            print("\nRunning window parameter sweep...")

        sweep_results = run_window_sweep(
            n_lenses=min(5000, n_lenses),
            seed=seed,
            detection_mode=detection_mode,
            verbose=verbose
        )

    # Generate plots
    if HAS_MATPLOTLIB:
        if verbose:
            print("\nGenerating plots...")
        generate_window_plots(results, debias_dict, sweep_results, output_dir)
    else:
        if verbose:
            print("Skipping plots (matplotlib not available)")

    # Generate report
    total_runtime = time.time() - start_time
    report_path = output_dir / "report.md"
    report = generate_window_report(
        results, stats_dict, debias_dict, sweep_results,
        report_path, total_runtime, detection_mode
    )

    # Save summary JSON
    summary = {
        'n_lenses': n_lenses,
        'seed': seed,
        'detection_mode': detection_mode,
        'runtime_seconds': total_runtime,
        'models': {}
    }

    for model, s in stats_dict.items():
        model_summary = {
            'mean_count': float(s.mean_count),
            'fano_factor': float(s.fano_factor),
            'clustering_C': float(s.clustering_C),
        }

        if model in debias_dict:
            d = debias_dict[model]
            model_summary['C_obs'] = float(d.C_obs_mean)
            model_summary['C_excess'] = float(d.C_excess_mean)
            model_summary['Z_mean'] = float(d.Z_mean)
            model_summary['global_pvalue'] = float(d.global_pvalue)

        summary['models'][model] = model_summary

    if sweep_results:
        summary['sweep'] = {
            'n_configs': len(sweep_results['params']),
            'max_C_obs': float(np.nanmax(sweep_results['C_obs'])) if sweep_results['C_obs'] else None,
        }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    if verbose:
        print(f"\nTotal runtime: {total_runtime:.2f}s")
        print(f"Results saved to: {output_dir}")

        if debias_dict:
            print("\n" + format_debiased_stats_table(list(debias_dict.values())))

    return {
        'results': results,
        'stats': stats_dict,
        'debias_stats': debias_dict,
        'sweep_results': sweep_results,
        'report': report,
        'runtime': total_runtime,
    }


def main():
    """CLI main function."""
    parser = argparse.ArgumentParser(
        description="Msoup Substructure Simulation: Compare Poisson, Cox, and Cluster models"
    )
    parser.add_argument('--model', type=str, default='all',
                        choices=['poisson', 'cox', 'cluster', 'all',
                                 'poisson_window', 'cox_window', 'cluster_window'],
                        help='Model to run (default: all)')
    parser.add_argument('--N_lenses', type=int, default=10000,
                        help='Number of lenses (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run sample size sweep')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    # Window-specific options
    parser.add_argument('--use-windows', type=str, default='none',
                        choices=['none', 'thin', 'threshold'],
                        help='Detection mode for window models (default: none)')
    parser.add_argument('--sweep-windows', action='store_true',
                        help='Run parameter sweep over window configurations')
    parser.add_argument('--debias-clustering', action='store_true',
                        help='Compute debiased clustering statistics (C_excess, Z)')

    args = parser.parse_args()

    # Window analysis mode
    if args.use_windows != 'none' or args.sweep_windows:
        detection_mode = args.use_windows if args.use_windows != 'none' else 'thin'
        output_dir = Path(args.outdir) if args.outdir else None

        if output_dir is None:
            output_dir = Path('results') / 'substructure_sim_window'

        run_window_analysis(
            n_lenses=args.N_lenses,
            seed=args.seed,
            detection_mode=detection_mode,
            output_dir=output_dir,
            do_sweep=args.sweep_windows,
            debias=args.debias_clustering or True,  # Always debias for window analysis
            verbose=not args.quiet
        )
        return

    if args.model == 'all':
        # Run full comparison
        output_dir = Path(args.outdir) if args.outdir else None
        run_full_analysis(
            n_lenses=args.N_lenses,
            seed=args.seed,
            output_dir=output_dir,
            do_sweep=args.sweep,
            verbose=not args.quiet
        )
    else:
        # Run single model
        config = SimConfig(n_lenses=args.N_lenses, seed=args.seed, model=args.model)
        config = calibrate_models_to_match_mean(config)
        result = run_simulation(config, calibrate=False)

        if not args.quiet:
            print(f"Model: {args.model}")
            print(f"N_lenses: {args.N_lenses}")
            print(f"Mean count: {result.mean_count:.3f}")
            print(f"Fano factor: {result.fano_factor:.3f}")


if __name__ == '__main__':
    main()
