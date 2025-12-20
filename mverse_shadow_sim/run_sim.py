"""
CLI Entrypoint for Mverse Shadow Simulation

Run the full simulation pipeline to calibrate K1/K2 kill conditions.
"""

import argparse
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import json

from .config import SimConfig, get_default_config
from .truth import generate_cluster_population, clusters_to_arrays
from .observables import generate_all_observables, generate_all_observables_vectorized
from .inference import run_inference_vectorized, run_inference_full
from .metrics import compute_all_metrics, format_metrics_table
from .report import generate_report

# Conditional import for plots
try:
    from .plots import generate_all_plots, HAS_MATPLOTLIB
except ImportError:
    HAS_MATPLOTLIB = False


def run_simulation(config: SimConfig,
                   output_dir: Path,
                   verbose: bool = True) -> dict:
    """
    Run the full simulation pipeline.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration
    output_dir : Path
        Output directory
    verbose : bool
        Print progress

    Returns
    -------
    dict
        Results dictionary
    """
    start_time = time.time()

    rng = np.random.default_rng(config.seed)

    # Step 1: Generate truth
    if verbose:
        print(f"Generating {config.n_clusters} clusters...")
    clusters = generate_cluster_population(config, rng)

    truth_time = time.time()
    if verbose:
        print(f"  Truth generation: {truth_time - start_time:.2f}s")

    # Step 2: Generate observables
    if verbose:
        print("Generating observables...")

    if config.fast_mode:
        lensing, dynamics = generate_all_observables_vectorized(clusters, config, rng)
    else:
        lensing_obs, dynamics_obs = generate_all_observables(clusters, config, rng)

    obs_time = time.time()
    if verbose:
        print(f"  Observable generation: {obs_time - truth_time:.2f}s")

    # Step 3: Run inference
    if verbose:
        print("Running inference...")

    if config.fast_mode:
        inference_results = run_inference_vectorized(clusters, lensing, dynamics, config, rng)
    else:
        inference_results = run_inference_full(clusters, lensing_obs, dynamics_obs, config, rng)

    inf_time = time.time()
    if verbose:
        print(f"  Inference: {inf_time - obs_time:.2f}s")

    # Step 4: Compute metrics
    if verbose:
        print("Computing metrics...")

    metrics_r500 = compute_all_metrics(inference_results, "r500", rng)
    metrics_half = compute_all_metrics(inference_results, "half_r500", rng)

    met_time = time.time()
    if verbose:
        print(f"  Metrics: {met_time - inf_time:.2f}s")

    # Step 5: Generate plots
    if HAS_MATPLOTLIB:
        if verbose:
            print("Generating plots...")
        generate_all_plots(inference_results, output_dir, "r500")
        generate_all_plots(inference_results, output_dir, "half_r500")
        plot_time = time.time()
        if verbose:
            print(f"  Plots: {plot_time - met_time:.2f}s")
    else:
        if verbose:
            print("Skipping plots (matplotlib not available)")
        plot_time = met_time

    # Step 6: Generate report
    total_runtime = plot_time - start_time
    report_path = output_dir / "report.md"
    report_content = generate_report(config, metrics_r500, metrics_half,
                                      report_path, total_runtime)

    if verbose:
        print(f"\nTotal runtime: {total_runtime:.2f}s")
        print(f"Results saved to: {output_dir}")

    # Save summary JSON
    summary = {
        'n_clusters': config.n_clusters,
        'seed': config.seed,
        'fast_mode': config.fast_mode,
        'runtime_seconds': total_runtime,
        'metrics_r500': {
            'R_X_median': float(metrics_r500.k1.R_X_median),
            'R_X_std': float(metrics_r500.k1.R_X_std),
            'R_V_median': float(metrics_r500.k1.R_V_median),
            'R_V_std': float(metrics_r500.k1.R_V_std),
            'K1_tolerance': float(metrics_r500.recommended_K1_tolerance),
            'K2_tolerance_mass': float(metrics_r500.recommended_K2_tolerance_mass),
            'K2_tolerance_z': float(metrics_r500.recommended_K2_tolerance_z),
            'slope_mass_X': float(metrics_r500.k2.slope_mass_X),
            'slope_z_X': float(metrics_r500.k2.slope_z_X),
        },
        'metrics_half_r500': {
            'R_X_median': float(metrics_half.k1.R_X_median),
            'R_X_std': float(metrics_half.k1.R_X_std),
            'R_V_median': float(metrics_half.k1.R_V_median),
            'R_V_std': float(metrics_half.k1.R_V_std),
            'K1_tolerance': float(metrics_half.recommended_K1_tolerance),
        }
    }

    with open(output_dir / "summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    # Save raw R values for external analysis
    np.savez(output_dir / "R_values.npz",
             R_X_r500=inference_results['R_X_r500'],
             R_V_r500=inference_results['R_V_r500'],
             R_X_half_r500=inference_results['R_X_half_r500'],
             R_V_half_r500=inference_results['R_V_half_r500'],
             log_M200=inference_results['log_M200_true'],
             z=inference_results['z'],
             b_hse_true=inference_results['b_hse_true'],
             b_aniso_true=inference_results['b_aniso_true'],
             proj_boost=inference_results['proj_boost'])

    return {
        'config': config,
        'clusters': clusters,
        'inference_results': inference_results,
        'metrics_r500': metrics_r500,
        'metrics_half': metrics_half,
        'report': report_content,
        'runtime': total_runtime,
    }


def main():
    """CLI main function."""
    parser = argparse.ArgumentParser(
        description="Mverse Shadow Simulation: Calibrate K1/K2 kill conditions"
    )
    parser.add_argument('--N', type=int, default=10000,
                        help='Number of clusters (default: 10000)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--fast', action='store_true',
                        help='Use fast vectorized mode (default for N>1000)')
    parser.add_argument('--full', action='store_true',
                        help='Use full per-object fitting mode')
    parser.add_argument('--outdir', type=str, default=None,
                        help='Output directory (default: results/shadow_sim_TIMESTAMP)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress progress output')

    args = parser.parse_args()

    # Determine mode
    if args.full:
        fast_mode = False
    elif args.fast:
        fast_mode = True
    else:
        # Default: fast for large N, full for small N
        fast_mode = args.N > 1000

    # Create config
    config = get_default_config(n_clusters=args.N, seed=args.seed, fast_mode=fast_mode)

    # Output directory
    if args.outdir:
        output_dir = Path(args.outdir)
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = Path('results') / f'shadow_sim_{timestamp}'

    output_dir.mkdir(parents=True, exist_ok=True)

    # Run simulation
    results = run_simulation(config, output_dir, verbose=not args.quiet)

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("SIMULATION COMPLETE")
        print("=" * 60)
        print(format_metrics_table(results['metrics_r500']))

        k1 = results['metrics_r500'].k1
        print("\n" + "-" * 60)
        print("RECOMMENDED THRESHOLDS FOR REAL DATA")
        print("-" * 60)
        print(f"K1 tolerance (|R-1|): {results['metrics_r500'].recommended_K1_tolerance:.3f}")
        print(f"K2 mass slope tolerance: {results['metrics_r500'].recommended_K2_tolerance_mass:.4f}")
        print(f"K2 z slope tolerance: {results['metrics_r500'].recommended_K2_tolerance_z:.4f}")
        print()
        print(f"At K1 tolerance, violation fractions under null:")
        print(f"  R_X: {k1.violation_fracs_X.get(0.3, 'N/A'):.1%}")
        print(f"  R_V: {k1.violation_fracs_V.get(0.3, 'N/A'):.1%}")
        print()
        print(f"Output: {output_dir}")


if __name__ == '__main__':
    main()
