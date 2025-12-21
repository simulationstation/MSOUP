"""CLI for the Msoup ejection + decompression simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict
from datetime import datetime

from .config import SimulationConfig, CalibrationConfig
from .dynamics import run_simulation
from .calibrate import run_calibration, run_stage1_calibration, run_stage2_calibration
from .report import generate_report
from .inference import infer_expansion, morphology_stats


DEFAULT_RESULTS_DIR = Path("results") / "msoup_ejection_sim"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Msoup ejection + decompression simulation")
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke test")
    parser.add_argument("--fast", action="store_true", help="Fast mode: reduced grid/steps/seeds for quick iteration")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration loop (legacy)")
    parser.add_argument("--max-evals", type=int, default=4000, help="Max evaluations for calibration")
    parser.add_argument("--grid", type=int, default=256, help="Grid size")
    parser.add_argument("--steps", type=int, default=400, help="Number of time steps")
    parser.add_argument("--params", type=str, help="Path to parameter JSON file")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Where to store outputs")
    parser.add_argument("--timestamped", action="store_true", help="Create timestamped output subfolder")

    # New scientific modes
    parser.add_argument("--stage1-calibrate", action="store_true",
                        help="Stage 1 calibration: fit ONLY H0_early, observe emergent H0_late")
    parser.add_argument("--stage2-calibrate", action="store_true",
                        help="Stage 2 calibration: fit both H0_early and H0_late")
    parser.add_argument("--multiseed", type=int, metavar="N",
                        help="Run multi-seed robustness analysis with N seeds")
    parser.add_argument("--convergence", action="store_true",
                        help="Run resolution convergence analysis")
    parser.add_argument("--identifiability-scan", type=int, metavar="K",
                        help="Run identifiability scan with K parameter samples")
    parser.add_argument("--full-science", action="store_true",
                        help="Run full science pipeline: stage1 + multiseed + convergence + identifiability")
    parser.add_argument("--full-science-v2", action="store_true",
                        help="Run enhanced science pipeline with train/val split, multireadout, amplifier kill")
    parser.add_argument("--full-science-v3", action="store_true",
                        help="Run v3 science pipeline with M3 container field K")

    # K field controls (v3)
    parser.add_argument("--K-enabled", action="store_true", default=True,
                        help="Enable M3 container field K (default: True)")
    parser.add_argument("--K-disabled", action="store_true",
                        help="Disable M3 container field K for backward compatibility")
    parser.add_argument("--compare-K", action="store_true",
                        help="Compare K-enabled vs K-disabled results")

    # Readout and amplifier modes
    parser.add_argument("--readout", type=str, choices=["volume", "tracer", "both"], default="both",
                        help="Late-time readout mode: volume, tracer, or both (default: both)")
    parser.add_argument("--amplifier", type=str, choices=["none", "threshold"], default="none",
                        help="Amplifier mode: none or threshold (default: none)")
    parser.add_argument("--amplifier-kill-test", action="store_true",
                        help="Run amplifier kill test comparing baseline vs amplified")

    return parser.parse_args()


def load_params(path: str, base_cfg: SimulationConfig) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    cfg_dict = base_cfg.to_dict()
    cfg_dict.update(params)
    cfg_dict.update({"grid": base_cfg.grid, "steps": base_cfg.steps})
    return SimulationConfig.from_dict(cfg_dict)


def run_science_pipeline_v3(args, base_cfg: SimulationConfig, results_dir: str):
    """Run the v3 science pipeline with M3 container field K.

    This pipeline tests whether the K field (container constraint) produces
    a larger emergent ΔH0 while maintaining Stage-1 discipline and robustness.
    """
    from .analysis import (
        run_multiseed, run_convergence, run_identifiability_scan,
        aggregate_environment_curves_multiseed, aggregate_fingerprints_multiseed,
        run_train_val_split, run_multireadout_comparison
    )
    from .science_report import generate_science_report_v3

    runtime = get_runtime_settings(args)
    print("=== Msoup Ejection Simulation: v3 Science Pipeline (K Container) ===")
    print(f"Mode: {'smoke' if args.smoke else 'fast' if args.fast else 'full'}")
    print(f"K_enabled: {base_cfg.K_enabled}")
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Stage 1 calibration (early-only, does NOT target H0_late or 73)
    print("\n[1/9] Running Stage 1 calibration (early-only, K_enabled={})...".format(base_cfg.K_enabled))
    calib_cfg = CalibrationConfig(
        max_evals=max(500, args.max_evals // 4) if args.smoke else args.max_evals,
        smoke=args.smoke,
        grid_calib=runtime["grid"] // 2,
        steps_calib=runtime["steps"] // 2
    )
    stage1_result = run_stage1_calibration(base_cfg, calib_cfg, results_dir)
    print(f"  H0_early achieved: {stage1_result['H0_early_achieved']:.3f}")
    print(f"  H0_late emergent: {stage1_result['H0_late_emergent']:.3f}")
    print(f"  ΔH0 emergent: {stage1_result['Delta_H0_emergent']:.3f}")
    print(f"  Deviation from 73: {stage1_result['H0_late_deviation_from_73']:+.3f}")

    # Load stage1 params
    with open(results_path / "stage1_params.json", "r") as f:
        stage1_params = json.load(f)

    cfg_dict = base_cfg.to_dict()
    cfg_dict.update(stage1_params)
    cfg_dict.update({"grid": runtime["grid"], "steps": runtime["steps"]})
    stage1_cfg = SimulationConfig.from_dict(cfg_dict)

    # Train/Val/Stress split
    print(f"\n[2/9] Running train/val/stress split (train={runtime['n_train']}, val={runtime['n_val']}, stress={runtime['n_stress']})...")
    train_val_split = run_train_val_split(stage1_cfg, n_train=runtime["n_train"],
                                          n_val=runtime["n_val"], n_stress=runtime["n_stress"])
    print(f"  Train ΔH0: {train_val_split.train_results.Delta_H0_mean:.3f} +/- {train_val_split.train_results.Delta_H0_std:.3f}")
    print(f"  Val ΔH0: {train_val_split.val_results.Delta_H0_mean:.3f} +/- {train_val_split.val_results.Delta_H0_std:.3f}")
    print(f"  Stress ΔH0: {train_val_split.stress_results.Delta_H0_mean:.3f} +/- {train_val_split.stress_results.Delta_H0_std:.3f}")

    # v3: K diagnostics from train/val
    if train_val_split.val_results.K_release_time_mean is not None:
        print(f"  K release time: {train_val_split.val_results.K_release_time_mean:.3f}")
    if train_val_split.val_results.wlt2_onset_time_mean is not None:
        print(f"  wlt2 onset time: {train_val_split.val_results.wlt2_onset_time_mean:.3f}")

    with open(results_path / "train_val_split.json", "w") as f:
        json.dump({
            "train_seeds": train_val_split.train_seeds,
            "val_seeds": train_val_split.val_seeds,
            "stress_seeds": train_val_split.stress_seeds,
            "train_results": asdict(train_val_split.train_results),
            "val_results": asdict(train_val_split.val_results),
            "stress_results": asdict(train_val_split.stress_results),
        }, f, indent=2)

    # Multi-seed for K diagnostics
    print(f"\n[3/9] Running multi-seed analysis (N={runtime['n_seeds']})...")
    multiseed = run_multiseed(stage1_cfg, n_seeds=runtime["n_seeds"])
    print(f"  ΔH0: {multiseed.Delta_H0_mean:.3f} +/- {multiseed.Delta_H0_std:.3f}")
    print(f"  H0_late: {multiseed.H0_late_mean:.3f} +/- {multiseed.H0_late_std:.3f}")
    if multiseed.K_release_time_mean is not None:
        print(f"  K release time: {multiseed.K_release_time_mean:.3f} +/- {multiseed.K_release_time_std:.3f}")
    print(f"  mean_K_final: {multiseed.mean_K_final_mean:.4f}")

    with open(results_path / "multiseed_summary.json", "w") as f:
        json.dump(asdict(multiseed), f, indent=2)

    # Multi-readout comparison
    print(f"\n[4/9] Running multi-readout comparison...")
    multireadout = run_multireadout_comparison(stage1_cfg, n_seeds=runtime["n_seeds"] // 2)
    print(f"  Volume ΔH0: {multireadout['volume']['Delta_H0_mean']:.3f}")
    print(f"  Tracer ΔH0: {multireadout['tracer']['Delta_H0_mean']:.3f}")

    with open(results_path / "multireadout_comparison.json", "w") as f:
        json.dump(multireadout, f, indent=2)

    # Convergence analysis
    print("\n[5/9] Running convergence analysis...")
    if args.smoke:
        resolutions = [(32, 50), (64, 100)]
    elif args.fast:
        resolutions = [(64, 100), (128, 200)]
    else:
        resolutions = [(128, 250), (256, 400), (384, 600)]
    convergence_summary = run_convergence(stage1_cfg, resolutions=resolutions)
    print(f"  ΔH0 drift: {convergence_summary.Delta_H0_drift:.3f}")
    print(f"  Converged: {'YES' if convergence_summary.converged else 'NO'}")

    with open(results_path / "convergence_summary.json", "w") as f:
        json.dump(asdict(convergence_summary), f, indent=2)

    # Environment curves with K percentile
    print(f"\n[6/9] Computing environment curves (N={runtime['n_env_seeds']})...")
    env_curves = aggregate_environment_curves_multiseed(stage1_cfg, n_seeds=runtime["n_env_seeds"])
    fingerprints = aggregate_fingerprints_multiseed(stage1_cfg, n_seeds=runtime["n_env_seeds"])

    with open(results_path / "environment_curves.json", "w") as f:
        json.dump(env_curves, f, indent=2)
    with open(results_path / "mechanism_fingerprints.json", "w") as f:
        json.dump(fingerprints, f, indent=2)

    # Enhanced identifiability scan with K parameters
    print(f"\n[7/9] Running identifiability scan (K={runtime['n_ident']}) with tau_K0, chi_K...")
    ident_result = run_identifiability_scan(stage1_cfg, n_samples=runtime["n_ident"],
                                            fast_grid=runtime["grid"] // 2,
                                            fast_steps=runtime["steps"] // 2)
    print(f"  Stage 1 acceptance: {100*ident_result.stage1_acceptance_rate:.1f}%")
    print(f"  Stage 2 acceptance: {100*ident_result.stage2_acceptance_rate:.2f}%")
    print(f"  Fine-tuning metric (log10): {ident_result.stage2_fine_tuning_metric:.2f}")

    if ident_result.Delta_H0_importance:
        print("  Top ΔH0 drivers:")
        for imp in ident_result.Delta_H0_importance[:5]:
            print(f"    {imp['param']}: corr={imp['correlation']:+.3f}")

    with open(results_path / "identifiability_summary.json", "w") as f:
        json.dump(asdict(ident_result), f, indent=2)

    # v3: Compare K-enabled vs K-disabled (baseline comparison)
    print("\n[8/9] Running K comparison (K-enabled vs K-disabled)...")
    # Run with K disabled for comparison
    cfg_no_K = SimulationConfig.from_dict(stage1_cfg.to_dict())
    cfg_no_K.K_enabled = False
    cfg_no_K.dm3_to_A_mode = "legacy"

    baseline_summary = run_multiseed(cfg_no_K, n_seeds=min(10, runtime["n_seeds"] // 3),
                                      include_multireadout=False)
    print(f"  K-disabled ΔH0: {baseline_summary.Delta_H0_mean:.3f} +/- {baseline_summary.Delta_H0_std:.3f}")
    print(f"  K-enabled ΔH0: {multiseed.Delta_H0_mean:.3f} +/- {multiseed.Delta_H0_std:.3f}")

    delta_increase = multiseed.Delta_H0_mean - baseline_summary.Delta_H0_mean
    print(f"  ΔH0 increase from K: {delta_increase:+.3f}")

    K_comparison = {
        "K_disabled": {
            "Delta_H0_mean": baseline_summary.Delta_H0_mean,
            "Delta_H0_std": baseline_summary.Delta_H0_std,
            "H0_late_mean": baseline_summary.H0_late_mean,
        },
        "K_enabled": {
            "Delta_H0_mean": multiseed.Delta_H0_mean,
            "Delta_H0_std": multiseed.Delta_H0_std,
            "H0_late_mean": multiseed.H0_late_mean,
            "K_release_time_mean": multiseed.K_release_time_mean,
            "mean_K_final_mean": multiseed.mean_K_final_mean,
        },
        "Delta_H0_increase": delta_increase,
    }
    with open(results_path / "K_comparison.json", "w") as f:
        json.dump(K_comparison, f, indent=2)

    # Generate v3 report
    print("\n[9/9] Generating SCIENCE_REPORT_v3.md...")
    generate_science_report_v3(
        results_dir=results_dir,
        cfg=stage1_cfg,
        stage1_result=stage1_result,
        train_val_split=train_val_split,
        multiseed_summary=multiseed,
        multireadout=multireadout,
        convergence_summary=convergence_summary,
        environment_curves=env_curves,
        fingerprints=fingerprints,
        identifiability_result=ident_result,
        K_comparison=K_comparison,
    )

    print(f"\n=== v3 Science pipeline complete ===")
    print(f"Results saved to: {results_dir}")
    print(f"Main report: {results_dir}/SCIENCE_REPORT_v3.md")


def run_science_pipeline_v2(args, base_cfg: SimulationConfig, results_dir: str):
    """Run the enhanced science pipeline with all v2 features."""
    from .analysis import (
        run_multiseed, run_convergence, run_identifiability_scan,
        aggregate_environment_curves_multiseed, aggregate_fingerprints_multiseed,
        run_train_val_split, run_multireadout_comparison, run_amplifier_kill_test
    )
    from .science_report import generate_science_report_v2

    runtime = get_runtime_settings(args)
    print("=== Msoup Ejection Simulation: Enhanced Science Pipeline (v2) ===")
    print(f"Mode: {'smoke' if args.smoke else 'fast' if args.fast else 'full'}")
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Stage 1 calibration
    print("\n[1/8] Running Stage 1 calibration (early-only)...")
    calib_cfg = CalibrationConfig(
        max_evals=max(500, args.max_evals // 4) if args.smoke else args.max_evals,
        smoke=args.smoke,
        grid_calib=runtime["grid"] // 2,
        steps_calib=runtime["steps"] // 2
    )
    stage1_result = run_stage1_calibration(base_cfg, calib_cfg, results_dir)
    print(f"  H0_early achieved: {stage1_result['H0_early_achieved']:.3f}")
    print(f"  H0_late emergent: {stage1_result['H0_late_emergent']:.3f}")
    print(f"  Deviation from 73: {stage1_result['H0_late_deviation_from_73']:+.3f}")

    # Load stage1 params
    with open(results_path / "stage1_params.json", "r") as f:
        stage1_params = json.load(f)

    cfg_dict = base_cfg.to_dict()
    cfg_dict.update(stage1_params)
    cfg_dict.update({"grid": runtime["grid"], "steps": runtime["steps"]})
    stage1_cfg = SimulationConfig.from_dict(cfg_dict)

    # Train/Val/Stress split
    print(f"\n[2/8] Running train/val/stress split (train={runtime['n_train']}, val={runtime['n_val']}, stress={runtime['n_stress']})...")
    train_val_split = run_train_val_split(stage1_cfg, n_train=runtime["n_train"],
                                          n_val=runtime["n_val"], n_stress=runtime["n_stress"])
    print(f"  Train ΔH0: {train_val_split.train_results.Delta_H0_mean:.3f} +/- {train_val_split.train_results.Delta_H0_std:.3f}")
    print(f"  Val ΔH0: {train_val_split.val_results.Delta_H0_mean:.3f} +/- {train_val_split.val_results.Delta_H0_std:.3f}")
    print(f"  Stress ΔH0: {train_val_split.stress_results.Delta_H0_mean:.3f} +/- {train_val_split.stress_results.Delta_H0_std:.3f}")

    # Save train/val split
    with open(results_path / "train_val_split.json", "w") as f:
        json.dump({
            "train_seeds": train_val_split.train_seeds,
            "val_seeds": train_val_split.val_seeds,
            "stress_seeds": train_val_split.stress_seeds,
            "train_results": asdict(train_val_split.train_results),
            "val_results": asdict(train_val_split.val_results),
            "stress_results": asdict(train_val_split.stress_results),
        }, f, indent=2)

    # Multi-readout comparison
    print(f"\n[3/8] Running multi-readout comparison (N={runtime['n_seeds']})...")
    multireadout = run_multireadout_comparison(stage1_cfg, n_seeds=runtime["n_seeds"])
    print(f"  Volume ΔH0: {multireadout['volume']['Delta_H0_mean']:.3f} +/- {multireadout['volume']['Delta_H0_std']:.3f}")
    print(f"  Tracer ΔH0: {multireadout['tracer']['Delta_H0_mean']:.3f} +/- {multireadout['tracer']['Delta_H0_std']:.3f}")
    print(f"  Difference: {multireadout['difference']['tracer_minus_volume_mean']:+.3f}")

    with open(results_path / "multireadout_comparison.json", "w") as f:
        json.dump(multireadout, f, indent=2)

    # Convergence analysis
    print("\n[4/8] Running convergence analysis...")
    if args.smoke:
        resolutions = [(32, 50), (64, 100)]
    elif args.fast:
        resolutions = [(64, 100), (128, 200)]
    else:
        resolutions = [(128, 250), (256, 400), (384, 600)]
    convergence_summary = run_convergence(stage1_cfg, resolutions=resolutions)
    print(f"  H0_late drift: {convergence_summary.H0_late_drift:.3f}")
    print(f"  Converged: {'YES' if convergence_summary.converged else 'NO'}")

    with open(results_path / "convergence_summary.json", "w") as f:
        json.dump(asdict(convergence_summary), f, indent=2)

    # Non-targeted predictions
    print(f"\n[5/8] Computing environment curves (N={runtime['n_env_seeds']})...")
    env_curves = aggregate_environment_curves_multiseed(stage1_cfg, n_seeds=runtime["n_env_seeds"])
    fingerprints = aggregate_fingerprints_multiseed(stage1_cfg, n_seeds=runtime["n_env_seeds"])

    with open(results_path / "environment_curves.json", "w") as f:
        json.dump(env_curves, f, indent=2)
    with open(results_path / "mechanism_fingerprints.json", "w") as f:
        json.dump(fingerprints, f, indent=2)

    # Enhanced identifiability scan
    print(f"\n[6/8] Running enhanced identifiability scan (K={runtime['n_ident']})...")
    ident_result = run_identifiability_scan(stage1_cfg, n_samples=runtime["n_ident"],
                                            fast_grid=runtime["grid"] // 2,
                                            fast_steps=runtime["steps"] // 2)
    print(f"  Stage 1 valid: {ident_result.n_stage1_valid}/{ident_result.n_samples} ({100*ident_result.stage1_acceptance_rate:.1f}%)")
    print(f"  Stage 2 valid: {ident_result.n_stage2_valid}/{ident_result.n_samples} ({100*ident_result.stage2_acceptance_rate:.1f}%)")
    print(f"  Fine-tuning metric (log10): {ident_result.stage2_fine_tuning_metric:.2f}")

    if ident_result.Delta_H0_importance:
        print("  Top ΔH0 drivers:")
        for imp in ident_result.Delta_H0_importance[:5]:
            print(f"    {imp['param']}: corr={imp['correlation']:+.3f}")

    with open(results_path / "identifiability_summary.json", "w") as f:
        json.dump(asdict(ident_result), f, indent=2)

    # Amplifier kill test
    print(f"\n[7/8] Running amplifier kill test (N={runtime['n_amp_seeds']})...")
    amp_result = run_amplifier_kill_test(stage1_cfg, n_seeds=runtime["n_amp_seeds"],
                                         n_ident_samples=max(50, runtime["n_ident"] // 4))
    print(f"  Baseline ΔH0: {amp_result.baseline_Delta_H0_mean:.3f} +/- {amp_result.baseline_Delta_H0_std:.3f}")
    print(f"  Amplified ΔH0: {amp_result.amplified_Delta_H0_mean:.3f} +/- {amp_result.amplified_Delta_H0_std:.3f}")
    print(f"  Increase: {amp_result.Delta_H0_increase:+.3f} ({amp_result.Delta_H0_increase_pct:+.1f}%)")
    print(f"  Pathological fine-tuning: {'YES' if amp_result.pathological_fine_tuning else 'NO'}")

    with open(results_path / "amplifier_kill_test.json", "w") as f:
        json.dump(asdict(amp_result), f, indent=2)

    # Generate enhanced report
    print("\n[8/8] Generating SCIENCE_REPORT_v2.md...")
    generate_science_report_v2(
        results_dir=results_dir,
        cfg=stage1_cfg,
        stage1_result=stage1_result,
        train_val_split=train_val_split,
        multireadout=multireadout,
        convergence_summary=convergence_summary,
        environment_curves=env_curves,
        fingerprints=fingerprints,
        identifiability_result=ident_result,
        amplifier_result=amp_result,
    )

    print(f"\n=== Enhanced science pipeline complete ===")
    print(f"Results saved to: {results_dir}")
    print(f"Main report: {results_dir}/SCIENCE_REPORT_v2.md")


def run_science_pipeline(args, base_cfg: SimulationConfig, results_dir: str):
    """Run the full science pipeline."""
    from .analysis import (
        run_multiseed, run_convergence, run_identifiability_scan,
        aggregate_environment_curves_multiseed, aggregate_fingerprints_multiseed
    )
    from .science_report import generate_science_report

    print("=== Msoup Ejection Simulation: Science Pipeline ===")
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    # Stage 1 calibration
    print("\n[1/5] Running Stage 1 calibration (early-only)...")
    calib_cfg = CalibrationConfig(
        max_evals=max(2000, args.max_evals),
        smoke=args.smoke,
        grid_calib=128,
        steps_calib=250
    )
    stage1_result = run_stage1_calibration(base_cfg, calib_cfg, results_dir)
    print(f"  H0_early achieved: {stage1_result['H0_early_achieved']:.3f}")
    print(f"  H0_late emergent: {stage1_result['H0_late_emergent']:.3f}")
    print(f"  Deviation from 73: {stage1_result['H0_late_deviation_from_73']:+.3f}")

    # Load stage1 params for subsequent runs
    with open(results_path / "stage1_params.json", "r") as f:
        stage1_params = json.load(f)

    cfg_dict = base_cfg.to_dict()
    cfg_dict.update(stage1_params)
    cfg_dict.update({"grid": base_cfg.grid, "steps": base_cfg.steps})
    stage1_cfg = SimulationConfig.from_dict(cfg_dict)

    # Multi-seed analysis
    n_seeds = 50 if not args.smoke else 10
    print(f"\n[2/5] Running multi-seed analysis (N={n_seeds})...")
    multiseed_summary = run_multiseed(stage1_cfg, n_seeds=n_seeds)
    print(f"  H0_early: {multiseed_summary.H0_early_mean:.3f} +/- {multiseed_summary.H0_early_std:.3f}")
    print(f"  H0_late: {multiseed_summary.H0_late_mean:.3f} +/- {multiseed_summary.H0_late_std:.3f}")
    print(f"  Delta_H0: {multiseed_summary.Delta_H0_mean:.3f} +/- {multiseed_summary.Delta_H0_std:.3f}")

    # Save multiseed summary
    with open(results_path / "multiseed_summary.json", "w") as f:
        json.dump(asdict(multiseed_summary), f, indent=2)

    # Convergence analysis
    print("\n[3/5] Running convergence analysis...")
    resolutions = [(128, 250), (256, 400), (384, 600)] if not args.smoke else [(64, 100), (128, 250)]
    convergence_summary = run_convergence(stage1_cfg, resolutions=resolutions)
    print(f"  H0_late drift: {convergence_summary.H0_late_drift:.3f}")
    print(f"  Converged: {'YES' if convergence_summary.converged else 'NO'}")

    # Save convergence summary
    with open(results_path / "convergence_summary.json", "w") as f:
        json.dump(asdict(convergence_summary), f, indent=2)

    # Non-targeted predictions
    print("\n[4/5] Computing non-targeted predictions...")
    n_env_seeds = 20 if not args.smoke else 5
    env_curves = aggregate_environment_curves_multiseed(stage1_cfg, n_seeds=n_env_seeds)
    fingerprints = aggregate_fingerprints_multiseed(stage1_cfg, n_seeds=n_env_seeds)

    with open(results_path / "environment_curves.json", "w") as f:
        json.dump(env_curves, f, indent=2)
    with open(results_path / "mechanism_fingerprints.json", "w") as f:
        json.dump(fingerprints, f, indent=2)

    # Identifiability scan
    n_ident = 500 if not args.smoke else 100
    print(f"\n[5/5] Running identifiability scan (K={n_ident})...")
    ident_result = run_identifiability_scan(stage1_cfg, n_samples=n_ident)
    print(f"  Stage 1 valid: {ident_result.n_stage1_valid}/{ident_result.n_samples} ({100*ident_result.n_stage1_valid/ident_result.n_samples:.1f}%)")
    print(f"  Stage 2 valid: {ident_result.n_stage2_valid}/{ident_result.n_samples} ({100*ident_result.n_stage2_valid/ident_result.n_samples:.1f}%)")

    # Save identifiability result
    with open(results_path / "identifiability_summary.json", "w") as f:
        json.dump(asdict(ident_result), f, indent=2)

    # Generate science report
    print("\n[FINAL] Generating SCIENCE_REPORT.md...")
    generate_science_report(
        results_dir=results_dir,
        cfg=stage1_cfg,
        stage1_result=stage1_result,
        multiseed_summary=multiseed_summary,
        convergence_summary=convergence_summary,
        environment_curves=env_curves,
        fingerprints=fingerprints,
        identifiability_result=ident_result,
    )

    print(f"\n=== Science pipeline complete ===")
    print(f"Results saved to: {results_dir}")
    print(f"Main report: {results_dir}/SCIENCE_REPORT.md")


def get_runtime_settings(args):
    """Get grid/steps/seed counts based on runtime mode."""
    if args.smoke:
        return {
            "grid": 64, "steps": 80,
            "n_seeds": 5, "n_train": 3, "n_val": 5, "n_stress": 3,
            "n_ident": 50, "n_env_seeds": 3, "n_amp_seeds": 5
        }
    elif args.fast:
        return {
            "grid": 128, "steps": 200,
            "n_seeds": 15, "n_train": 5, "n_val": 15, "n_stress": 5,
            "n_ident": 300, "n_env_seeds": 8, "n_amp_seeds": 10
        }
    else:
        return {
            "grid": 256, "steps": 400,
            "n_seeds": 50, "n_train": 10, "n_val": 40, "n_stress": 10,
            "n_ident": 2000, "n_env_seeds": 20, "n_amp_seeds": 20
        }


def main():
    args = parse_args()
    results_dir = args.results_dir

    # Create timestamped output folder if requested
    if args.timestamped:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = str(Path(results_dir) / timestamp)

    base_cfg = SimulationConfig(grid=args.grid, steps=args.steps, dt=1.0 / args.steps)

    # Apply amplifier mode to base config
    base_cfg.amplifier_mode = args.amplifier

    if args.smoke:
        base_cfg.grid = min(base_cfg.grid, 64)
        base_cfg.steps = min(base_cfg.steps, 80)
        base_cfg.dt = 1.0 / base_cfg.steps
    elif args.fast:
        base_cfg.grid = min(base_cfg.grid, 128)
        base_cfg.steps = min(base_cfg.steps, 200)
        base_cfg.dt = 1.0 / base_cfg.steps

    if args.params:
        base_cfg = load_params(args.params, base_cfg)

    # Enhanced science pipeline (v2)
    if args.full_science_v2:
        run_science_pipeline_v2(args, base_cfg, results_dir)
        return

    # Full science pipeline
    if args.full_science:
        run_science_pipeline(args, base_cfg, results_dir)
        return

    # Amplifier kill test
    if args.amplifier_kill_test:
        from .analysis import run_amplifier_kill_test
        runtime = get_runtime_settings(args)
        print(f"Running amplifier kill test...")
        result = run_amplifier_kill_test(base_cfg, n_seeds=runtime["n_amp_seeds"],
                                          n_ident_samples=runtime["n_ident"] // 4)
        print(f"Baseline ΔH0: {result.baseline_Delta_H0_mean:.3f} +/- {result.baseline_Delta_H0_std:.3f}")
        print(f"Amplified ΔH0: {result.amplified_Delta_H0_mean:.3f} +/- {result.amplified_Delta_H0_std:.3f}")
        print(f"Increase: {result.Delta_H0_increase:+.3f} ({result.Delta_H0_increase_pct:+.1f}%)")
        print(f"Robust: {'YES' if result.robust_across_seeds else 'NO'}")
        print(f"Pathological fine-tuning: {'YES' if result.pathological_fine_tuning else 'NO'}")

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(results_dir) / "amplifier_kill_test.json", "w") as f:
            json.dump(asdict(result), f, indent=2)
        return

    # Stage 1 calibration
    if args.stage1_calibrate:
        calib_cfg = CalibrationConfig(max_evals=args.max_evals, smoke=args.smoke)
        result = run_stage1_calibration(base_cfg, calib_cfg, results_dir)
        print(f"Stage 1 complete:")
        print(f"  H0_early achieved: {result['H0_early_achieved']:.3f}")
        print(f"  H0_late emergent: {result['H0_late_emergent']:.3f}")
        print(f"  Deviation from 73: {result['H0_late_deviation_from_73']:+.3f}")
        return

    # Stage 2 calibration
    if args.stage2_calibrate:
        calib_cfg = CalibrationConfig(max_evals=args.max_evals, smoke=args.smoke)
        # Try to load stage1 params if available
        stage1_params = None
        stage1_path = Path(results_dir) / "stage1_params.json"
        if stage1_path.exists():
            with open(stage1_path, "r") as f:
                stage1_params = json.load(f)
        result = run_stage2_calibration(base_cfg, calib_cfg, results_dir, stage1_params)
        print(f"Stage 2 complete:")
        print(f"  H0_early achieved: {result['H0_early_achieved']:.3f}")
        print(f"  H0_late achieved: {result['H0_late_achieved']:.3f}")
        return

    # Multi-seed analysis
    if args.multiseed:
        from .analysis import run_multiseed
        print(f"Running multi-seed analysis with {args.multiseed} seeds...")
        summary = run_multiseed(base_cfg, n_seeds=args.multiseed)
        print(f"H0_early: {summary.H0_early_mean:.3f} +/- {summary.H0_early_std:.3f}")
        print(f"H0_late: {summary.H0_late_mean:.3f} +/- {summary.H0_late_std:.3f}")
        print(f"Delta_H0: {summary.Delta_H0_mean:.3f} +/- {summary.Delta_H0_std:.3f}")

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(results_dir) / "multiseed_summary.json", "w") as f:
            json.dump(asdict(summary), f, indent=2)
        return

    # Convergence analysis
    if args.convergence:
        from .analysis import run_convergence
        print("Running convergence analysis...")
        summary = run_convergence(base_cfg)
        for p in summary.points:
            pt = p if isinstance(p, dict) else asdict(p)
            print(f"  Grid {pt['grid']}: H0_late={pt['H0_late']:.3f}, Delta_H0={pt['Delta_H0']:.3f}")
        print(f"Converged: {'YES' if summary.converged else 'NO'}")

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(results_dir) / "convergence_summary.json", "w") as f:
            json.dump(asdict(summary), f, indent=2)
        return

    # Identifiability scan
    if args.identifiability_scan:
        from .analysis import run_identifiability_scan
        print(f"Running identifiability scan with {args.identifiability_scan} samples...")
        result = run_identifiability_scan(base_cfg, n_samples=args.identifiability_scan)
        print(f"Stage 1 valid: {result.n_stage1_valid}/{result.n_samples}")
        print(f"Stage 2 valid: {result.n_stage2_valid}/{result.n_samples}")

        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(results_dir) / "identifiability_summary.json", "w") as f:
            json.dump(asdict(result), f, indent=2)
        return

    # Legacy calibration
    if args.calibrate:
        calib_cfg = CalibrationConfig(max_evals=args.max_evals, smoke=args.smoke)
        best_params = run_calibration(base_cfg, calib_cfg, results_dir)
        cfg_dict = base_cfg.to_dict()
        cfg_dict.update(best_params)
        final_cfg = SimulationConfig.from_dict(cfg_dict)
        sim = run_simulation(final_cfg)
        generate_report(results_dir, final_cfg, sim, best_params, calibration_used=True)
    else:
        # Simple run
        sim = run_simulation(base_cfg)
        diagnostics = infer_expansion(base_cfg, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, base_cfg)
        numeric_params = {k: v for k, v in base_cfg.to_dict().items() if isinstance(v, (int, float))}
        params = {**numeric_params, "H0_early": diagnostics["H0_early"], "H0_late": diagnostics["H0_late"], **morph}
        generate_report(results_dir, base_cfg, sim, params, calibration_used=False)


if __name__ == "__main__":
    main()
