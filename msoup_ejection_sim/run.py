"""CLI for the Msoup ejection + decompression simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import asdict

from .config import SimulationConfig, CalibrationConfig
from .dynamics import run_simulation
from .calibrate import run_calibration, run_stage1_calibration, run_stage2_calibration
from .report import generate_report
from .inference import infer_expansion, morphology_stats


DEFAULT_RESULTS_DIR = Path("results") / "msoup_ejection_sim"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Msoup ejection + decompression simulation")
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke test")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration loop (legacy)")
    parser.add_argument("--max-evals", type=int, default=4000, help="Max evaluations for calibration")
    parser.add_argument("--grid", type=int, default=256, help="Grid size")
    parser.add_argument("--steps", type=int, default=400, help="Number of time steps")
    parser.add_argument("--params", type=str, help="Path to parameter JSON file")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Where to store outputs")

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

    return parser.parse_args()


def load_params(path: str, base_cfg: SimulationConfig) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    cfg_dict = base_cfg.to_dict()
    cfg_dict.update(params)
    cfg_dict.update({"grid": base_cfg.grid, "steps": base_cfg.steps})
    return SimulationConfig.from_dict(cfg_dict)


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


def main():
    args = parse_args()
    results_dir = args.results_dir

    base_cfg = SimulationConfig(grid=args.grid, steps=args.steps, dt=1.0 / args.steps)

    if args.smoke:
        base_cfg.grid = min(base_cfg.grid, 64)
        base_cfg.steps = min(base_cfg.steps, 80)
        base_cfg.dt = 1.0 / base_cfg.steps

    if args.params:
        base_cfg = load_params(args.params, base_cfg)

    # Full science pipeline
    if args.full_science:
        run_science_pipeline(args, base_cfg, results_dir)
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
