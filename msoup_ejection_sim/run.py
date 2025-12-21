"""CLI for the Msoup ejection + decompression simulation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import SimulationConfig, CalibrationConfig
from .dynamics import run_simulation
from .calibrate import run_calibration
from .report import generate_report
from .inference import infer_expansion, morphology_stats


DEFAULT_RESULTS_DIR = Path("results") / "msoup_ejection_sim"


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Msoup ejection + decompression toy simulation")
    parser.add_argument("--smoke", action="store_true", help="Run a fast smoke test")
    parser.add_argument("--calibrate", action="store_true", help="Run calibration loop")
    parser.add_argument("--max-evals", type=int, default=4000, help="Max evaluations for calibration")
    parser.add_argument("--grid", type=int, default=256, help="Grid size")
    parser.add_argument("--steps", type=int, default=400, help="Number of time steps")
    parser.add_argument("--params", type=str, help="Path to parameter JSON file")
    parser.add_argument("--results-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Where to store outputs")
    return parser.parse_args()


def load_params(path: str, base_cfg: SimulationConfig) -> SimulationConfig:
    with open(path, "r", encoding="utf-8") as f:
        params = json.load(f)
    cfg_dict = base_cfg.to_dict()
    cfg_dict.update(params)
    cfg_dict.update({"grid": base_cfg.grid, "steps": base_cfg.steps})
    return SimulationConfig.from_dict(cfg_dict)


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

    if args.calibrate:
        calib_cfg = CalibrationConfig(max_evals=args.max_evals, smoke=args.smoke)
        best_params = run_calibration(base_cfg, calib_cfg, results_dir)
        cfg_dict = base_cfg.to_dict()
        cfg_dict.update(best_params)
        final_cfg = SimulationConfig.from_dict(cfg_dict)
        sim = run_simulation(final_cfg)
        generate_report(results_dir, final_cfg, sim, best_params, calibration_used=True)
    else:
        sim = run_simulation(base_cfg)
        diagnostics = infer_expansion(base_cfg, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, base_cfg)
        numeric_params = {k: v for k, v in base_cfg.to_dict().items() if isinstance(v, (int, float))}
        params = {**numeric_params, "H0_early": diagnostics["H0_early"], "H0_late": diagnostics["H0_late"], **morph}
        generate_report(results_dir, base_cfg, sim, params, calibration_used=False)


if __name__ == "__main__":
    main()
