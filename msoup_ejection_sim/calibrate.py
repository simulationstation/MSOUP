"""Calibration routine for the ejection + decompression simulation."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List

from .config import SimulationConfig, CalibrationConfig, PARAM_RANGES
from .dynamics import run_simulation
from .inference import infer_expansion, morphology_stats


def _build_config(base: SimulationConfig, params: Dict, grid: int, steps: int, seed: int) -> SimulationConfig:
    cfg_dict = base.to_dict()
    cfg_dict.update(params)
    cfg_dict.update({"grid": grid, "steps": steps, "seed": seed, "dt": 1.0 / steps})
    return SimulationConfig.from_dict(cfg_dict)


def _loss(diagnostics: Dict[str, float], morph: Dict[str, float]) -> float:
    w1, w2, w3 = 1.0, 1.0, 5.0
    # Target H0_late at 73.4 to compensate for resolution effects
    loss = w1 * (diagnostics["H0_early"] - 67.0) ** 2 + w2 * (diagnostics["H0_late"] - 73.4) ** 2
    penalty = 0.0
    if not 0.02 <= morph["void_fraction"] <= 0.85:
        penalty += abs(morph["void_fraction"] - 0.2)
    if not 0.0 <= morph["high_density_fraction"] <= 0.35:
        penalty += abs(morph["high_density_fraction"] - 0.15)
    if not 0.5 <= morph["structure_amp"] <= 2.0:
        penalty += abs(morph["structure_amp"] - 1.0)
    loss += w3 * penalty
    return float(loss)


def _sample_params(rng: np.random.Generator) -> Dict[str, float]:
    return {k: rng.uniform(low, high) for k, (low, high) in PARAM_RANGES.items()}


def _refine(rng: np.random.Generator, best: Dict[str, float], scale: float = 0.15) -> Dict[str, float]:
    refined = dict(best)
    for key in ["beta", "beta_loc", "gamma0", "tau_dm3_0"]:
        low, high = PARAM_RANGES[key]
        step = (high - low) * scale
        refined[key] = np.clip(best[key] + rng.normal(scale=step * 0.5), low, high)
    return refined


def _targeted_push(rng: np.random.Generator, best: Dict[str, float],
                   diagnostics: Dict[str, float], scale: float = 0.1) -> Dict[str, float]:
    """Push parameters in directions that move H0_early/H0_late toward targets."""
    refined = dict(best)
    h0_early = diagnostics["H0_early"]
    h0_late = diagnostics["H0_late"]

    # Push beta/beta_loc higher if H0_late is too low (target 73.4 to compensate for resolution)
    if h0_late < 73.4:
        for key in ["beta", "beta_loc"]:
            low, high = PARAM_RANGES[key]
            push = (73.4 - h0_late) / 73.4 * (high - low) * 0.4
            step = rng.uniform(0, push + 0.01)
            refined[key] = np.clip(best[key] + step, low, high)

    # Push H_base to track H0_early toward 67
    if h0_early < 66.5:
        low, high = PARAM_RANGES["H_base"]
        step = (67.0 - h0_early) * 0.8
        refined["H_base"] = np.clip(best["H_base"] + step, low, high)
    elif h0_early > 67.5:
        low, high = PARAM_RANGES["H_base"]
        step = (h0_early - 67.0) * 0.8
        refined["H_base"] = np.clip(best["H_base"] - step, low, high)

    # Also perturb gamma0 and tau_dm3_0 which affect dm3 decay dynamics
    for key in ["gamma0", "tau_dm3_0", "dm3_to_A"]:
        low, high = PARAM_RANGES[key]
        step = (high - low) * scale
        refined[key] = np.clip(best[key] + rng.normal(scale=step * 0.3), low, high)

    return refined


def run_calibration(base_cfg: SimulationConfig, calib_cfg: CalibrationConfig, results_dir: str) -> Dict[str, float]:
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(base_cfg.seed)
    records: List[Dict] = []
    best_loss = np.inf
    best_params: Dict[str, float] = {}

    evals = calib_cfg.max_evals if not calib_cfg.smoke else min(300, calib_cfg.max_evals)
    grid = calib_cfg.grid_calib if not calib_cfg.smoke else max(64, calib_cfg.grid_calib // 2)
    steps = calib_cfg.steps_calib if not calib_cfg.smoke else max(80, calib_cfg.steps_calib // 2)

    for i in range(evals):
        params = _sample_params(rng)
        cfg = _build_config(base_cfg, params, grid, steps, seed=base_cfg.seed + i)
        sim = run_simulation(cfg)
        diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, cfg)
        loss = _loss(diagnostics, morph)

        record = {**params, **diagnostics, **morph, "loss": loss}
        records.append(record)

        if loss < best_loss:
            best_loss = loss
            best_params = params

    # local refinement around best
    for j in range(calib_cfg.refine_steps if not calib_cfg.smoke else 5):
        params = _refine(rng, best_params)
        cfg = _build_config(base_cfg, params, grid, steps, seed=base_cfg.seed + evals + j)
        sim = run_simulation(cfg)
        diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, cfg)
        loss = _loss(diagnostics, morph)
        record = {**params, **diagnostics, **morph, "loss": loss}
        records.append(record)
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_diag = diagnostics.copy()

    # Second phase: targeted push toward H0 targets
    best_diag = {"H0_early": 67.0, "H0_late": 72.0}
    cfg = _build_config(base_cfg, best_params, grid, steps, seed=base_cfg.seed)
    sim = run_simulation(cfg)
    best_diag = infer_expansion(cfg, sim.history, sim.final_fields)

    targeted_rounds = calib_cfg.refine_steps * 2 if not calib_cfg.smoke else 10
    for k in range(targeted_rounds):
        params = _targeted_push(rng, best_params, best_diag)
        cfg = _build_config(base_cfg, params, grid, steps, seed=base_cfg.seed + evals + calib_cfg.refine_steps + k)
        sim = run_simulation(cfg)
        diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, cfg)
        loss = _loss(diagnostics, morph)
        record = {**params, **diagnostics, **morph, "loss": loss}
        records.append(record)
        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_diag = diagnostics.copy()

    best_candidates = sorted(records, key=lambda r: r["loss"])[:calib_cfg.n_keep]
    df = pd.DataFrame(best_candidates)
    df.to_csv(f"{results_dir}/best_candidates.csv", index=False)

    with open(f"{results_dir}/best_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    full_cfg = _build_config(base_cfg, best_params, base_cfg.grid, base_cfg.steps, seed=base_cfg.seed)
    final_sim = run_simulation(full_cfg)
    diagnostics = infer_expansion(full_cfg, final_sim.history, final_sim.final_fields)
    morph = morphology_stats(final_sim.final_fields, full_cfg)
    summary = {"params": best_params, "diagnostics": diagnostics, "morphology": morph}
    with open(f"{results_dir}/calibration_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return best_params


def _stage1_loss(diagnostics: Dict[str, float], morph: Dict[str, float]) -> float:
    """Stage 1 loss: ONLY fit H0_early ≈ 67 plus morphology constraints.

    CRITICAL: H0_late is NOT included in this loss function.
    This allows us to test whether the late-time effect is emergent.
    """
    w1, w3 = 1.0, 5.0
    # ONLY H0_early - no H0_late term at all
    loss = w1 * (diagnostics["H0_early"] - 67.0) ** 2
    penalty = 0.0
    if not 0.02 <= morph["void_fraction"] <= 0.85:
        penalty += abs(morph["void_fraction"] - 0.4)
    if not 0.0 <= morph["high_density_fraction"] <= 0.35:
        penalty += abs(morph["high_density_fraction"] - 0.1)
    if not 0.5 <= morph["structure_amp"] <= 2.0:
        penalty += abs(morph["structure_amp"] - 1.0)
    loss += w3 * penalty
    return float(loss)


def _stage2_loss(diagnostics: Dict[str, float], morph: Dict[str, float]) -> float:
    """Stage 2 loss: Fit BOTH H0_early ≈ 67 and H0_late ≈ 73."""
    w1, w2, w3 = 1.0, 1.0, 5.0
    loss = w1 * (diagnostics["H0_early"] - 67.0) ** 2 + w2 * (diagnostics["H0_late"] - 73.0) ** 2
    penalty = 0.0
    if not 0.02 <= morph["void_fraction"] <= 0.85:
        penalty += abs(morph["void_fraction"] - 0.4)
    if not 0.0 <= morph["high_density_fraction"] <= 0.35:
        penalty += abs(morph["high_density_fraction"] - 0.1)
    if not 0.5 <= morph["structure_amp"] <= 2.0:
        penalty += abs(morph["structure_amp"] - 1.0)
    loss += w3 * penalty
    return float(loss)


def run_stage1_calibration(base_cfg: SimulationConfig, calib_cfg: CalibrationConfig,
                           results_dir: str) -> Dict:
    """Stage 1 calibration: Fit ONLY to H0_early ≈ 67.

    Returns both the best params AND the emergent H0_late prediction.
    This is the critical test: does H0_late naturally land near 73?
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(base_cfg.seed)
    records: List[Dict] = []
    best_loss = np.inf
    best_params: Dict[str, float] = {}

    evals = calib_cfg.max_evals if not calib_cfg.smoke else min(300, calib_cfg.max_evals)
    grid = calib_cfg.grid_calib if not calib_cfg.smoke else max(64, calib_cfg.grid_calib // 2)
    steps = calib_cfg.steps_calib if not calib_cfg.smoke else max(80, calib_cfg.steps_calib // 2)

    for i in range(evals):
        params = _sample_params(rng)
        cfg = _build_config(base_cfg, params, grid, steps, seed=base_cfg.seed + i)
        sim = run_simulation(cfg)
        diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, cfg)
        loss = _stage1_loss(diagnostics, morph)

        record = {**params, **diagnostics, **morph, "loss": loss, "stage": 1}
        records.append(record)

        if loss < best_loss:
            best_loss = loss
            best_params = params

    # Local refinement - still using stage1 loss
    for j in range(calib_cfg.refine_steps if not calib_cfg.smoke else 5):
        params = _refine(rng, best_params)
        cfg = _build_config(base_cfg, params, grid, steps, seed=base_cfg.seed + evals + j)
        sim = run_simulation(cfg)
        diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, cfg)
        loss = _stage1_loss(diagnostics, morph)
        record = {**params, **diagnostics, **morph, "loss": loss, "stage": 1}
        records.append(record)
        if loss < best_loss:
            best_loss = loss
            best_params = params

    # Save candidates
    best_candidates = sorted(records, key=lambda r: r["loss"])[:calib_cfg.n_keep]
    df = pd.DataFrame(best_candidates)
    df.to_csv(f"{results_dir}/stage1_candidates.csv", index=False)

    with open(f"{results_dir}/stage1_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # Run at full resolution to get final diagnostics
    full_cfg = _build_config(base_cfg, best_params, base_cfg.grid, base_cfg.steps, seed=base_cfg.seed)
    final_sim = run_simulation(full_cfg)
    diagnostics = infer_expansion(full_cfg, final_sim.history, final_sim.final_fields)
    morph = morphology_stats(final_sim.final_fields, full_cfg)

    # The key result: H0_late is an OUT-OF-SAMPLE prediction here
    result = {
        "stage": 1,
        "params": best_params,
        "diagnostics": diagnostics,
        "morphology": morph,
        "H0_early_target": 67.0,
        "H0_early_achieved": diagnostics["H0_early"],
        "H0_late_emergent": diagnostics["H0_late"],  # NOT targeted!
        "Delta_H0_emergent": diagnostics["Delta_H0"],
        "H0_late_deviation_from_73": diagnostics["H0_late"] - 73.0,
        "calibration_grid": grid,
        "calibration_steps": steps,
        "validation_grid": base_cfg.grid,
        "validation_steps": base_cfg.steps,
    }

    with open(f"{results_dir}/stage1_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result


def run_stage2_calibration(base_cfg: SimulationConfig, calib_cfg: CalibrationConfig,
                           results_dir: str, stage1_params: Dict[str, float] = None) -> Dict:
    """Stage 2 calibration: Fit BOTH H0_early and H0_late.

    Optionally starts from Stage 1 parameters.
    Reports how much extra tuning was needed beyond Stage 1.
    """
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(base_cfg.seed + 5000)
    records: List[Dict] = []
    best_loss = np.inf
    best_params: Dict[str, float] = stage1_params.copy() if stage1_params else {}

    evals = calib_cfg.max_evals if not calib_cfg.smoke else min(300, calib_cfg.max_evals)
    grid = calib_cfg.grid_calib if not calib_cfg.smoke else max(64, calib_cfg.grid_calib // 2)
    steps = calib_cfg.steps_calib if not calib_cfg.smoke else max(80, calib_cfg.steps_calib // 2)

    # If starting from Stage 1, first evaluate those params
    if stage1_params:
        cfg = _build_config(base_cfg, stage1_params, grid, steps, seed=base_cfg.seed)
        sim = run_simulation(cfg)
        diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, cfg)
        initial_loss = _stage2_loss(diagnostics, morph)
        best_loss = initial_loss
        records.append({**stage1_params, **diagnostics, **morph, "loss": initial_loss, "stage": 2})

    # Explore around stage1 params or from scratch
    for i in range(evals):
        if stage1_params and i < evals // 2:
            # First half: refine from stage1
            params = _refine(rng, best_params, scale=0.2)
        else:
            # Second half: broader search
            params = _sample_params(rng)

        cfg = _build_config(base_cfg, params, grid, steps, seed=base_cfg.seed + i)
        sim = run_simulation(cfg)
        diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, cfg)
        loss = _stage2_loss(diagnostics, morph)

        record = {**params, **diagnostics, **morph, "loss": loss, "stage": 2}
        records.append(record)

        if loss < best_loss:
            best_loss = loss
            best_params = params

    # Final refinement
    for j in range(calib_cfg.refine_steps if not calib_cfg.smoke else 5):
        params = _refine(rng, best_params)
        cfg = _build_config(base_cfg, params, grid, steps, seed=base_cfg.seed + evals + j)
        sim = run_simulation(cfg)
        diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, cfg)
        loss = _stage2_loss(diagnostics, morph)
        record = {**params, **diagnostics, **morph, "loss": loss, "stage": 2}
        records.append(record)
        if loss < best_loss:
            best_loss = loss
            best_params = params

    # Save candidates
    best_candidates = sorted(records, key=lambda r: r["loss"])[:calib_cfg.n_keep]
    df = pd.DataFrame(best_candidates)
    df.to_csv(f"{results_dir}/stage2_candidates.csv", index=False)

    with open(f"{results_dir}/stage2_params.json", "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2)

    # Run at full resolution
    full_cfg = _build_config(base_cfg, best_params, base_cfg.grid, base_cfg.steps, seed=base_cfg.seed)
    final_sim = run_simulation(full_cfg)
    diagnostics = infer_expansion(full_cfg, final_sim.history, final_sim.final_fields)
    morph = morphology_stats(final_sim.final_fields, full_cfg)

    # Compute how much tuning was needed if we have stage1
    param_drift = {}
    if stage1_params:
        for k in ["beta", "beta_loc", "gamma0", "tau_dm3_0", "dm3_to_A", "H_base"]:
            if k in stage1_params and k in best_params:
                param_drift[k] = best_params[k] - stage1_params[k]

    result = {
        "stage": 2,
        "params": best_params,
        "diagnostics": diagnostics,
        "morphology": morph,
        "H0_early_target": 67.0,
        "H0_late_target": 73.0,
        "H0_early_achieved": diagnostics["H0_early"],
        "H0_late_achieved": diagnostics["H0_late"],
        "Delta_H0_achieved": diagnostics["Delta_H0"],
        "param_drift_from_stage1": param_drift,
        "calibration_grid": grid,
        "calibration_steps": steps,
        "validation_grid": base_cfg.grid,
        "validation_steps": base_cfg.steps,
    }

    with open(f"{results_dir}/stage2_summary.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    return result
