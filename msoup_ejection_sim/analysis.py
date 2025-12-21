"""Scientific analysis runners for the ejection simulation.

This module provides:
- Multi-seed robustness analysis
- Resolution convergence testing
- Non-targeted predictions (environment dependence, mechanism fingerprints)
- Parameter identifiability scans
"""
from __future__ import annotations

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
import numpy as np

from .config import SimulationConfig, PARAM_RANGES
from .dynamics import run_simulation, SimulationResult
from .inference import infer_expansion, morphology_stats


@dataclass
class SeedRunResult:
    """Result from a single seed run."""
    seed: int
    H0_early: float
    H0_late: float
    H0_late_global: float
    Delta_H0: float
    void_fraction: float
    high_density_fraction: float
    structure_amp: float
    X_V_final: float
    mean_dm3_final: float


@dataclass
class MultiSeedSummary:
    """Summary statistics across multiple seeds."""
    n_seeds: int
    H0_early_mean: float
    H0_early_std: float
    H0_late_mean: float
    H0_late_std: float
    Delta_H0_mean: float
    Delta_H0_std: float
    H0_late_global_mean: float
    H0_late_global_std: float
    void_fraction_mean: float
    void_fraction_std: float
    structure_amp_mean: float
    structure_amp_std: float
    quantiles: Dict[str, Dict[str, float]]
    individual_results: List[Dict]


def run_multiseed(cfg: SimulationConfig, n_seeds: int = 50,
                  base_seed: int = 100) -> MultiSeedSummary:
    """Run simulation across multiple seeds and collect statistics."""
    results: List[SeedRunResult] = []

    for i in range(n_seeds):
        seed = base_seed + i
        cfg_copy = SimulationConfig.from_dict(cfg.to_dict())
        cfg_copy.seed = seed

        sim = run_simulation(cfg_copy)
        diag = infer_expansion(cfg_copy, sim.history, sim.final_fields)
        morph = morphology_stats(sim.final_fields, cfg_copy)

        result = SeedRunResult(
            seed=seed,
            H0_early=diag["H0_early"],
            H0_late=diag["H0_late"],
            H0_late_global=diag["H0_late_global"],
            Delta_H0=diag["Delta_H0"],
            void_fraction=morph["void_fraction"],
            high_density_fraction=morph["high_density_fraction"],
            structure_amp=morph["structure_amp"],
            X_V_final=float(sim.history["X_V"][-1]),
            mean_dm3_final=float(sim.history["mean_dm3"][-1]),
        )
        results.append(result)

    # Compute summary statistics
    h0_early = np.array([r.H0_early for r in results])
    h0_late = np.array([r.H0_late for r in results])
    h0_late_global = np.array([r.H0_late_global for r in results])
    delta_h0 = np.array([r.Delta_H0 for r in results])
    void_frac = np.array([r.void_fraction for r in results])
    struct_amp = np.array([r.structure_amp for r in results])

    quantile_pcts = [5, 25, 50, 75, 95]
    quantiles = {
        "H0_early": {f"q{p}": float(np.percentile(h0_early, p)) for p in quantile_pcts},
        "H0_late": {f"q{p}": float(np.percentile(h0_late, p)) for p in quantile_pcts},
        "Delta_H0": {f"q{p}": float(np.percentile(delta_h0, p)) for p in quantile_pcts},
    }

    return MultiSeedSummary(
        n_seeds=n_seeds,
        H0_early_mean=float(np.mean(h0_early)),
        H0_early_std=float(np.std(h0_early)),
        H0_late_mean=float(np.mean(h0_late)),
        H0_late_std=float(np.std(h0_late)),
        Delta_H0_mean=float(np.mean(delta_h0)),
        Delta_H0_std=float(np.std(delta_h0)),
        H0_late_global_mean=float(np.mean(h0_late_global)),
        H0_late_global_std=float(np.std(h0_late_global)),
        void_fraction_mean=float(np.mean(void_frac)),
        void_fraction_std=float(np.std(void_frac)),
        structure_amp_mean=float(np.mean(struct_amp)),
        structure_amp_std=float(np.std(struct_amp)),
        quantiles=quantiles,
        individual_results=[asdict(r) for r in results],
    )


@dataclass
class ResolutionPoint:
    """Results at a single resolution."""
    grid: int
    steps: int
    H0_early: float
    H0_late: float
    H0_late_global: float
    Delta_H0: float
    mean_X_V: float
    mean_dm3: float
    mean_A: float
    runtime_estimate: float


@dataclass
class ConvergenceSummary:
    """Summary of convergence across resolutions."""
    points: List[ResolutionPoint]
    H0_early_drift: float
    H0_late_drift: float
    Delta_H0_drift: float
    converged: bool
    convergence_threshold: float


def run_convergence(cfg: SimulationConfig,
                    resolutions: List[Tuple[int, int]] = None) -> ConvergenceSummary:
    """Run simulation at multiple resolutions to test convergence.

    Default resolutions: (128, 250), (256, 400), (384, 600)
    """
    if resolutions is None:
        resolutions = [(128, 250), (256, 400), (384, 600)]

    points: List[ResolutionPoint] = []

    for grid, steps in resolutions:
        import time
        start = time.time()

        cfg_copy = SimulationConfig.from_dict(cfg.to_dict())
        cfg_copy.grid = grid
        cfg_copy.steps = steps
        cfg_copy.dt = 1.0 / steps

        sim = run_simulation(cfg_copy)
        diag = infer_expansion(cfg_copy, sim.history, sim.final_fields)

        elapsed = time.time() - start

        point = ResolutionPoint(
            grid=grid,
            steps=steps,
            H0_early=diag["H0_early"],
            H0_late=diag["H0_late"],
            H0_late_global=diag["H0_late_global"],
            Delta_H0=diag["Delta_H0"],
            mean_X_V=float(np.mean(sim.history["X_V"])),
            mean_dm3=float(np.mean(sim.history["mean_dm3"])),
            mean_A=float(np.mean(sim.final_fields["A"])),
            runtime_estimate=elapsed,
        )
        points.append(point)

    # Compute drift between lowest and highest resolution
    if len(points) >= 2:
        h0_early_drift = abs(points[-1].H0_early - points[0].H0_early)
        h0_late_drift = abs(points[-1].H0_late - points[0].H0_late)
        delta_h0_drift = abs(points[-1].Delta_H0 - points[0].Delta_H0)
    else:
        h0_early_drift = 0.0
        h0_late_drift = 0.0
        delta_h0_drift = 0.0

    threshold = 0.5  # Convergence threshold
    converged = (h0_early_drift < threshold and
                 h0_late_drift < threshold and
                 delta_h0_drift < threshold)

    return ConvergenceSummary(
        points=[asdict(p) for p in points],
        H0_early_drift=h0_early_drift,
        H0_late_drift=h0_late_drift,
        Delta_H0_drift=delta_h0_drift,
        converged=converged,
        convergence_threshold=threshold,
    )


@dataclass
class EnvironmentCurve:
    """Binned relationship between H_local and environment."""
    bin_centers: List[float]
    H_local_mean: List[float]
    H_local_std: List[float]
    A_mean: List[float]
    wlt2_mean: List[float]
    n_per_bin: List[int]


def compute_environment_dependence(cfg: SimulationConfig,
                                   sim: SimulationResult,
                                   n_bins: int = 10) -> Dict[str, EnvironmentCurve]:
    """Compute binned relationships at final epoch.

    Returns curves for:
    - H_local vs density percentile
    - H_local vs binding proxy B percentile
    """
    final = sim.final_fields
    rho_tot = final["rho_b"] + cfg.g_dm * final["rho_dm3"]
    A = final["A"]
    wlt2 = final["wlt2"]
    B = final["B"]
    rho_mem = final["rho_mem"]

    # Compute H_local for each cell
    X_V_final = float(sim.history["X_V"][-1])
    H_local = cfg.H_base * (1.0 + cfg.beta * X_V_final) * (1.0 + cfg.beta_loc * (1.0 - A))

    curves = {}

    # Density percentile curve
    density_percentiles = np.percentile(rho_tot.flatten(), np.linspace(0, 100, n_bins + 1))
    bin_centers_density = []
    H_mean_density = []
    H_std_density = []
    A_mean_density = []
    wlt2_mean_density = []
    n_per_bin_density = []

    for i in range(n_bins):
        low, high = density_percentiles[i], density_percentiles[i + 1]
        mask = (rho_tot >= low) & (rho_tot < high) if i < n_bins - 1 else (rho_tot >= low)
        if mask.sum() > 0:
            bin_centers_density.append(float((i + 0.5) * 10))  # percentile center
            H_mean_density.append(float(np.mean(H_local[mask])))
            H_std_density.append(float(np.std(H_local[mask])))
            A_mean_density.append(float(np.mean(A[mask])))
            wlt2_mean_density.append(float(np.mean(wlt2[mask])))
            n_per_bin_density.append(int(mask.sum()))

    curves["density"] = EnvironmentCurve(
        bin_centers=bin_centers_density,
        H_local_mean=H_mean_density,
        H_local_std=H_std_density,
        A_mean=A_mean_density,
        wlt2_mean=wlt2_mean_density,
        n_per_bin=n_per_bin_density,
    )

    # Binding percentile curve
    B_percentiles = np.percentile(B.flatten(), np.linspace(0, 100, n_bins + 1))
    bin_centers_B = []
    H_mean_B = []
    H_std_B = []
    A_mean_B = []
    wlt2_mean_B = []
    n_per_bin_B = []

    for i in range(n_bins):
        low, high = B_percentiles[i], B_percentiles[i + 1]
        mask = (B >= low) & (B < high) if i < n_bins - 1 else (B >= low)
        if mask.sum() > 0:
            bin_centers_B.append(float((i + 0.5) * 10))
            H_mean_B.append(float(np.mean(H_local[mask])))
            H_std_B.append(float(np.std(H_local[mask])))
            A_mean_B.append(float(np.mean(A[mask])))
            wlt2_mean_B.append(float(np.mean(wlt2[mask])))
            n_per_bin_B.append(int(mask.sum()))

    curves["binding"] = EnvironmentCurve(
        bin_centers=bin_centers_B,
        H_local_mean=H_mean_B,
        H_local_std=H_std_B,
        A_mean=A_mean_B,
        wlt2_mean=wlt2_mean_B,
        n_per_bin=n_per_bin_B,
    )

    return {k: asdict(v) for k, v in curves.items()}


@dataclass
class MechanismFingerprint:
    """Correlation statistics for mechanism fingerprinting."""
    dm3_decay_A_increase_corr: float
    dm3_pockets_H_hotspots_corr: float
    dm3_decay_wlt2_corr: float
    n_cells: int


def compute_mechanism_fingerprints(cfg: SimulationConfig,
                                   sim: SimulationResult) -> MechanismFingerprint:
    """Compute correlations to verify mechanism predictions.

    - dm3 decay history proxy vs A increase
    - dm3 pockets vs late-time H_local hotspots
    """
    # Get initial and final fields
    t0_snap = sim.snapshots.get("t0", {})
    final = sim.final_fields

    dm3_initial = t0_snap.get("rho_dm3", np.ones_like(final["rho_dm3"]) * cfg.f_dm3_0)
    dm3_final = final["rho_dm3"]
    A_initial = t0_snap.get("A", np.ones_like(final["A"]) * cfg.A0)
    A_final = final["A"]
    wlt2_final = final["wlt2"]

    # Compute proxies
    dm3_decay = (dm3_initial - dm3_final).flatten()  # How much dm3 decayed
    A_increase = (A_final - A_initial).flatten()  # How much A increased (should be small/negative)

    # dm3 pockets: where dm3 was initially high
    dm3_pockets = dm3_initial.flatten()

    # H_local hotspots
    X_V_final = float(sim.history["X_V"][-1])
    H_local = cfg.H_base * (1.0 + cfg.beta * X_V_final) * (1.0 + cfg.beta_loc * (1.0 - A_final))
    H_hotspots = H_local.flatten()

    # Compute correlations (handle edge cases)
    def safe_corr(x, y):
        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])

    corr_dm3_A = safe_corr(dm3_decay, A_increase)
    corr_dm3_H = safe_corr(dm3_pockets, H_hotspots)
    corr_dm3_wlt2 = safe_corr(dm3_decay, wlt2_final.flatten())

    return MechanismFingerprint(
        dm3_decay_A_increase_corr=corr_dm3_A,
        dm3_pockets_H_hotspots_corr=corr_dm3_H,
        dm3_decay_wlt2_corr=corr_dm3_wlt2,
        n_cells=int(dm3_decay.size),
    )


@dataclass
class IdentifiabilityResult:
    """Result from identifiability scan."""
    n_samples: int
    n_stage1_valid: int
    n_stage2_valid: int
    stage1_tolerance: float
    stage2_tolerance: float
    param_ranges_explored: Dict[str, Tuple[float, float]]
    valid_samples: List[Dict]
    corner_data: Dict[str, List[float]]


def run_identifiability_scan(base_cfg: SimulationConfig,
                             n_samples: int = 500,
                             perturbation_scale: float = 0.2,
                             stage1_tol: float = 0.5,
                             stage2_tol: float = 0.5,
                             fast_grid: int = 128,
                             fast_steps: int = 250) -> IdentifiabilityResult:
    """Scan parameter space to assess identifiability.

    Uses Latin hypercube-like sampling around the base parameters.
    """
    key_params = ["beta", "beta_loc", "gamma0", "tau_dm3_0", "dm3_to_A", "H_base"]

    # Get base values
    base_values = {k: getattr(base_cfg, k) for k in key_params}

    # Define exploration ranges (±perturbation_scale around base)
    ranges_explored = {}
    for k in key_params:
        base_val = base_values[k]
        low, high = PARAM_RANGES.get(k, (base_val * 0.5, base_val * 1.5))
        center = base_val
        half_width = (high - low) * perturbation_scale / 2
        ranges_explored[k] = (max(low, center - half_width), min(high, center + half_width))

    rng = np.random.default_rng(base_cfg.seed + 9999)

    valid_samples = []
    corner_data = {k: [] for k in key_params}
    corner_data["H0_early"] = []
    corner_data["H0_late"] = []
    corner_data["Delta_H0"] = []
    corner_data["stage1_valid"] = []
    corner_data["stage2_valid"] = []

    for i in range(n_samples):
        # Sample parameters
        params = {}
        for k in key_params:
            low, high = ranges_explored[k]
            params[k] = rng.uniform(low, high)

        # Build config
        cfg_dict = base_cfg.to_dict()
        cfg_dict.update(params)
        cfg_dict.update({"grid": fast_grid, "steps": fast_steps, "dt": 1.0 / fast_steps})
        cfg = SimulationConfig.from_dict(cfg_dict)

        # Run simulation
        sim = run_simulation(cfg)
        diag = infer_expansion(cfg, sim.history, sim.final_fields)

        h0_early = diag["H0_early"]
        h0_late = diag["H0_late"]
        delta_h0 = diag["Delta_H0"]

        # Check validity
        stage1_valid = abs(h0_early - 67.0) < stage1_tol
        stage2_valid = stage1_valid and abs(h0_late - 73.0) < stage2_tol

        # Store corner data
        for k in key_params:
            corner_data[k].append(params[k])
        corner_data["H0_early"].append(h0_early)
        corner_data["H0_late"].append(h0_late)
        corner_data["Delta_H0"].append(delta_h0)
        corner_data["stage1_valid"].append(stage1_valid)
        corner_data["stage2_valid"].append(stage2_valid)

        if stage1_valid:
            sample_record = {
                "params": params,
                "H0_early": h0_early,
                "H0_late": h0_late,
                "Delta_H0": delta_h0,
                "stage1_valid": stage1_valid,
                "stage2_valid": stage2_valid,
            }
            valid_samples.append(sample_record)

    n_stage1 = sum(corner_data["stage1_valid"])
    n_stage2 = sum(corner_data["stage2_valid"])

    return IdentifiabilityResult(
        n_samples=n_samples,
        n_stage1_valid=n_stage1,
        n_stage2_valid=n_stage2,
        stage1_tolerance=stage1_tol,
        stage2_tolerance=stage2_tol,
        param_ranges_explored=ranges_explored,
        valid_samples=valid_samples,
        corner_data=corner_data,
    )


def aggregate_environment_curves_multiseed(cfg: SimulationConfig,
                                           n_seeds: int = 20,
                                           base_seed: int = 200) -> Dict:
    """Run environment dependence across multiple seeds and aggregate."""
    all_density_H = []
    all_binding_H = []

    for i in range(n_seeds):
        cfg_copy = SimulationConfig.from_dict(cfg.to_dict())
        cfg_copy.seed = base_seed + i

        sim = run_simulation(cfg_copy)
        curves = compute_environment_dependence(cfg_copy, sim)

        all_density_H.append(curves["density"]["H_local_mean"])
        all_binding_H.append(curves["binding"]["H_local_mean"])

    # Aggregate
    density_H = np.array(all_density_H)
    binding_H = np.array(all_binding_H)

    return {
        "density_H_mean": np.mean(density_H, axis=0).tolist(),
        "density_H_std": np.std(density_H, axis=0).tolist(),
        "binding_H_mean": np.mean(binding_H, axis=0).tolist(),
        "binding_H_std": np.std(binding_H, axis=0).tolist(),
        "n_seeds": n_seeds,
        "bin_centers": list(range(5, 100, 10)),  # 5, 15, 25, ... percentiles
    }


def aggregate_fingerprints_multiseed(cfg: SimulationConfig,
                                     n_seeds: int = 20,
                                     base_seed: int = 300) -> Dict:
    """Run mechanism fingerprints across multiple seeds and aggregate."""
    corrs_dm3_A = []
    corrs_dm3_H = []
    corrs_dm3_wlt2 = []

    for i in range(n_seeds):
        cfg_copy = SimulationConfig.from_dict(cfg.to_dict())
        cfg_copy.seed = base_seed + i

        sim = run_simulation(cfg_copy)
        fp = compute_mechanism_fingerprints(cfg_copy, sim)

        corrs_dm3_A.append(fp.dm3_decay_A_increase_corr)
        corrs_dm3_H.append(fp.dm3_pockets_H_hotspots_corr)
        corrs_dm3_wlt2.append(fp.dm3_decay_wlt2_corr)

    return {
        "dm3_decay_A_increase_corr_mean": float(np.mean(corrs_dm3_A)),
        "dm3_decay_A_increase_corr_std": float(np.std(corrs_dm3_A)),
        "dm3_pockets_H_hotspots_corr_mean": float(np.mean(corrs_dm3_H)),
        "dm3_pockets_H_hotspots_corr_std": float(np.std(corrs_dm3_H)),
        "dm3_decay_wlt2_corr_mean": float(np.mean(corrs_dm3_wlt2)),
        "dm3_decay_wlt2_corr_std": float(np.std(corrs_dm3_wlt2)),
        "n_seeds": n_seeds,
    }
