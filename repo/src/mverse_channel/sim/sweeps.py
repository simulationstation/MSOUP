"""Power analysis and parameter sweeps with null-calibrated thresholds.

Detection is based on calibrated thresholds from null distribution (epsilon=0),
ensuring false positive rate matches the specified alpha level.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from mverse_channel.config import SweepConfig, SimulationConfig
from mverse_channel.metrics.anomaly_score import (
    CalibratedThresholds,
    compute_null_quantile,
)
from mverse_channel.sim.generate_data import generate_metrics
from mverse_channel.rng import get_rng


def calibrate_thresholds(
    base_config: SimulationConfig,
    n_null: int = 50,
    alpha: float = 0.05,
    seed_start: int = 10000,
) -> CalibratedThresholds:
    """Calibrate detection thresholds from null distribution (epsilon=0).

    Runs n_null simulations with epsilon=0 and computes (1-alpha) quantile
    thresholds for coherence_mag and anomaly_score.

    Args:
        base_config: Base simulation configuration
        n_null: Number of null replicates to run
        alpha: Target false positive rate (default 0.05 = 5%)
        seed_start: Starting seed for null replicates

    Returns:
        CalibratedThresholds with quantile-based thresholds
    """
    coherence_mags = []
    anomaly_scores = []
    seeds_used = []

    # Create null config (epsilon=0)
    null_config = base_config.model_copy(deep=True)
    null_config.hidden_channel.enabled = True
    null_config.hidden_channel.epsilon_max = 0.0
    null_config.hidden_channel.epsilon0 = 0.0

    for i in range(n_null):
        seed = seed_start + i
        null_config.seed = seed
        seeds_used.append(seed)
        rng = get_rng(seed)
        metrics = generate_metrics(null_config, rng)
        coherence_mags.append(metrics.get("coherence_mag", 0.0))
        anomaly_scores.append(metrics.get("anomaly_score", 0.0))

    # Compute (1-alpha) quantile thresholds
    coh_threshold = compute_null_quantile(coherence_mags, alpha)
    anomaly_threshold = compute_null_quantile(anomaly_scores, alpha)

    return CalibratedThresholds(
        alpha=alpha,
        n_null=n_null,
        coherence_mag_threshold=coh_threshold,
        anomaly_score_threshold=anomaly_threshold,
        seeds_used=seeds_used,
    )


def power_sweep(
    config: SweepConfig,
    out_dir: Path,
    alpha: float = 0.05,
    n_calibration: int = 50,
    calibration_seed: int = 10000,
) -> Dict[str, List[float]]:
    """Run power sweep with null-calibrated detection thresholds.

    Detection at epsilon=0 should match alpha (e.g., 5% false positives).

    Args:
        config: Sweep configuration
        out_dir: Output directory
        alpha: Target false positive rate
        n_calibration: Number of null samples for calibration
        calibration_seed: Starting seed for calibration

    Returns:
        Results dictionary with epsilon, tau, rho, detection_prob
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Calibrate thresholds from null distribution
    print(f"Calibrating thresholds with {n_calibration} null samples (alpha={alpha})...")
    thresholds = calibrate_thresholds(
        config.base.simulation,
        n_null=n_calibration,
        alpha=alpha,
        seed_start=calibration_seed,
    )

    # Save thresholds
    threshold_data = thresholds.to_dict()
    threshold_data["calibration_date"] = datetime.now().isoformat()
    threshold_data["calibration_config"] = {
        "n_levels": config.base.simulation.n_levels,
        "duration": config.base.simulation.duration,
        "dt": config.base.simulation.dt,
    }
    (out_dir / "thresholds.json").write_text(json.dumps(threshold_data, indent=2))
    print(f"  coherence_mag threshold: {thresholds.coherence_mag_threshold:.4f}")
    print(f"  anomaly_score threshold: {thresholds.anomaly_score_threshold:.4f}")

    # Step 2: Run power sweep
    results: Dict[str, List[float]] = {
        "epsilon": [],
        "tau": [],
        "rho": [],
        "detection_prob": [],
        "mean_coherence_mag": [],
        "mean_anomaly_score": [],
    }

    for epsilon in config.epsilon_values:
        for tau in config.tau_values:
            for rho in config.rho_values:
                detections = 0
                coh_mags = []
                anomalies = []

                for rep in range(config.replicates):
                    sim_config = config.base.simulation.model_copy(deep=True)
                    sim_config.hidden_channel.enabled = True
                    sim_config.hidden_channel.epsilon_max = epsilon
                    sim_config.hidden_channel.tau_x = tau
                    sim_config.hidden_channel.rho = rho
                    sim_config.seed = config.base.simulation.seed + rep

                    rng = get_rng(sim_config.seed)
                    metrics = generate_metrics(sim_config, rng)

                    coh_mag = metrics.get("coherence_mag", 0.0)
                    anomaly = metrics.get("anomaly_score", 0.0)
                    coh_mags.append(coh_mag)
                    anomalies.append(anomaly)

                    # Detection requires BOTH metrics to exceed thresholds
                    if (coh_mag > thresholds.coherence_mag_threshold and
                        anomaly > thresholds.anomaly_score_threshold):
                        detections += 1

                results["epsilon"].append(epsilon)
                results["tau"].append(tau)
                results["rho"].append(rho)
                results["detection_prob"].append(detections / config.replicates)
                results["mean_coherence_mag"].append(float(np.mean(coh_mags)))
                results["mean_anomaly_score"].append(float(np.mean(anomalies)))

    # Save results
    results_with_meta = {
        "alpha": alpha,
        "n_calibration": n_calibration,
        "thresholds": thresholds.to_dict(),
        **results,
    }
    (out_dir / "power_sweep.json").write_text(json.dumps(results_with_meta, indent=2))

    return results


def power_sweep_legacy(config: SweepConfig, out_dir: Path) -> Dict[str, List[float]]:
    """Legacy sweep without calibration (deprecated).

    WARNING: This uses hardcoded thresholds and will have incorrect
    false positive rates. Use power_sweep() instead.
    """
    import warnings
    warnings.warn(
        "power_sweep_legacy uses hardcoded thresholds. Use power_sweep() for "
        "calibrated detection.",
        DeprecationWarning,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, List[float]] = {
        "epsilon": [],
        "tau": [],
        "rho": [],
        "detection_prob": [],
    }
    for epsilon in config.epsilon_values:
        for tau in config.tau_values:
            for rho in config.rho_values:
                detections = 0
                for rep in range(config.replicates):
                    sim_config = config.base.simulation.model_copy(deep=True)
                    sim_config.hidden_channel.enabled = True
                    sim_config.hidden_channel.epsilon_max = epsilon
                    sim_config.hidden_channel.tau_x = tau
                    sim_config.hidden_channel.rho = rho
                    sim_config.seed += rep
                    rng = get_rng(sim_config.seed)
                    metrics = generate_metrics(sim_config, rng)
                    # Legacy hardcoded thresholds (incorrect FPR)
                    if metrics.get("anomaly_score", 0.0) > 0.2 and metrics.get("coherence_peak", 0.0) > 0.1:
                        detections += 1
                results["epsilon"].append(epsilon)
                results["tau"].append(tau)
                results["rho"].append(rho)
                results["detection_prob"].append(detections / config.replicates)
    (out_dir / "power_sweep.json").write_text(json.dumps(results, indent=2))
    return results
