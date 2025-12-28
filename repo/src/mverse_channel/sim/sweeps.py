"""Power analysis and parameter sweeps."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from mverse_channel.config import SweepConfig
from mverse_channel.sim.generate_data import generate_metrics
from mverse_channel.rng import get_rng


def power_sweep(config: SweepConfig, out_dir: Path) -> Dict[str, List[float]]:
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
                    sim_config = config.base.simulation.copy(deep=True)
                    sim_config.hidden_channel.enabled = True
                    sim_config.hidden_channel.epsilon_max = epsilon
                    sim_config.hidden_channel.tau_x = tau
                    sim_config.hidden_channel.rho = rho
                    sim_config.seed += rep
                    rng = get_rng(sim_config.seed)
                    metrics = generate_metrics(sim_config, rng)
                    if metrics.get("anomaly_score", 0.0) > 0.2 and metrics.get("coherence_peak", 0.0) > 0.1:
                        detections += 1
                results["epsilon"].append(epsilon)
                results["tau"].append(tau)
                results["rho"].append(rho)
                results["detection_prob"].append(detections / config.replicates)
    (out_dir / "power_sweep.json").write_text(json.dumps(results, indent=2))
    return results
