#!/usr/bin/env python3
"""Run extended power sweep with recalibration.

Uses:
- Extended epsilon values [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
- duration=10.0, dt=0.05 for adequate spectral estimation (201 samples)
- Null-calibrated thresholds
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "repo" / "src"))

from mverse_channel.config import RunConfig, SweepConfig
from mverse_channel.sim.sweeps import power_sweep
from mverse_channel.reporting.figures import plot_detection_curve

# Create output directory
out_dir = Path(__file__).parent / "extended_sweep"
out_dir.mkdir(parents=True, exist_ok=True)

# Create sweep config with extended epsilon range
sweep_config = SweepConfig(
    epsilon_values=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    tau_values=[1.5],  # Single tau for clarity
    rho_values=[0.8],  # Single rho for clarity
    replicates=15,     # More replicates for smoother curve
)

# Update base config for adequate spectral estimation
sweep_config.base.simulation.duration = 10.0
sweep_config.base.simulation.dt = 0.05
sweep_config.base.simulation.n_levels = 4
sweep_config.base.simulation.hidden_channel.enabled = True

print(f"Extended Power Sweep Configuration:")
print(f"  epsilon_values: {sweep_config.epsilon_values}")
print(f"  duration: {sweep_config.base.simulation.duration}s")
print(f"  dt: {sweep_config.base.simulation.dt}s")
print(f"  samples: ~{int(sweep_config.base.simulation.duration / sweep_config.base.simulation.dt) + 1}")
print(f"  replicates: {sweep_config.replicates}")
print()

# Run power sweep with recalibration
results = power_sweep(
    sweep_config,
    out_dir,
    alpha=0.05,
    n_calibration=50,
)

# Plot detection curve
plot_detection_curve(results, out_dir / "detection_curve.png", alpha=0.05)

# Print summary
print()
print("Results Summary:")
print("-" * 50)
for i, eps in enumerate(results["epsilon"]):
    det_prob = results["detection_prob"][i]
    mean_coh = results["mean_coherence_mag"][i]
    mean_anom = results["mean_anomaly_score"][i]
    print(f"  ε={eps:.2f}: detection={det_prob:.2f}, coh_mag={mean_coh:.4f}, anomaly={mean_anom:.4f}")

print()
print(f"Output saved to: {out_dir}")
print(f"  - power_sweep.json")
print(f"  - thresholds.json")
print(f"  - detection_curve.png")
