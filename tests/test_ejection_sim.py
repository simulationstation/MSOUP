import json
from pathlib import Path

import numpy as np

from msoup_ejection_sim.config import SimulationConfig, CalibrationConfig
from msoup_ejection_sim.dynamics import run_simulation
from msoup_ejection_sim.inference import infer_expansion
from msoup_ejection_sim.calibrate import run_calibration


def test_alignment_and_peel_bounded():
    cfg = SimulationConfig(grid=32, steps=40, dt=1.0 / 40, seed=1)
    sim = run_simulation(cfg)
    A = sim.final_fields["A"]
    wlt2 = sim.final_fields["wlt2"]
    assert np.all((A >= 0.0) & (A <= 1.0))
    assert np.all((wlt2 >= 0.0) & (wlt2 <= 1.0))


def test_dm3_decay_behavior():
    cfg_fast_decay = SimulationConfig(grid=32, steps=30, dt=1.0 / 30, tau_dm3_0=0.05)
    sim_fast = run_simulation(cfg_fast_decay)
    mean_fast = sim_fast.history["mean_dm3"][-1]

    cfg_slow_decay = SimulationConfig(grid=32, steps=30, dt=1.0 / 30, tau_dm3_0=5.0)
    sim_slow = run_simulation(cfg_slow_decay)
    mean_slow = sim_slow.history["mean_dm3"][-1]

    assert mean_fast < mean_slow
    assert mean_slow > 0.0


def test_peel_off_suppressed_when_gamma_zero():
    cfg = SimulationConfig(
        grid=32,
        steps=40,
        dt=1.0 / 40,
        gamma0=0.0,
        dm3_to_lt2=0.0,
        beta=0.08,
        beta_loc=0.05,
    )
    sim = run_simulation(cfg)
    diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
    assert sim.history["X_V"].max() < 1e-3
    assert abs(diagnostics["H0_early"] - cfg.H_base) < 1.0
    assert abs(diagnostics["H0_late"] - cfg.H_base) < 1.0


def test_beta_monotonic_delta():
    cfg_low = SimulationConfig(grid=32, steps=40, dt=1.0 / 40, beta=0.05, beta_loc=0.04)
    sim_low = run_simulation(cfg_low)
    diag_low = infer_expansion(cfg_low, sim_low.history, sim_low.final_fields)

    cfg_high = SimulationConfig(grid=32, steps=40, dt=1.0 / 40, beta=0.15, beta_loc=0.04)
    sim_high = run_simulation(cfg_high)
    diag_high = infer_expansion(cfg_high, sim_high.history, sim_high.final_fields)

    assert diag_high["Delta_H0"] > diag_low["Delta_H0"]


def test_smoke_calibration_hits_targets(tmp_path):
    base_cfg = SimulationConfig(grid=48, steps=60, dt=1.0 / 60, seed=7)
    calib_cfg = CalibrationConfig(max_evals=20, smoke=True)
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    best_params = run_calibration(base_cfg, calib_cfg, str(results_dir))

    cfg_dict = base_cfg.to_dict()
    cfg_dict.update(best_params)
    tuned_cfg = SimulationConfig.from_dict(cfg_dict)
    sim = run_simulation(tuned_cfg)
    diag = infer_expansion(tuned_cfg, sim.history, sim.final_fields)

    assert 60.0 <= diag["H0_early"] <= 74.0
    assert 65.0 <= diag["H0_late"] <= 78.0
    assert diag["H0_late"] > diag["H0_early"]

    best_params_path = Path(results_dir) / "best_params.json"
    assert best_params_path.exists()
    loaded = json.loads(best_params_path.read_text())
    assert set(best_params.keys()) == set(loaded.keys())
