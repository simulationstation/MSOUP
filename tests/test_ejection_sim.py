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


# === New Scientific Tests ===

def test_multiseed_runner_outputs_stable_keys():
    """Test that multi-seed runner produces expected output structure."""
    from msoup_ejection_sim.analysis import run_multiseed

    cfg = SimulationConfig(grid=32, steps=40, dt=1.0 / 40, seed=42)
    summary = run_multiseed(cfg, n_seeds=3, base_seed=100)

    # Check required keys
    assert hasattr(summary, "n_seeds")
    assert hasattr(summary, "H0_early_mean")
    assert hasattr(summary, "H0_early_std")
    assert hasattr(summary, "H0_late_mean")
    assert hasattr(summary, "H0_late_std")
    assert hasattr(summary, "Delta_H0_mean")
    assert hasattr(summary, "Delta_H0_std")
    assert hasattr(summary, "quantiles")
    assert hasattr(summary, "individual_results")

    assert summary.n_seeds == 3
    assert len(summary.individual_results) == 3
    assert summary.H0_early_std >= 0
    assert summary.Delta_H0_std >= 0


def test_convergence_runner_bounded_drift():
    """Test that convergence runner produces bounded metric drift."""
    from msoup_ejection_sim.analysis import run_convergence

    cfg = SimulationConfig(grid=64, steps=80, dt=1.0 / 80, seed=42)
    summary = run_convergence(cfg, resolutions=[(32, 40), (64, 80)])

    assert len(summary.points) == 2
    assert hasattr(summary, "H0_early_drift")
    assert hasattr(summary, "H0_late_drift")
    assert hasattr(summary, "Delta_H0_drift")
    assert hasattr(summary, "converged")

    # Drift should be finite
    assert np.isfinite(summary.H0_early_drift)
    assert np.isfinite(summary.H0_late_drift)
    assert np.isfinite(summary.Delta_H0_drift)


def test_stage1_calibration_does_not_reference_h0_late(tmp_path):
    """Test that Stage 1 calibration does not include H0_late in loss."""
    from msoup_ejection_sim.calibrate import run_stage1_calibration, _stage1_loss

    # Check that _stage1_loss does not penalize H0_late deviation
    diag1 = {"H0_early": 67.0, "H0_late": 70.0}
    diag2 = {"H0_early": 67.0, "H0_late": 80.0}
    morph = {"void_fraction": 0.5, "high_density_fraction": 0.1, "structure_amp": 1.0}

    loss1 = _stage1_loss(diag1, morph)
    loss2 = _stage1_loss(diag2, morph)

    # Loss should be the same regardless of H0_late
    assert loss1 == loss2, "Stage 1 loss should not depend on H0_late"


def test_stage1_calibration_runs(tmp_path):
    """Test that Stage 1 calibration runs and produces expected outputs."""
    from msoup_ejection_sim.calibrate import run_stage1_calibration

    base_cfg = SimulationConfig(grid=32, steps=40, dt=1.0 / 40, seed=7)
    calib_cfg = CalibrationConfig(max_evals=10, smoke=True)
    results_dir = str(tmp_path / "results")

    result = run_stage1_calibration(base_cfg, calib_cfg, results_dir)

    assert "H0_early_achieved" in result
    assert "H0_late_emergent" in result
    assert "Delta_H0_emergent" in result
    assert "H0_late_deviation_from_73" in result
    assert result["stage"] == 1

    # Check files created
    assert (tmp_path / "results" / "stage1_params.json").exists()
    assert (tmp_path / "results" / "stage1_summary.json").exists()


def test_environment_dependence_computes():
    """Test that environment dependence curves are computed."""
    from msoup_ejection_sim.analysis import compute_environment_dependence

    cfg = SimulationConfig(grid=32, steps=40, dt=1.0 / 40, seed=42)
    sim = run_simulation(cfg)

    curves = compute_environment_dependence(cfg, sim, n_bins=5)

    assert "density" in curves
    assert "binding" in curves
    assert "H_local_mean" in curves["density"]
    assert len(curves["density"]["H_local_mean"]) == 5


def test_mechanism_fingerprints_computes():
    """Test that mechanism fingerprints are computed."""
    from msoup_ejection_sim.analysis import compute_mechanism_fingerprints

    cfg = SimulationConfig(grid=32, steps=40, dt=1.0 / 40, seed=42)
    sim = run_simulation(cfg)

    fp = compute_mechanism_fingerprints(cfg, sim)

    assert hasattr(fp, "dm3_decay_A_increase_corr")
    assert hasattr(fp, "dm3_pockets_H_hotspots_corr")
    assert hasattr(fp, "dm3_decay_wlt2_corr")
    assert np.isfinite(fp.dm3_decay_A_increase_corr)
    assert np.isfinite(fp.dm3_pockets_H_hotspots_corr)


def test_identifiability_scan_runs():
    """Test that identifiability scan runs and produces results."""
    from msoup_ejection_sim.analysis import run_identifiability_scan

    cfg = SimulationConfig(grid=32, steps=40, dt=1.0 / 40, seed=42)
    result = run_identifiability_scan(cfg, n_samples=10, fast_grid=32, fast_steps=40)

    assert result.n_samples == 10
    assert hasattr(result, "n_stage1_valid")
    assert hasattr(result, "n_stage2_valid")
    assert hasattr(result, "corner_data")
    assert len(result.corner_data["H0_early"]) == 10


def test_science_report_generation(tmp_path):
    """Test that science report generates without errors."""
    from msoup_ejection_sim.science_report import generate_science_report
    from msoup_ejection_sim.analysis import run_multiseed, run_convergence

    cfg = SimulationConfig(grid=32, steps=40, dt=1.0 / 40, seed=42)
    results_dir = str(tmp_path / "results")

    # Minimal stage1 result
    stage1_result = {
        "stage": 1,
        "H0_early_achieved": 67.1,
        "H0_late_emergent": 71.5,
        "Delta_H0_emergent": 4.4,
        "H0_late_deviation_from_73": -1.5,
    }

    # Run minimal analyses
    multiseed = run_multiseed(cfg, n_seeds=3)
    convergence = run_convergence(cfg, resolutions=[(32, 40)])

    report = generate_science_report(
        results_dir=results_dir,
        cfg=cfg,
        stage1_result=stage1_result,
        multiseed_summary=multiseed,
        convergence_summary=convergence,
    )

    assert (tmp_path / "results" / "SCIENCE_REPORT.md").exists()
    assert "Stage 1" in report
    assert "Seed Robustness" in report
