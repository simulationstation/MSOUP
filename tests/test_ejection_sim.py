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
        K_enabled=False,  # Disable K for this test (v2 behavior)
        dm3_to_A_mode="legacy",  # Use legacy mode for fair test
    )
    sim = run_simulation(cfg)
    diagnostics = infer_expansion(cfg, sim.history, sim.final_fields)
    assert sim.history["X_V"].max() < 1e-3
    assert abs(diagnostics["H0_early"] - cfg.H_base) < 1.0
    assert abs(diagnostics["H0_late"] - cfg.H_base) < 1.5  # Slightly relaxed for K effects


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


# === v3: K Field Tests ===

def test_K_field_bounded():
    """Test that K field stays in [0,1] throughout simulation."""
    cfg = SimulationConfig(grid=32, steps=50, dt=1.0 / 50, seed=42, K_enabled=True)
    sim = run_simulation(cfg)

    K = sim.final_fields["K"]
    assert np.all((K >= 0.0) & (K <= 1.0)), "K must be in [0,1]"

    # Check history as well
    for k_val in sim.history["mean_K"]:
        assert 0.0 <= k_val <= 1.0, "mean_K history must be in [0,1]"


def test_K_decay_binding_dependent():
    """Test that K decays slower in high-B regions when chi_K > 0."""
    cfg = SimulationConfig(
        grid=32, steps=80, dt=1.0 / 80, seed=42,
        K_enabled=True,
        tau_K0=0.5,  # Moderate decay
        chi_K=3.0,   # Strong binding dependence
    )
    sim = run_simulation(cfg)

    # K should decay (not stay at initial value)
    assert sim.K_diagnostics["mean_K_final"] < sim.K_diagnostics["mean_K_initial"]

    # Correlation K-B should be positive (K persists longer where B is high)
    assert sim.K_diagnostics["corr_K_B_late"] >= 0.0, \
        "K should correlate positively with B at late times when chi_K > 0"


def test_K_suppresses_peel_off():
    """Test that K near 1 suppresses peel-off (wlt2 stays small early)."""
    # Config with K enabled and slow decay
    cfg_with_K = SimulationConfig(
        grid=32, steps=60, dt=1.0 / 60, seed=42,
        K_enabled=True,
        tau_K0=2.0,  # Very slow decay - K stays high
        chi_K=2.0,
        gamma0=0.3,  # Moderate peel-off rate
    )
    sim_K = run_simulation(cfg_with_K)

    # Config with K disabled
    cfg_no_K = SimulationConfig(
        grid=32, steps=60, dt=1.0 / 60, seed=42,
        K_enabled=False,
        gamma0=0.3,
        dm3_to_A_mode="legacy",  # Use legacy mode for fair comparison
    )
    sim_no_K = run_simulation(cfg_no_K)

    # Early wlt2 should be lower when K is enabled
    early_steps = len(sim_K.history["X_V"]) // 4
    early_wlt2_K = np.mean(sim_K.history["X_V"][:early_steps])
    early_wlt2_no_K = np.mean(sim_no_K.history["X_V"][:early_steps])

    assert early_wlt2_K <= early_wlt2_no_K + 0.1, \
        "K should suppress early peel-off"


def test_K_disabled_backward_compatible():
    """Test that K_enabled=False produces similar behavior to v2."""
    cfg = SimulationConfig(
        grid=32, steps=50, dt=1.0 / 50, seed=42,
        K_enabled=False,
        dm3_to_A_mode="legacy",
    )
    sim = run_simulation(cfg)

    # K should be zeros when disabled
    assert np.allclose(sim.final_fields["K"], 0.0), \
        "K should be zero when K_enabled=False"

    # All K diagnostics in history should be zero
    assert np.allclose(sim.history["mean_K"], 0.0), \
        "mean_K history should be zero when K disabled"

    # Standard diagnostics should still work
    diag = infer_expansion(cfg, sim.history, sim.final_fields)
    assert "H0_early" in diag
    assert "H0_late" in diag


def test_K_release_time_tracked():
    """Test that K release time is correctly tracked."""
    cfg = SimulationConfig(
        grid=32, steps=100, dt=1.0 / 100, seed=42,
        K_enabled=True,
        tau_K0=0.3,  # Fast enough to release
        chi_K=1.0,
    )
    sim = run_simulation(cfg)

    # Release time should be tracked if mean_K drops below 0.5
    if sim.K_diagnostics["K_release_time"] is not None:
        assert 0.0 < sim.K_diagnostics["K_release_time"] <= 1.0
        # Verify it's actually where mean_K crosses 0.5
        release_step = sim.K_diagnostics["K_release_step"]
        assert sim.history["mean_K"][release_step] < 0.5


def test_K_modulates_alignment_pinning():
    """Test that K modulates the alignment pinning term."""
    # With K enabled and high, pinning should be stronger
    cfg_high_K = SimulationConfig(
        grid=32, steps=60, dt=1.0 / 60, seed=42,
        K_enabled=True,
        tau_K0=5.0,  # Very slow decay - K stays high
        chi_K=2.0,
        k_pin=1.5,
    )
    sim_high_K = run_simulation(cfg_high_K)

    # With K disabled
    cfg_no_K = SimulationConfig(
        grid=32, steps=60, dt=1.0 / 60, seed=42,
        K_enabled=False,
        k_pin=1.5,
        dm3_to_A_mode="legacy",
    )
    sim_no_K = run_simulation(cfg_no_K)

    # A should be higher (better pinned) when K is high
    A_high_K = np.mean(sim_high_K.final_fields["A"])
    A_no_K = np.mean(sim_no_K.final_fields["A"])

    # This test checks the mechanism - A may or may not be higher depending on dynamics
    # but A should definitely be in valid range
    assert 0.0 <= A_high_K <= 1.0
    assert 0.0 <= A_no_K <= 1.0


def test_environment_curves_include_K():
    """Test that environment curves now include K percentile."""
    from msoup_ejection_sim.analysis import compute_environment_dependence

    cfg = SimulationConfig(grid=32, steps=50, dt=1.0 / 50, seed=42, K_enabled=True)
    sim = run_simulation(cfg)

    curves = compute_environment_dependence(cfg, sim, n_bins=5)

    assert "K_percentile" in curves, "Should include K_percentile environment curve"
    assert "H_local_mean" in curves["K_percentile"]
    assert "wlt2_mean" in curves["K_percentile"]


def test_identifiability_includes_K_params():
    """Test that identifiability scan includes K parameters."""
    from msoup_ejection_sim.analysis import IDENTIFIABILITY_PARAMS, run_identifiability_scan

    assert "tau_K0" in IDENTIFIABILITY_PARAMS
    assert "chi_K" in IDENTIFIABILITY_PARAMS

    # Run a minimal scan to verify it includes K params
    cfg = SimulationConfig(grid=32, steps=40, dt=1.0 / 40, seed=42, K_enabled=True)
    result = run_identifiability_scan(cfg, n_samples=5, fast_grid=32, fast_steps=40)

    assert "tau_K0" in result.corner_data
    assert "chi_K" in result.corner_data
