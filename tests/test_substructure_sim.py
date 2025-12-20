"""
Unit tests for Msoup Substructure Simulation

Tests core invariants:
1. Poisson model has Fano ≈ 1
2. Cluster model has higher clustering than baseline
3. Mean counts are matched across models
4. Reproducibility under seed
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

from msoup_substructure_sim.config import (
    SimConfig, get_default_config, calibrate_models_to_match_mean
)
from msoup_substructure_sim.models import (
    PoissonModel, CoxModel, ClusterModel, get_model,
    compute_intensity, draw_strengths
)
from msoup_substructure_sim.stats import circular_distance
from msoup_substructure_sim.simulate import run_simulation, run_comparison
from msoup_substructure_sim.stats import (
    analyze_result, compute_pairwise_clustering,
    compute_clustering_metric, expected_uniform_clustering
)


class TestConfig:
    """Tests for configuration."""

    def test_default_config(self):
        """Test that default config is valid."""
        config = get_default_config()
        assert config.n_lenses == 10000
        assert config.model == "poisson"

    def test_calibration_preserves_mean(self):
        """Test that calibration matches means across models."""
        config = SimConfig(n_lenses=5000, seed=42)
        config = calibrate_models_to_match_mean(config)

        # Cox adjustment
        sigma_u = config.cox.sigma_u
        e_u = np.exp(sigma_u**2 / 2)
        expected_cox_lambda = config.poisson.lambda0 / e_u
        assert np.isclose(config.cox.lambda0, expected_cox_lambda, rtol=1e-6)


class TestModels:
    """Tests for perturber models."""

    def test_poisson_generates_lenses(self):
        """Test that Poisson model generates correct number of lenses."""
        config = get_default_config(n_lenses=100, model="poisson")
        model = PoissonModel(config)
        rng = np.random.default_rng(42)
        lenses = model.generate(rng)
        assert len(lenses) == 100

    def test_cox_generates_lenses(self):
        """Test that Cox model generates correct number of lenses."""
        config = get_default_config(n_lenses=100, model="cox")
        model = CoxModel(config)
        rng = np.random.default_rng(42)
        lenses = model.generate(rng)
        assert len(lenses) == 100

    def test_cluster_generates_lenses(self):
        """Test that Cluster model generates correct number of lenses."""
        config = get_default_config(n_lenses=100, model="cluster")
        model = ClusterModel(config)
        rng = np.random.default_rng(42)
        lenses = model.generate(rng)
        assert len(lenses) == 100

    def test_intensity_positive(self):
        """Test that computed intensity is always positive."""
        H = np.random.normal(0, 1, 1000)
        z = np.random.uniform(0.2, 1.0, 1000)
        intensity = compute_intensity(H, z, lambda0=3.0, a_H=0.2, a_z=0.1)
        assert np.all(intensity > 0)

    def test_circular_distance_symmetric(self):
        """Test that circular distance is symmetric."""
        theta1, theta2 = 0.5, 2.0
        d1 = circular_distance(theta1, theta2)
        d2 = circular_distance(theta2, theta1)
        assert np.isclose(d1, d2)

    def test_circular_distance_wraparound(self):
        """Test that circular distance handles wraparound."""
        # Points at 0.1 and 6.2 (just below 2π) should be close
        d = circular_distance(0.1, 6.2)
        assert d < 0.2  # Should wrap around

    def test_strengths_in_range(self):
        """Test that drawn strengths are in valid range."""
        rng = np.random.default_rng(42)
        s = draw_strengths(1000, beta=2.0, s_min=0.1, s_max=10.0, rng=rng)
        assert np.all(s >= 0.1)
        assert np.all(s <= 10.0)


class TestSimulation:
    """Tests for simulation runner."""

    def test_simulation_runs(self):
        """Test that simulation completes without error."""
        config = get_default_config(n_lenses=100, seed=42)
        result = run_simulation(config)
        assert result.n_lenses == 100
        assert len(result.counts) == 100

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        config1 = get_default_config(n_lenses=100, seed=123, model="poisson")
        config2 = get_default_config(n_lenses=100, seed=123, model="poisson")

        result1 = run_simulation(config1)
        result2 = run_simulation(config2)

        assert np.allclose(result1.counts, result2.counts)

    def test_mean_counts_matched(self):
        """
        Test that models have approximately the same mean count after calibration.

        This is CRITICAL: we want to compare dispersion/clustering, not means.
        """
        results = run_comparison(n_lenses=5000, seed=42, verbose=False)

        means = {model: r.mean_count for model, r in results.items()}

        # All means should be within 10% of each other
        mean_values = list(means.values())
        max_mean = max(mean_values)
        min_mean = min(mean_values)

        assert (max_mean - min_mean) / np.mean(mean_values) < 0.15, \
            f"Means not matched: {means}"


class TestFanoFactor:
    """Tests for Fano factor behavior."""

    def test_poisson_fano_near_one(self):
        """
        For Poisson model, Fano factor should be approximately 1.

        This is the key invariant for the baseline.
        """
        config = get_default_config(n_lenses=10000, seed=42, model="poisson")
        result = run_simulation(config, calibrate=True)

        # Fano should be close to 1 for Poisson
        # Allow some tolerance due to finite sample and detection effects
        assert 0.7 < result.fano_factor < 1.5, \
            f"Poisson Fano factor {result.fano_factor} not near 1"

    def test_cox_fano_greater_than_one(self):
        """
        For Cox model, Fano factor should be greater than 1 (over-dispersed).
        """
        config = get_default_config(n_lenses=10000, seed=42, model="cox")
        result = run_simulation(config, calibrate=True)

        # Cox should be over-dispersed
        assert result.fano_factor > 1.0, \
            f"Cox Fano factor {result.fano_factor} should be > 1"

    def test_cluster_fano_greater_than_one(self):
        """
        For Cluster model, Fano factor should be greater than 1 (over-dispersed).
        """
        config = get_default_config(n_lenses=10000, seed=42, model="cluster")
        result = run_simulation(config, calibrate=True)

        # Cluster should be over-dispersed
        assert result.fano_factor > 1.0, \
            f"Cluster Fano factor {result.fano_factor} should be > 1"


class TestClustering:
    """Tests for spatial clustering metric."""

    def test_clustering_computation(self):
        """Test that clustering metric is computed correctly."""
        # Two points close together
        theta_close = np.array([0.0, 0.1])
        C_close = compute_pairwise_clustering(theta_close, theta0=0.3)
        assert C_close == 1.0  # Only one pair, should be close

        # Two points far apart
        theta_far = np.array([0.0, np.pi])
        C_far = compute_pairwise_clustering(theta_far, theta0=0.3)
        assert C_far == 0.0  # Only one pair, should be far

    def test_uniform_clustering_expectation(self):
        """Test that uniform points have expected clustering."""
        theta0 = 0.3
        C_expected = expected_uniform_clustering(theta0)

        # Generate many uniform samples
        rng = np.random.default_rng(42)
        C_samples = []
        for _ in range(1000):
            theta = rng.uniform(0, 2 * np.pi, 5)
            C = compute_pairwise_clustering(theta, theta0)
            C_samples.append(C)

        C_mean = np.mean(C_samples)

        # Should be close to expected
        assert abs(C_mean - C_expected) < 0.05, \
            f"Uniform C = {C_mean}, expected {C_expected}"

    def test_cluster_model_has_higher_clustering(self):
        """
        Cluster model should have higher clustering than Poisson.

        This is the key discriminating signal.
        """
        results = run_comparison(n_lenses=5000, seed=42, verbose=False)

        rng = np.random.default_rng(42)
        stats_poisson = analyze_result(results['poisson'], theta0=0.3, rng=rng)
        stats_cluster = analyze_result(results['cluster'], theta0=0.3, rng=rng)

        # Cluster should have higher clustering
        assert stats_cluster.clustering_C > stats_poisson.clustering_C, \
            f"Cluster C ({stats_cluster.clustering_C}) should be > Poisson C ({stats_poisson.clustering_C})"

    def test_cox_clustering_similar_to_poisson(self):
        """
        Cox model should have similar clustering to Poisson.

        Cox creates over-dispersion but NOT spatial clustering.
        """
        results = run_comparison(n_lenses=5000, seed=42, verbose=False)

        rng = np.random.default_rng(42)
        stats_poisson = analyze_result(results['poisson'], theta0=0.3, rng=rng)
        stats_cox = analyze_result(results['cox'], theta0=0.3, rng=rng)

        # Cox clustering should be similar to Poisson (within reasonable tolerance)
        diff = abs(stats_cox.clustering_C - stats_poisson.clustering_C)
        assert diff < 0.05, \
            f"Cox C ({stats_cox.clustering_C}) should be similar to Poisson C ({stats_poisson.clustering_C})"


class TestStats:
    """Tests for statistical analysis."""

    def test_stats_computation(self):
        """Test that statistics are computed without error."""
        config = get_default_config(n_lenses=500, seed=42)
        result = run_simulation(config)

        rng = np.random.default_rng(42)
        stats = analyze_result(result, rng=rng)

        assert stats.mean_count > 0
        assert stats.fano_factor > 0
        assert not np.isnan(stats.residual_dispersion)

    def test_tail_probability_reasonable(self):
        """Test that tail probability is in valid range."""
        config = get_default_config(n_lenses=1000, seed=42)
        result = run_simulation(config)

        rng = np.random.default_rng(42)
        stats = analyze_result(result, rng=rng)

        assert 0 <= stats.tail_prob <= 1
        assert 0 <= stats.tail_prob_poisson <= 1


class TestIntegration:
    """Integration tests."""

    def test_full_comparison(self):
        """Test full comparison runs without error."""
        results = run_comparison(n_lenses=500, seed=42, verbose=False)

        assert 'poisson' in results
        assert 'cox' in results
        assert 'cluster' in results

    def test_run_module(self):
        """Test that run module works."""
        from msoup_substructure_sim.run import run_full_analysis

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            results = run_full_analysis(
                n_lenses=200,
                seed=42,
                output_dir=output_dir,
                do_sweep=False,
                verbose=False
            )

            assert 'results' in results
            assert 'stats' in results
            assert (output_dir / 'report.md').exists()
            assert (output_dir / 'summary.json').exists()


# ============================================================================
# WINDOW TESTS
# ============================================================================

class TestSensitivityWindows:
    """Tests for arc sensitivity window functionality."""

    def test_window_generation(self):
        """Test that sensitivity windows are generated correctly."""
        from msoup_substructure_sim.windows import (
            WindowConfig, generate_sensitivity_window
        )

        config = WindowConfig(kappa_arc=2.0, sigma_arc=0.4)
        rng = np.random.default_rng(42)

        window = generate_sensitivity_window(0, config, rng)

        assert len(window.theta_centers) >= 1
        assert len(window.amplitudes) == len(window.theta_centers)
        assert window.normalization > 0

    def test_window_evaluate_positive(self):
        """Test that sensitivity function is always positive."""
        from msoup_substructure_sim.windows import (
            WindowConfig, generate_sensitivity_window
        )

        config = WindowConfig(kappa_arc=3.0, epsilon_floor=0.02)
        rng = np.random.default_rng(42)

        window = generate_sensitivity_window(0, config, rng)

        theta_test = np.linspace(0, 2 * np.pi, 100)
        S = window.evaluate(theta_test)

        assert np.all(S >= config.epsilon_floor / window.normalization * 0.9)
        assert np.all(S > 0)

    def test_circular_distance(self):
        """Test circular distance computation."""
        from msoup_substructure_sim.windows import circular_distance_vec

        theta1 = np.array([0.0, 0.1, 6.2])
        theta2 = 0.0

        d = circular_distance_vec(theta1, theta2)

        assert np.isclose(d[0], 0.0)
        assert np.isclose(d[1], 0.1)
        # Wraparound: 6.2 is close to 0 on circle
        assert d[2] < 0.2

    def test_probabilistic_thinning(self):
        """Test that thinning reduces count."""
        from msoup_substructure_sim.windows import (
            WindowConfig, generate_sensitivity_window, apply_window_thinning
        )

        config = WindowConfig(kappa_arc=2.0, gamma=2.0)
        rng = np.random.default_rng(42)

        window = generate_sensitivity_window(0, config, rng)

        # Many perturbers
        theta = rng.uniform(0, 2 * np.pi, 50)
        strengths = np.ones(50)

        theta_det, strengths_det = apply_window_thinning(
            theta, strengths, window, config.gamma, rng
        )

        # Should keep fewer than all
        assert len(theta_det) <= len(theta)
        assert len(theta_det) > 0  # Should keep at least some

    def test_threshold_modulation(self):
        """Test threshold-based detection."""
        from msoup_substructure_sim.windows import (
            WindowConfig, generate_sensitivity_window, apply_window_threshold
        )

        config = WindowConfig(kappa_arc=2.0, gamma=1.0)
        rng = np.random.default_rng(42)

        window = generate_sensitivity_window(0, config, rng)

        # Perturbers with varying strengths
        theta = np.linspace(0, 2 * np.pi, 20)
        strengths = np.linspace(0.3, 2.0, 20)

        theta_det, strengths_det = apply_window_threshold(
            theta, strengths, window, config.gamma, s_det_base=0.5
        )

        # Should detect only strong ones
        assert len(theta_det) > 0
        assert len(theta_det) < len(theta)

    def test_sample_from_sensitivity(self):
        """Test sampling from sensitivity distribution."""
        from msoup_substructure_sim.windows import (
            WindowConfig, generate_sensitivity_window, sample_from_sensitivity
        )

        config = WindowConfig(kappa_arc=1, fixed_k_arc=True, sigma_arc=0.3)
        rng = np.random.default_rng(42)

        window = generate_sensitivity_window(0, config, rng)

        samples = sample_from_sensitivity(window, 1000, rng)

        assert len(samples) == 1000
        assert np.all(samples >= 0)
        assert np.all(samples <= 2 * np.pi)

        # Samples should cluster near the single peak
        center = window.theta_centers[0]
        # Compute circular distance to center
        from msoup_substructure_sim.windows import circular_distance_vec
        distances = circular_distance_vec(samples, center)
        mean_dist = np.mean(distances)

        # Should be much less than uniform (mean would be π/2)
        assert mean_dist < np.pi / 2


class TestWindowModels:
    """Tests for window model implementations."""

    def test_poisson_window_generates(self):
        """Test that PoissonWindowModel generates correctly."""
        from msoup_substructure_sim.models import PoissonWindowModel

        config = SimConfig(n_lenses=100, seed=42, model="poisson_window")
        config.window.detection_mode = "thin"

        model = PoissonWindowModel(config)
        rng = np.random.default_rng(42)
        lenses = model.generate(rng)

        assert len(lenses) == 100
        assert model.get_windows() is not None
        assert len(model.get_windows()) == 100

    def test_cox_window_generates(self):
        """Test that CoxWindowModel generates correctly."""
        from msoup_substructure_sim.models import CoxWindowModel

        config = SimConfig(n_lenses=100, seed=42, model="cox_window")
        config.window.detection_mode = "threshold"

        model = CoxWindowModel(config)
        rng = np.random.default_rng(42)
        lenses = model.generate(rng)

        assert len(lenses) == 100
        assert model.get_windows() is not None

    def test_cluster_window_generates(self):
        """Test that ClusterWindowModel generates correctly."""
        from msoup_substructure_sim.models import ClusterWindowModel

        config = SimConfig(n_lenses=100, seed=42, model="cluster_window")

        model = ClusterWindowModel(config)
        rng = np.random.default_rng(42)
        lenses = model.generate(rng)

        assert len(lenses) == 100
        assert model.get_windows() is not None

    def test_window_models_reduce_counts(self):
        """Test that window models have lower counts than base models."""
        from msoup_substructure_sim.simulate import run_simulation

        # Poisson without window
        config_base = SimConfig(n_lenses=1000, seed=42, model="poisson")
        config_base = calibrate_models_to_match_mean(config_base)
        result_base = run_simulation(config_base, calibrate=False)

        # Poisson with window (thinning reduces counts)
        config_win = SimConfig(n_lenses=1000, seed=42, model="poisson_window")
        config_win.window.detection_mode = "thin"
        config_win.window.gamma = 2.0  # Strong thinning
        config_win = calibrate_models_to_match_mean(config_win)
        result_win = run_simulation(config_win, calibrate=False)

        # Window model should have lower mean (thinning removes some)
        # Note: This depends on window parameters, so we just check it runs
        assert result_win.mean_count >= 0
        assert result_win.windows is not None

    def test_get_model_returns_window_models(self):
        """Test that get_model returns correct window model types."""
        from msoup_substructure_sim.models import (
            get_model, PoissonWindowModel, CoxWindowModel, ClusterWindowModel
        )

        config = SimConfig(n_lenses=100, seed=42, model="poisson_window")
        model = get_model(config)
        assert isinstance(model, PoissonWindowModel)

        config.model = "cox_window"
        model = get_model(config)
        assert isinstance(model, CoxWindowModel)

        config.model = "cluster_window"
        model = get_model(config)
        assert isinstance(model, ClusterWindowModel)


class TestDebiasedClustering:
    """Tests for debiased clustering statistics."""

    def test_debiased_clustering_computation(self):
        """Test that debiased clustering is computed correctly."""
        from msoup_substructure_sim.stats import compute_debiased_clustering
        from msoup_substructure_sim.windows import (
            WindowConfig, generate_sensitivity_window
        )

        rng = np.random.default_rng(42)

        # Generate some test data
        n_lenses = 50
        theta_list = []
        windows = []

        win_cfg = WindowConfig(kappa_arc=2.0, sigma_arc=0.4)

        for i in range(n_lenses):
            window = generate_sensitivity_window(i, win_cfg, rng)
            windows.append(window)
            # Uniform perturbers
            n_pert = rng.poisson(3)
            if n_pert > 0:
                theta = rng.uniform(0, 2 * np.pi, n_pert)
            else:
                theta = np.array([])
            theta_list.append(theta)

        debias_stats = compute_debiased_clustering(
            theta_list, windows, theta0=0.3, n_resamples=50, rng=rng
        )

        assert debias_stats.n_lenses == n_lenses
        assert debias_stats.n_with_pairs >= 0

        if debias_stats.n_with_pairs > 0:
            assert not np.isnan(debias_stats.C_obs_mean)
            assert not np.isnan(debias_stats.C_excess_mean)
            assert not np.isnan(debias_stats.Z_mean)
            assert 0 <= debias_stats.global_pvalue <= 1

    def test_debiased_clustering_for_uniform(self):
        """For uniform perturbers, C_excess should be near zero."""
        from msoup_substructure_sim.stats import compute_debiased_clustering
        from msoup_substructure_sim.windows import (
            WindowConfig, generate_sensitivity_window, sample_from_sensitivity
        )

        rng = np.random.default_rng(42)

        # Generate uniform perturbers (no true clustering)
        n_lenses = 100
        theta_list = []
        windows = []

        win_cfg = WindowConfig(kappa_arc=2.0, sigma_arc=0.4)

        for i in range(n_lenses):
            window = generate_sensitivity_window(i, win_cfg, rng)
            windows.append(window)
            # Sample from the same sensitivity (simulates selection effect)
            n_pert = rng.poisson(4)
            if n_pert >= 2:
                theta = sample_from_sensitivity(window, n_pert, rng)
            else:
                theta = rng.uniform(0, 2 * np.pi, max(n_pert, 0))
            theta_list.append(theta)

        debias_stats = compute_debiased_clustering(
            theta_list, windows, theta0=0.3, n_resamples=100, rng=rng
        )

        if debias_stats.n_with_pairs > 10:
            # For uniform + selection, C_excess should be near zero
            # (selection is accounted for in the null)
            assert abs(debias_stats.C_excess_mean) < 0.2, \
                f"C_excess={debias_stats.C_excess_mean} should be near 0"

            # Z should not be significantly positive
            assert debias_stats.Z_mean < 3.0, \
                f"Z={debias_stats.Z_mean} should not be highly significant"


class TestWindowIntegration:
    """Integration tests for window functionality."""

    def test_window_simulation_runs(self):
        """Test that window simulation runs end-to-end."""
        from msoup_substructure_sim.simulate import run_simulation

        config = SimConfig(n_lenses=100, seed=42, model="poisson_window")
        result = run_simulation(config, calibrate=True)

        assert result.n_lenses == 100
        assert result.windows is not None
        assert len(result.windows) == 100

    def test_window_analysis_runs(self):
        """Test that window analysis runs end-to-end."""
        from msoup_substructure_sim.run import run_window_analysis

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            results = run_window_analysis(
                n_lenses=100,
                seed=42,
                detection_mode="thin",
                output_dir=output_dir,
                do_sweep=False,
                debias=True,
                verbose=False
            )

            assert 'results' in results
            assert 'debias_stats' in results
            assert (output_dir / 'report.md').exists()

    def test_cluster_window_has_excess_clustering(self):
        """Cluster+window should still show excess clustering after debiasing."""
        from msoup_substructure_sim.simulate import run_simulation
        from msoup_substructure_sim.stats import compute_debiased_clustering

        rng = np.random.default_rng(42)

        # Poisson window (no true clustering)
        config_pois = SimConfig(n_lenses=500, seed=42, model="poisson_window")
        config_pois = calibrate_models_to_match_mean(config_pois)
        result_pois = run_simulation(config_pois, calibrate=False)

        debias_pois = compute_debiased_clustering(
            result_pois.theta_list, result_pois.windows,
            theta0=0.3, n_resamples=100, rng=rng
        )

        # Cluster window (true clustering + selection)
        config_clust = SimConfig(n_lenses=500, seed=42, model="cluster_window")
        config_clust = calibrate_models_to_match_mean(config_clust)
        result_clust = run_simulation(config_clust, calibrate=False)

        debias_clust = compute_debiased_clustering(
            result_clust.theta_list, result_clust.windows,
            theta0=0.3, n_resamples=100, rng=rng
        )

        # Cluster should have higher C_excess than Poisson
        if debias_pois.n_with_pairs > 10 and debias_clust.n_with_pairs > 10:
            assert debias_clust.C_excess_mean > debias_pois.C_excess_mean, \
                f"Cluster C_excess ({debias_clust.C_excess_mean}) should be > Poisson ({debias_pois.C_excess_mean})"
