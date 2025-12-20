"""
Unit tests for Mverse Shadow Simulation

Tests core functions and invariants:
1. When biases are zero and noise is tiny, R ~ 1
2. Increasing hydrostatic bias shifts R_X upward
3. Projection scatter broadens R without shifting mean dramatically
4. M_exc is non-negative for most objects
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile

# Import simulation modules
from mverse_shadow_sim.config import SimConfig, get_default_config
from mverse_shadow_sim.truth import (
    generate_cluster_population,
    nfw_mass_enclosed,
    compute_r500_from_r200,
    clusters_to_arrays
)
from mverse_shadow_sim.observables import (
    generate_all_observables_vectorized,
    compute_sigma_crit,
    nfw_shear_profile
)
from mverse_shadow_sim.inference import run_inference_vectorized
from mverse_shadow_sim.metrics import compute_k1_metrics, compute_k2_metrics, compute_all_metrics


class TestTruth:
    """Tests for truth generator."""

    def test_cluster_population_sizes(self):
        """Test that population has correct size."""
        config = get_default_config(n_clusters=100, seed=42)
        clusters = generate_cluster_population(config)
        assert len(clusters) == 100

    def test_mass_range(self):
        """Test that masses are within specified range."""
        config = get_default_config(n_clusters=1000, seed=42)
        clusters = generate_cluster_population(config)

        masses = np.array([c.M200 for c in clusters])
        log_masses = np.log10(masses)

        assert np.all(log_masses >= config.truth.log_mass_min)
        assert np.all(log_masses <= config.truth.log_mass_max)

    def test_redshift_range(self):
        """Test that redshifts are within specified range."""
        config = get_default_config(n_clusters=1000, seed=42)
        clusters = generate_cluster_population(config)

        zs = np.array([c.z for c in clusters])

        assert np.all(zs >= config.truth.z_min)
        assert np.all(zs <= config.truth.z_max)

    def test_alpha_universal(self):
        """Test that alpha is universal (1.0) by default."""
        config = get_default_config(n_clusters=100, seed=42)
        clusters = generate_cluster_population(config)

        alphas = np.array([c.alpha_true for c in clusters])
        assert np.allclose(alphas, 1.0)

    def test_nfw_mass_enclosed_monotonic(self):
        """Test that NFW enclosed mass is monotonically increasing."""
        M200 = 1e14
        c = 5.0
        r200 = 1.0  # Mpc

        r = np.linspace(0.01, 2.0, 100)
        M_enc = nfw_mass_enclosed(r, M200, c, r200)

        assert np.all(np.diff(M_enc) > 0), "M(<r) should be monotonically increasing"

    def test_nfw_mass_at_r200(self):
        """Test that M(<r200) approximately equals M200."""
        M200 = 1e14
        c = 5.0
        r200 = 1.0

        M_at_r200 = nfw_mass_enclosed(np.array([r200]), M200, c, r200)[0]
        # Should be exactly M200 by definition
        assert np.isclose(M_at_r200, M200, rtol=1e-6)

    def test_r500_less_than_r200(self):
        """Test that r500 < r200."""
        config = get_default_config(n_clusters=100, seed=42)
        clusters = generate_cluster_population(config)

        for c in clusters:
            assert c.r500 < c.r200, f"r500 ({c.r500}) should be < r200 ({c.r200})"

    def test_excess_mass_positive(self):
        """Test that M_exc is positive (dark matter dominated)."""
        config = get_default_config(n_clusters=500, seed=42)
        clusters = generate_cluster_population(config)

        M_exc = np.array([c.M_exc_r500 for c in clusters])
        # Should be positive for all clusters (DM dominated)
        assert np.all(M_exc > 0), "Excess mass should be positive"

        # M_exc should be >> M_vis (DM dominated)
        M_vis = np.array([c.M_vis_r500 for c in clusters])
        assert np.median(M_exc / M_vis) > 5, "Excess mass should dominate visible mass"


class TestObservables:
    """Tests for observable generation."""

    def test_sigma_crit_positive(self):
        """Test that critical surface density is positive."""
        for z in [0.1, 0.3, 0.5, 0.7]:
            Sigma_crit = compute_sigma_crit(z)
            assert Sigma_crit > 0
            assert Sigma_crit > 1e10  # Reasonable order of magnitude

    def test_shear_positive(self):
        """Test that tangential shear is positive."""
        r = np.linspace(0.2, 2.0, 10)
        M200 = 3e14
        c = 4.0
        r200 = 1.5
        z = 0.3
        Sigma_crit = compute_sigma_crit(z)

        gamma = nfw_shear_profile(r, M200, c, r200, z, Sigma_crit)
        assert np.all(gamma >= 0)

    def test_observables_generation(self):
        """Test that observables can be generated."""
        config = get_default_config(n_clusters=50, seed=42, fast_mode=True)
        clusters = generate_cluster_population(config)

        rng = np.random.default_rng(42)
        lensing, dynamics = generate_all_observables_vectorized(clusters, config, rng)

        assert 'M_lens_r500' in lensing
        assert 'M_X_r500' in dynamics
        assert len(lensing['M_lens_r500']) == 50
        assert len(dynamics['M_X_r500']) == 50


class TestInference:
    """Tests for inference pipelines."""

    def test_inference_runs(self):
        """Test that inference pipeline runs without error."""
        config = get_default_config(n_clusters=50, seed=42, fast_mode=True)
        clusters = generate_cluster_population(config)

        rng = np.random.default_rng(42)
        lensing, dynamics = generate_all_observables_vectorized(clusters, config, rng)
        results = run_inference_vectorized(clusters, lensing, dynamics, config, rng)

        assert 'R_X_r500' in results
        assert 'R_V_r500' in results
        assert len(results['R_X_r500']) == 50

    def test_zero_bias_r_near_one(self):
        """
        When all biases are zero and noise is tiny, R should be close to 1.

        This is the key invariant: under universal alpha and perfect measurements,
        M_exc^lens / M_exc^dyn = 1.
        """
        config = get_default_config(n_clusters=200, seed=42, fast_mode=True)

        # Set all biases to zero
        config.dynamics.b_hse_mean = 0.0
        config.dynamics.b_hse_scatter = 0.0
        config.dynamics.b_hse_mass_slope = 0.0
        config.dynamics.b_hse_z_slope = 0.0
        config.dynamics.b_aniso_mean = 0.0
        config.dynamics.b_aniso_scatter = 0.0
        config.lensing.m_cal_mean = 0.0
        config.lensing.m_cal_scatter = 0.0
        config.truth.proj_boost_scatter = 0.0

        # Set noise to tiny
        config.dynamics.xray_noise = 0.001
        config.dynamics.vel_noise = 0.001
        config.inference.f_gas_inferred_bias = 0.0
        config.inference.f_gas_inferred_scatter = 0.001
        config.inference.f_star_inferred_scatter = 0.001

        clusters = generate_cluster_population(config)
        rng = np.random.default_rng(42)
        lensing, dynamics = generate_all_observables_vectorized(clusters, config, rng)
        results = run_inference_vectorized(clusters, lensing, dynamics, config, rng)

        R_X = results['R_X_r500']
        R_V = results['R_V_r500']

        # R should be very close to 1
        assert np.abs(np.median(R_X) - 1.0) < 0.15, f"R_X median ({np.median(R_X):.3f}) should be ~1"
        assert np.abs(np.median(R_V) - 1.0) < 0.15, f"R_V median ({np.median(R_V):.3f}) should be ~1"

        # Scatter should be smaller than with full biases
        # (Note: some scatter remains due to mass/concentration variations and
        # the approximate mass ratio used in velocity inference)
        assert np.std(R_X) < 0.5, f"R_X std ({np.std(R_X):.3f}) should be moderate"
        assert np.std(R_V) < 0.5, f"R_V std ({np.std(R_V):.3f}) should be moderate"

    def test_hydrostatic_bias_shifts_rx(self):
        """
        Increasing hydrostatic bias should shift R_X upward.

        When b_hse > 0, M_X is biased low, so M_exc^dynX is biased low,
        and R_X = M_exc^lens / M_exc^dynX should be > 1.
        """
        # Run with zero bias
        config_zero = get_default_config(n_clusters=200, seed=42, fast_mode=True)
        config_zero.dynamics.b_hse_mean = 0.0
        config_zero.dynamics.b_hse_scatter = 0.01
        config_zero.truth.proj_boost_scatter = 0.0
        config_zero.lensing.m_cal_scatter = 0.0

        clusters = generate_cluster_population(config_zero)
        rng = np.random.default_rng(42)
        lensing, dynamics = generate_all_observables_vectorized(clusters, config_zero, rng)
        results_zero = run_inference_vectorized(clusters, lensing, dynamics, config_zero, rng)

        # Run with high bias
        config_high = get_default_config(n_clusters=200, seed=42, fast_mode=True)
        config_high.dynamics.b_hse_mean = 0.3  # 30% bias
        config_high.dynamics.b_hse_scatter = 0.01
        config_high.truth.proj_boost_scatter = 0.0
        config_high.lensing.m_cal_scatter = 0.0

        clusters = generate_cluster_population(config_high)
        rng = np.random.default_rng(42)
        lensing, dynamics = generate_all_observables_vectorized(clusters, config_high, rng)
        results_high = run_inference_vectorized(clusters, lensing, dynamics, config_high, rng)

        R_X_zero = np.median(results_zero['R_X_r500'])
        R_X_high = np.median(results_high['R_X_r500'])

        # High bias should give higher R_X
        assert R_X_high > R_X_zero, f"R_X with b_hse=0.3 ({R_X_high:.2f}) should be > R_X with b_hse=0 ({R_X_zero:.2f})"

        # Expected ratio: R_X ~ 1/(1-b_hse) ~ 1.43 for b_hse=0.3
        expected_ratio = 1 / (1 - 0.3)
        assert R_X_high > 1.2, f"R_X with b_hse=0.3 should be > 1.2, got {R_X_high:.2f}"

    def test_projection_scatter_broadens_r(self):
        """
        Projection scatter should broaden R distribution but not shift mean dramatically.
        """
        # Run with no projection scatter
        config_no_proj = get_default_config(n_clusters=500, seed=42, fast_mode=True)
        config_no_proj.truth.proj_boost_scatter = 0.0
        config_no_proj.dynamics.b_hse_mean = 0.0
        config_no_proj.dynamics.b_hse_scatter = 0.0

        clusters = generate_cluster_population(config_no_proj)
        rng = np.random.default_rng(42)
        lensing, dynamics = generate_all_observables_vectorized(clusters, config_no_proj, rng)
        results_no_proj = run_inference_vectorized(clusters, lensing, dynamics, config_no_proj, rng)

        # Run with high projection scatter
        config_proj = get_default_config(n_clusters=500, seed=42, fast_mode=True)
        config_proj.truth.proj_boost_scatter = 0.2  # 0.2 dex
        config_proj.dynamics.b_hse_mean = 0.0
        config_proj.dynamics.b_hse_scatter = 0.0

        clusters = generate_cluster_population(config_proj)
        rng = np.random.default_rng(42)
        lensing, dynamics = generate_all_observables_vectorized(clusters, config_proj, rng)
        results_proj = run_inference_vectorized(clusters, lensing, dynamics, config_proj, rng)

        R_X_no_proj = results_no_proj['R_X_r500']
        R_X_proj = results_proj['R_X_r500']

        # Scatter should be larger with projection effects
        std_no_proj = np.std(R_X_no_proj)
        std_proj = np.std(R_X_proj)
        assert std_proj > std_no_proj, f"Projection scatter ({std_proj:.3f}) should increase R scatter (was {std_no_proj:.3f})"

        # Mean should not shift dramatically (projection is symmetric in log)
        mean_no_proj = np.mean(R_X_no_proj)
        mean_proj = np.mean(R_X_proj)
        assert np.abs(mean_proj - mean_no_proj) < 0.3, f"Mean shift ({np.abs(mean_proj - mean_no_proj):.3f}) should be small"


class TestMetrics:
    """Tests for metrics computation."""

    def test_k1_metrics_computation(self):
        """Test that K1 metrics can be computed."""
        R_X = np.random.normal(1.2, 0.2, 1000)
        R_V = np.random.normal(1.0, 0.15, 1000)

        k1 = compute_k1_metrics(R_X, R_V)

        assert k1.R_X_mean is not None
        assert k1.R_X_median is not None
        assert k1.K1_tolerance_X > 0
        assert len(k1.violation_fracs_X) > 0

    def test_k2_metrics_computation(self):
        """Test that K2 metrics can be computed."""
        rng = np.random.default_rng(42)
        N = 500
        R_X = np.random.normal(1.2, 0.2, N)
        R_V = np.random.normal(1.0, 0.15, N)
        log_M = np.random.normal(14.5, 0.4, N)
        z = np.random.uniform(0.1, 0.8, N)

        k2 = compute_k2_metrics(R_X, R_V, log_M, z, n_bootstrap=20, rng=rng)

        assert k2.slope_mass_X is not None
        assert k2.K2_tolerance_mass > 0
        assert k2.K2_tolerance_z > 0

    def test_excess_mass_nonnegative(self):
        """Test that M_exc is non-negative for most objects at r500."""
        config = get_default_config(n_clusters=500, seed=42, fast_mode=True)
        clusters = generate_cluster_population(config)

        rng = np.random.default_rng(42)
        lensing, dynamics = generate_all_observables_vectorized(clusters, config, rng)
        results = run_inference_vectorized(clusters, lensing, dynamics, config, rng)

        M_exc_lens = results['M_exc_lens_r500']
        M_exc_dynX = results['M_exc_dynX_r500']

        # Should be positive for vast majority
        frac_positive_lens = np.mean(M_exc_lens > 0)
        frac_positive_dynX = np.mean(M_exc_dynX > 0)

        assert frac_positive_lens > 0.99, f"Only {frac_positive_lens:.1%} have positive M_exc_lens"
        assert frac_positive_dynX > 0.99, f"Only {frac_positive_dynX:.1%} have positive M_exc_dynX"


class TestIntegration:
    """Integration tests for full pipeline."""

    def test_full_pipeline_fast_mode(self):
        """Test that full pipeline runs in fast mode."""
        config = get_default_config(n_clusters=100, seed=42, fast_mode=True)
        clusters = generate_cluster_population(config)

        rng = np.random.default_rng(42)
        lensing, dynamics = generate_all_observables_vectorized(clusters, config, rng)
        results = run_inference_vectorized(clusters, lensing, dynamics, config, rng)
        metrics = compute_all_metrics(results, "r500", rng)

        assert metrics.n_clusters == 100
        assert metrics.recommended_K1_tolerance > 0
        assert metrics.k1.R_X_median > 0

    def test_reproducibility(self):
        """Test that results are reproducible with same seed."""
        config = get_default_config(n_clusters=100, seed=123, fast_mode=True)

        # Run 1
        clusters1 = generate_cluster_population(config)
        rng1 = np.random.default_rng(123)
        lensing1, dynamics1 = generate_all_observables_vectorized(clusters1, config, rng1)
        results1 = run_inference_vectorized(clusters1, lensing1, dynamics1, config, rng1)

        # Run 2 (same seed)
        clusters2 = generate_cluster_population(config)
        rng2 = np.random.default_rng(123)
        lensing2, dynamics2 = generate_all_observables_vectorized(clusters2, config, rng2)
        results2 = run_inference_vectorized(clusters2, lensing2, dynamics2, config, rng2)

        # Should be identical
        assert np.allclose(results1['R_X_r500'], results2['R_X_r500'])
        assert np.allclose(results1['R_V_r500'], results2['R_V_r500'])

    def test_run_sim_module(self):
        """Test that run_sim module can be imported and run."""
        from mverse_shadow_sim.run_sim import run_simulation

        config = get_default_config(n_clusters=50, seed=42, fast_mode=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            results = run_simulation(config, output_dir, verbose=False)

            assert 'metrics_r500' in results
            assert (output_dir / 'report.md').exists()
            assert (output_dir / 'summary.json').exists()


class TestPhysicalConsistency:
    """Tests for physical consistency of results."""

    def test_r_distribution_reasonable(self):
        """Test that R distribution has reasonable properties."""
        config = get_default_config(n_clusters=1000, seed=42, fast_mode=True)
        clusters = generate_cluster_population(config)

        rng = np.random.default_rng(42)
        lensing, dynamics = generate_all_observables_vectorized(clusters, config, rng)
        results = run_inference_vectorized(clusters, lensing, dynamics, config, rng)

        R_X = results['R_X_r500']
        R_V = results['R_V_r500']

        # R should be positive
        assert np.all(R_X > 0)
        assert np.all(R_V > 0)

        # R should typically be between 0.5 and 3 (not extreme)
        assert np.percentile(R_X, 5) > 0.3
        assert np.percentile(R_X, 95) < 5.0
        assert np.percentile(R_V, 5) > 0.3
        assert np.percentile(R_V, 95) < 5.0

        # Median R_X should be > 1 due to hydrostatic bias
        assert np.median(R_X) > 1.0, "R_X should be > 1 due to hydrostatic bias"

    def test_mass_hierarchy(self):
        """Test that mass hierarchy is correct: M_vis < M_tot."""
        config = get_default_config(n_clusters=100, seed=42)
        clusters = generate_cluster_population(config)

        for c in clusters:
            assert c.M_vis_r500 < c.M_tot_r500
            assert c.M_vis_half_r500 < c.M_tot_half_r500
            assert c.M_exc_r500 > 0
            assert c.M_exc_half_r500 > 0
