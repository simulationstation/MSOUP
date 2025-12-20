import numpy as np

from cowls_field_study.window import build_window
from cowls_field_study.ring_profile import compute_ring_profile, RingProfile
from cowls_field_study.nulls import draw_null_statistics
from cowls_field_study.stats_field import zscore
from cowls_field_study.run import aggregate_results
from cowls_field_study.report import LensResult


def test_window_normalization():
    mask = np.ones((4, 4), dtype=bool)
    data = np.ones_like(mask, dtype=float)
    noise = np.ones_like(mask, dtype=float)
    s_theta, edges = build_window(mask, data, noise, center=(1.5, 1.5), theta_bins=np.linspace(0, 360, 5))
    assert np.isclose(np.mean(s_theta), 1.0, atol=1e-6)
    assert s_theta.shape[0] == edges.size - 1


def test_ring_profile_construction():
    residual = np.arange(16, dtype=float).reshape(4, 4)
    noise = np.ones_like(residual)
    mask = np.ones_like(residual, dtype=bool)
    theta_bins = np.array([0, 180, 360])
    profile = compute_ring_profile(residual, noise, mask, center=(1.5, 1.5), theta_bins=theta_bins, window_weights=np.array([1.0, 2.0]))
    assert profile.r_theta.shape[0] == 2
    assert np.isfinite(profile.r_theta).any()
    assert np.isclose(np.mean(profile.weights), 1.0, atol=1e-6)


def test_null_generation_near_zero():
    theta_bins = np.array([0, 180, 360])
    base_profile = RingProfile(theta_edges=theta_bins, r_theta=np.zeros(2), uncertainty=np.zeros(2), coverage=np.ones(2), weights=np.ones(2))
    residual_samples = np.zeros(10)
    t_corr_null, t_pow_null = draw_null_statistics(base_profile, mode="shift", residual_samples=residual_samples, lag_max=3, hf_fraction=0.5, draws=50, seed=1)
    z_corr, _, _ = zscore(0.0, t_corr_null)
    z_pow, _, _ = zscore(0.0, t_pow_null)
    assert abs(z_corr) < 1.0
    assert abs(z_pow) < 1.0


def test_aggregate_results_math():
    results = [
        LensResult(lens_id="A", band="F115W", score_bin="M25", mode="model_residual", t_corr=0.1, t_pow=0.2, z_corr=1.0, z_pow=0.5),
        LensResult(lens_id="B", band="F150W", score_bin="S10", mode="approx_residual", t_corr=0.0, t_pow=0.1, z_corr=0.5, z_pow=0.2),
    ]
    bundle = aggregate_results(results)
    assert bundle.n_processed == 2
    assert bundle.model_count == 1
    assert bundle.approx_count == 1
    assert bundle.z_corr_global > 0.0
