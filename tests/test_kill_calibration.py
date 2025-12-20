import numpy as np

from cowls_field_study.kill_calibration import (
    LensNullContext,
    build_global_null_distribution,
    compute_low_m_ratio,
    empirical_p_value,
    prepare_null_baseline,
)
from cowls_field_study.ring_profile import RingProfile


def _simple_profile(values: np.ndarray) -> RingProfile:
    theta_edges = np.linspace(0, 360, len(values) + 1)
    weights = np.ones_like(values, dtype=float)
    return RingProfile(theta_edges=theta_edges, r_theta=values, uncertainty=np.ones_like(values), coverage=np.ones_like(values), weights=weights)


def test_shift_null_variance():
    profile = _simple_profile(np.array([0.0, 1.0, 2.0, 3.0]))
    residual_samples = np.ones(10)

    ctx = LensNullContext(lens_id="lensA", band="F150W", profile=profile, residual_samples=residual_samples)
    null_vals, mean, std = prepare_null_baseline(
        profile=profile,
        residual_samples=residual_samples,
        method="shift",
        lag_max=3,
        hf_fraction=0.5,
        draws=20,
        default_draws=20,
        allow_reduction=True,
        seed=1,
    )
    ctx.null_means["shift"] = mean
    ctx.null_stds["shift"] = std

    draws = build_global_null_distribution(
        contexts=[ctx],
        method="shift",
        lag_max=3,
        hf_fraction=0.5,
        draws=50,
        default_draws=50,
        allow_reduction=True,
        seed=2,
    )
    assert np.nanvar(draws) > 0.0


def test_global_null_empirical_pvalue_monotonicity():
    profile = _simple_profile(np.array([1.0, 2.0, 3.0, 4.0]))
    residual_samples = np.ones(10)
    ctx = LensNullContext(lens_id="lensB", band="F150W", profile=profile, residual_samples=residual_samples)

    null_vals, mean, std = prepare_null_baseline(
        profile=profile,
        residual_samples=residual_samples,
        method="resample",
        lag_max=2,
        hf_fraction=0.5,
        draws=30,
        default_draws=30,
        allow_reduction=True,
        seed=3,
    )
    ctx.null_means["resample"] = mean
    ctx.null_stds["resample"] = std

    draws = build_global_null_distribution(
        contexts=[ctx],
        method="resample",
        lag_max=2,
        hf_fraction=0.5,
        draws=100,
        default_draws=100,
        allow_reduction=True,
        seed=4,
    )
    low_obs = np.nanmin(draws) - 1.0
    high_obs = np.nanmax(draws) + 1.0
    p_low = empirical_p_value(draws, low_obs)
    p_high = empirical_p_value(draws, high_obs)
    assert p_low > p_high


def test_low_m_ratio_computable_for_all_lenses_on_synthetic():
    profiles = [
        _simple_profile(np.array([1.0, 0.5, -0.5, 0.2])),
        _simple_profile(np.array([0.0, 1.0, 0.0, -1.0])),
        RingProfile(theta_edges=np.array([0, 120, 240, 360]), r_theta=np.array([np.nan, 1.0, -1.0]), uncertainty=np.ones(3), coverage=np.ones(3), weights=np.array([1.0, 0.0, 2.0])),
    ]

    ratios = [compute_low_m_ratio(p) for p in profiles]
    for ratio in ratios:
        assert np.isfinite(ratio)
        assert 0.0 <= ratio <= 1.0
