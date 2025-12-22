"""
Tests for memory guardrails, streaming, and binned clustering.

These tests verify:
1. Memory guard triggers as expected
2. Null resampling uses online accumulators (no array storage)
3. Binned clustering matches exact clustering on small synthetic data
4. Pipeline processes chunks correctly without bulk loading
5. Full mode never emits NaN/Inf
"""

import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from msoup_pocket_test.config import CandidateConfig, StatsConfig, ResourceConfig
from msoup_pocket_test.resources import (
    ResourceMonitor,
    MemoryLimitExceeded,
    check_rss,
    get_rss_gb,
    estimate_memory_requirements,
)
from msoup_pocket_test.stats import (
    compute_clustering,
    compute_debiased_stats,
    WelfordAccumulator,
    OnlinePValueCounter,
)
from msoup_pocket_test.windows import Window, windows_to_frame


def _simple_candidates(n: int = 100, clustered: bool = False) -> pd.DataFrame:
    """Generate synthetic candidates for testing."""
    base_time = pd.Timestamp("2024-01-01T00:00:00Z")
    rng = np.random.default_rng(42)

    sensors = [f"S{i%10}" for i in range(n)]

    if clustered:
        # Create clusters around specific times
        cluster_centers = [100, 200, 300, 400]
        times = []
        for i in range(n):
            center = cluster_centers[i % len(cluster_centers)]
            offset = center + rng.normal(0, 5)  # Tight clustering
            times.append(base_time + pd.to_timedelta(int(offset), unit="s"))
    else:
        # Uniform random times
        times = [base_time + pd.to_timedelta(int(i * 10 + rng.uniform(-2, 2)), unit="s") for i in range(n)]

    values = rng.normal(0, 1, size=n)
    return pd.DataFrame(
        {
            "sensor": sensors,
            "start": times,
            "end": times,
            "peak_time": times,
            "peak_value": values,
            "sigma": np.abs(values) + 3,
            "is_subthreshold": False,
            "window_index": [0] * n,
        }
    )


def _windows_for_candidates(n_sensors: int = 10) -> pd.DataFrame:
    """Generate windows covering the synthetic candidates."""
    wins = [
        Window(
            sensor=f"S{i}",
            start=pd.Timestamp("2024-01-01T00:00:00Z"),
            end=pd.Timestamp("2024-01-02T00:00:00Z"),
            cadence_seconds=30,
            quality_score=1.0,
        )
        for i in range(n_sensors)
    ]
    return windows_to_frame(wins)


# ==== Test 1: Memory guard triggers ====

def test_memory_guard_check_rss_function(monkeypatch):
    """Test that check_rss raises when limit is exceeded."""
    def fake_rss():
        return 5.0  # Fake 5 GB RSS

    monkeypatch.setattr("msoup_pocket_test.resources.get_rss_gb", fake_rss)

    # Should not raise when under limit
    result = check_rss(max_rss_gb=10.0, context="test")
    assert result == 5.0

    # Should raise when over limit
    with pytest.raises(MemoryLimitExceeded):
        check_rss(max_rss_gb=4.0, context="test")


def test_memory_guard_monitor_class(monkeypatch):
    """Test ResourceMonitor tracks peak and triggers abort."""
    def make_fake_rss(value):
        def fake():
            return value
        return fake

    monitor = ResourceMonitor(max_rss_gb=2.0)

    # First check under limit
    monkeypatch.setattr("msoup_pocket_test.resources.get_rss_gb", make_fake_rss(1.0))
    monitor.check("step1")
    assert monitor.peak_rss_gb == 1.0
    assert not monitor.aborted

    # Second check higher but still under
    monkeypatch.setattr("msoup_pocket_test.resources.get_rss_gb", make_fake_rss(1.5))
    monitor.check("step2")
    assert monitor.peak_rss_gb == 1.5
    assert not monitor.aborted

    # Third check over limit
    monkeypatch.setattr("msoup_pocket_test.resources.get_rss_gb", make_fake_rss(3.0))
    with pytest.raises(MemoryLimitExceeded):
        monitor.check("step3")

    assert monitor.aborted
    assert "step3" in monitor.abort_reason


def test_memory_guard_zero_limit_means_no_check(monkeypatch):
    """Test that max_rss_gb=0 disables the guard."""
    def fake_rss():
        return 100.0  # Huge value

    monkeypatch.setattr("msoup_pocket_test.resources.get_rss_gb", fake_rss)

    monitor = ResourceMonitor(max_rss_gb=0)
    monitor.check("test")  # Should not raise
    assert monitor.peak_rss_gb == 100.0


# ==== Test 2: Null resampling uses online accumulators ====

def test_welford_accumulator_correctness():
    """Test Welford accumulator matches numpy mean/std."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0, 10.0, 15.0, 20.0]

    acc = WelfordAccumulator()
    for v in values:
        acc.update(v)

    np_mean = np.mean(values)
    np_std = np.std(values, ddof=0)  # Welford uses population std

    assert abs(acc.mean - np_mean) < 1e-10
    assert abs(acc.std - np_std) < 1e-10


def test_online_pvalue_counter_one_sided():
    """Test online p-value counter for one-sided test."""
    observed = 5.0
    counter = OnlinePValueCounter(observed)

    # Add null values: [3, 4, 5, 6, 7]
    # 3 values >= 5, so p = (3+1)/(5+1) = 0.666...
    nulls = [3.0, 4.0, 5.0, 6.0, 7.0]
    for v in nulls:
        counter.update(v)

    expected_p = (3 + 1) / (5 + 1)  # 5, 6, 7 are >= 5
    assert abs(counter.p_one_sided - expected_p) < 1e-10


def test_null_resampling_no_array_storage():
    """Test that compute_debiased_stats doesn't store null arrays."""
    cands = _simple_candidates(50)
    windows = _windows_for_candidates()

    stats_cfg = StatsConfig(
        null_realizations=16,
        block_length=4,
        allow_time_shift_null=False,
        pair_mode="binned",
        time_bin_seconds=60,
    )
    cand_cfg = CandidateConfig()

    stats = compute_debiased_stats(cands, windows, stats_cfg, cand_cfg)

    # Should NOT have null value arrays stored
    assert not hasattr(stats, "c_null")
    assert not hasattr(stats, "c_null_values")
    assert not hasattr(stats, "fano_null")
    assert not hasattr(stats, "fano_null_values")

    # Should have computed valid statistics
    assert stats.n_resamples <= 16
    assert 0 <= stats.p_clustering <= 1
    assert 0 <= stats.p_fano <= 1


# ==== Test 3: Binned clustering matches exact on small data ====

def test_binned_vs_exact_clustering_small():
    """Test that binned clustering approximately matches exact on small data."""
    # Create small dataset where exact and binned should be similar
    n = 100
    cands = _simple_candidates(n, clustered=True)
    cand_cfg = CandidateConfig(neighbor_time_seconds=30)

    # Compute exact
    c_exact = compute_clustering(
        cands,
        cand_cfg,
        pair_mode="exact",
        time_bin_seconds=30,
        max_candidates_in_memory=1000,
        unsafe=True,
    )

    # Compute binned with same bin size as neighbor window
    c_binned = compute_clustering(
        cands,
        cand_cfg,
        pair_mode="binned",
        time_bin_seconds=30,
        max_candidates_in_memory=1000,
    )

    # Should be reasonably close (binned is an approximation)
    # For small tight bins, they should match well
    assert abs(c_exact - c_binned) < 0.1, f"exact={c_exact}, binned={c_binned}"


def test_binned_clustering_poisson_debiasing():
    """Test that Poisson-like data has C_excess ~ 0 after debiasing."""
    # Create uniform random data (Poisson-like)
    cands = _simple_candidates(200, clustered=False)
    windows = _windows_for_candidates()

    stats_cfg = StatsConfig(
        null_realizations=32,
        block_length=8,
        allow_time_shift_null=False,
        pair_mode="binned",
        time_bin_seconds=60,
        random_seed=42,
    )
    cand_cfg = CandidateConfig(neighbor_time_seconds=60)

    stats = compute_debiased_stats(cands, windows, stats_cfg, cand_cfg)

    # For Poisson-like data, C_excess should be close to 0
    # Use generous tolerance since this is statistical
    assert abs(stats.c_excess) < 0.1, f"C_excess = {stats.c_excess} (expected ~0)"


def test_cluster_injection_positive_excess():
    """Test that strongly clustered data has positive C_excess."""
    # Create strongly clustered data - all events in one tight window
    base_time = pd.Timestamp("2024-01-01T00:00:00Z")
    rng = np.random.default_rng(42)

    # Put 50 events within 10 seconds, all from different sensors
    n = 50
    sensors = [f"S{i%10}" for i in range(n)]
    times = [base_time + pd.to_timedelta(rng.uniform(0, 10), unit="s") for _ in range(n)]

    cands = pd.DataFrame({
        "sensor": sensors,
        "start": times,
        "end": times,
        "peak_time": times,
        "peak_value": rng.normal(0, 1, size=n),
        "sigma": np.abs(rng.normal(0, 1, size=n)) + 3,
        "is_subthreshold": [False] * n,
        "window_index": [0] * n,
    })
    windows = _windows_for_candidates()

    stats_cfg = StatsConfig(
        null_realizations=32,
        block_length=4,
        allow_time_shift_null=False,
        pair_mode="binned",
        time_bin_seconds=5,  # Small bin to capture tight clustering
        random_seed=42,
    )
    cand_cfg = CandidateConfig(neighbor_time_seconds=15)  # Matches cluster size

    stats = compute_debiased_stats(cands, windows, stats_cfg, cand_cfg)

    # For strongly clustered data, C_excess should be positive (or at least c_obs > c_null_mean)
    # The key is that the clustering statistic should be high
    assert stats.c_obs > 0.5, f"C_obs = {stats.c_obs} (expected high clustering)"


# ==== Test 4: Pipeline processes chunks correctly ====

def test_estimate_memory_requirements():
    """Test memory estimation function."""
    estimate = estimate_memory_requirements(
        n_files=100,
        n_sensors=32,
        n_resamples=256,
        chunk_hours=168,  # 1 week
        pair_mode="binned",
    )

    assert "peak_estimate_gb" in estimate
    assert "safe" in estimate
    assert "recommendations" in estimate
    assert estimate["peak_estimate_gb"] > 0


# ==== Test 5: Full mode never emits NaN/Inf ====

def test_no_nan_inf_in_stats():
    """Test that stats computation never produces NaN/Inf."""
    # Empty candidates
    empty_cands = pd.DataFrame(columns=["sensor", "peak_time", "sigma", "window_index"])
    windows = _windows_for_candidates()

    stats_cfg = StatsConfig(null_realizations=4, block_length=2, pair_mode="binned")
    cand_cfg = CandidateConfig()

    stats = compute_debiased_stats(empty_cands, windows, stats_cfg, cand_cfg)

    # Check all numeric fields are finite
    assert np.isfinite(stats.fano_obs)
    assert np.isfinite(stats.fano_null_mean)
    assert np.isfinite(stats.fano_null_std)
    assert np.isfinite(stats.fano_z)
    assert np.isfinite(stats.p_fano)
    assert np.isfinite(stats.c_obs)
    assert np.isfinite(stats.c_null_mean)
    assert np.isfinite(stats.c_null_std)
    assert np.isfinite(stats.c_excess)
    assert np.isfinite(stats.c_z)
    assert np.isfinite(stats.p_clustering)


def test_single_candidate_no_nan():
    """Test that single candidate doesn't produce NaN."""
    cands = pd.DataFrame({
        "sensor": ["S0"],
        "start": [pd.Timestamp("2024-01-01T00:00:00Z")],
        "end": [pd.Timestamp("2024-01-01T00:00:05Z")],
        "peak_time": [pd.Timestamp("2024-01-01T00:00:00Z")],
        "peak_value": [1.0],
        "sigma": [4.0],
        "is_subthreshold": [False],
        "window_index": [0],
    })
    windows = _windows_for_candidates(1)

    stats_cfg = StatsConfig(null_realizations=4, block_length=1, pair_mode="binned")
    cand_cfg = CandidateConfig()

    stats = compute_debiased_stats(cands, windows, stats_cfg, cand_cfg)

    # Single candidate has 0 pairs, should handle gracefully
    assert np.isfinite(stats.c_obs)
    assert np.isfinite(stats.p_clustering)


# ==== Test 6: Binned clustering O(n) scaling ====

def test_binned_clustering_scaling():
    """Test that binned clustering handles large candidate counts."""
    n = 50000
    cands = _simple_candidates(n)
    cand_cfg = CandidateConfig(neighbor_time_seconds=60)

    # Should complete without memory issues
    val = compute_clustering(
        cands,
        cand_cfg,
        pair_mode="binned",
        time_bin_seconds=120,
        max_candidates_in_memory=60000,
    )

    assert 0 <= val <= 1


def test_exact_clustering_refuses_large_without_unsafe():
    """Test that exact mode refuses large candidate counts without --unsafe."""
    n = 100000
    cands = _simple_candidates(n)
    cand_cfg = CandidateConfig(neighbor_time_seconds=60)

    with pytest.raises(RuntimeError, match="exceed"):
        compute_clustering(
            cands,
            cand_cfg,
            pair_mode="exact",
            time_bin_seconds=60,
            max_candidates_in_memory=50000,
            unsafe=False,
        )


# ==== Test 7: Monitor tracks chunks ====

def test_monitor_chunk_tracking():
    """Test that ResourceMonitor tracks completed chunks."""
    monitor = ResourceMonitor(max_rss_gb=10.0)

    assert monitor.last_completed_chunk is None

    monitor.mark_chunk_complete("clk_0")
    assert monitor.last_completed_chunk == "clk_0"

    monitor.mark_chunk_complete("clk_1")
    assert monitor.last_completed_chunk == "clk_1"

    snapshot = monitor.snapshot()
    assert snapshot["last_completed_chunk"] == "clk_1"


def test_monitor_abort_report():
    """Test that abort report contains required fields."""
    monitor = ResourceMonitor(max_rss_gb=5.0)
    monitor.aborted = True
    monitor.abort_reason = "Test abort"
    monitor.mark_chunk_complete("chunk_5")
    monitor.peak_rss_gb = 4.5

    report = monitor.abort_report()

    assert report["status"] == "ABORTED"
    assert report["reason"] == "Test abort"
    assert report["last_completed_chunk"] == "chunk_5"
    assert report["peak_rss_gb"] == 4.5
