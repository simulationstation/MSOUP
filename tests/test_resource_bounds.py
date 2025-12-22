import pandas as pd
import numpy as np
import pytest

from msoup_pocket_test.config import CandidateConfig, StatsConfig
from msoup_pocket_test.resources import ResourceMonitor
from msoup_pocket_test.stats import compute_clustering, compute_debiased_stats
from msoup_pocket_test.windows import Window, windows_to_frame


def _simple_candidates(n: int = 100) -> pd.DataFrame:
    base_time = pd.Timestamp("2024-01-01T00:00:00Z")
    sensors = [f"S{i%10}" for i in range(n)]
    times = [base_time + pd.to_timedelta(int(i * 2), unit="s") for i in range(n)]
    values = np.random.normal(0, 1, size=n)
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


def _windows_for_candidates() -> pd.DataFrame:
    wins = [
        Window(
            sensor=f"S{i}",
            start=pd.Timestamp("2024-01-01T00:00:00Z"),
            end=pd.Timestamp("2024-01-02T00:00:00Z"),
            cadence_seconds=30,
            quality_score=1.0,
        )
        for i in range(10)
    ]
    return windows_to_frame(wins)


def test_no_store_null_arrays():
    cands = _simple_candidates(50)
    windows = _windows_for_candidates()
    stats_cfg = StatsConfig(null_realizations=8, block_length=2, allow_time_shift_null=False)
    cand_cfg = CandidateConfig()
    stats = compute_debiased_stats(cands, windows, stats_cfg, cand_cfg)
    # Ensure we didn't stash null arrays on the result object
    assert not hasattr(stats, "c_null")
    assert not hasattr(stats, "fano_null")
    assert stats.n_resamples <= stats_cfg.null_realizations


def test_binned_clustering_scaling():
    n = 50_000
    cands = _simple_candidates(n)
    cand_cfg = CandidateConfig(neighbor_time_seconds=60)
    val = compute_clustering(
        cands,
        cand_cfg,
        pair_mode="binned",
        time_bin_seconds=120,
        max_candidates_in_memory=60_000,
    )
    assert 0 <= val <= 1


def test_memory_guard_raises(monkeypatch):
    monitor = ResourceMonitor(max_rss_gb=1.0)

    def fake_rss():
        return 2.0

    monkeypatch.setattr("msoup_pocket_test.resources.get_rss_gb", fake_rss)
    with pytest.raises(RuntimeError):
        monitor.check("after_load")
