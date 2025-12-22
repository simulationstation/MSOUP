import pandas as pd
import numpy as np

from msoup_pocket_test.candidates import Candidate, candidates_to_frame
from msoup_pocket_test.nulls import can_time_shift, generate_null_catalogs
from msoup_pocket_test.stats import compute_debiased_stats
from msoup_pocket_test.windows import Window, windows_to_frame
from msoup_pocket_test.config import CandidateConfig, StatsConfig


def _synthetic_candidates(cluster: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    sensors = ["S1", "S2", "S3", "S4"]
    rows = []
    base_time = pd.Timestamp("2024-01-01T00:00:00Z")
    for sensor in sensors:
        times = []
        if cluster:
            cluster_time = base_time + pd.to_timedelta(100, unit="s")
            times.extend([cluster_time + pd.to_timedelta(delta, unit="s") for delta in rng.normal(0, 3, 12)])
        # background Poisson-like
        background_count = 12 if cluster else 20
        times.extend([base_time + pd.to_timedelta(int(t), unit="s") for t in rng.uniform(0, 500, background_count)])
        for t in times:
            rows.append(
                Candidate(
                    sensor=sensor,
                    start=t,
                    end=t + pd.to_timedelta(5, unit="s"),
                    peak_time=t,
                    peak_value=rng.normal(0, 1),
                    sigma=rng.normal(3, 0.5),
                    is_subthreshold=False,
                    window_index=0,
                )
            )
    return candidates_to_frame(rows)


def _windows() -> pd.DataFrame:
    wins = [
        Window(sensor=s, start=pd.Timestamp("2024-01-01T00:00:00Z"), end=pd.Timestamp("2024-01-01T01:00:00Z"), cadence_seconds=30, quality_score=1.0)
        for s in ["S1", "S2", "S3", "S4"]
    ]
    return windows_to_frame(wins)


def test_window_debias_poisson_like():
    candidates = _synthetic_candidates(cluster=False)
    stats_cfg = StatsConfig(null_realizations=32, block_length=4, random_seed=2, allow_time_shift_null=False)
    cand_cfg = CandidateConfig()
    stats = compute_debiased_stats(candidates, _windows(), stats_cfg, cand_cfg)
    assert abs(stats.c_excess) < 0.06


def test_cluster_injection_positive_c_excess():
    candidates = _synthetic_candidates(cluster=True)
    stats_cfg = StatsConfig(null_realizations=32, block_length=4, random_seed=3, allow_time_shift_null=False)
    cand_cfg = CandidateConfig(neighbor_time_seconds=30)
    stats = compute_debiased_stats(candidates, _windows(), stats_cfg, cand_cfg)
    assert stats.c_excess > 0.02


def test_null_construction_respects_windows():
    candidates = _synthetic_candidates(cluster=False)
    windows = _windows()
    nulls = generate_null_catalogs(candidates, windows, n_realizations=4, block_length=2, allow_time_shift=False, seed=5)
    for cat in nulls:
        assert not cat.empty
        assert (cat["peak_time"] >= windows["start"].min()).all()
        assert (cat["peak_time"] <= windows["end"].max()).all()


def test_time_shift_allowed_only_with_full_coverage():
    windows = _windows()
    assert can_time_shift(windows)
    bad_windows = windows.copy()
    bad_windows.loc[0, "start"] = pd.Timestamp("2024-01-01T00:00:00Z")
    bad_windows.loc[0, "end"] = pd.Timestamp("2024-01-01T00:05:00Z")
    assert not can_time_shift(bad_windows)


def test_sensor_weights_prevent_dominance():
    candidates = _synthetic_candidates(cluster=False)
    # duplicate one sensor to inflate counts
    extra = candidates[candidates["sensor"] == "S1"]
    extra = extra.assign(sensor="S1")
    inflated = pd.concat([candidates, extra], ignore_index=True)
    stats_cfg = StatsConfig(null_realizations=16, block_length=2, random_seed=1, allow_time_shift_null=False)
    cand_cfg = CandidateConfig()
    stats = compute_debiased_stats(inflated, _windows(), stats_cfg, cand_cfg)
    assert max(stats.sensor_weights.values()) < 0.6
