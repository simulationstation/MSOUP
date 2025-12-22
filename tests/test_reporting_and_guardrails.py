import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msoup_pocket_test.geometry import GeometryResult, evaluate_geometry
from msoup_pocket_test.config import GeometryConfig
from msoup_pocket_test.report import compute_counts_coverage, generate_report
from msoup_pocket_test.run import generate_summary_json
from msoup_pocket_test.stats import GlobalStats
from msoup_pocket_test.windows import Window


def _geom_positions(sensors):
    return pd.DataFrame({
        "satellite": sensors,
        "x": np.arange(len(sensors)) * 1000.0,
        "y": np.arange(len(sensors)) * 1000.0,
        "z": np.arange(len(sensors)) * 1000.0,
    })


def test_geometry_not_testable_reason():
    sensors = ["A", "B", "C"]
    candidate_frame = pd.DataFrame({
        "sensor": sensors,
        "peak_time": pd.to_datetime(["2024-01-01T00:00:00Z"] * len(sensors)),
    })
    cfg = GeometryConfig(min_sensors=1, min_events=5, n_shuffles=2)
    geom = evaluate_geometry(candidate_frame, _geom_positions(sensors), cfg, run_mode="sanity")
    assert geom.status == "NOT_TESTABLE"
    assert geom.chi2_obs is None
    assert geom.p_value is None
    assert "insufficient coincident events" in (geom.reason or "")


def test_nan_raises_full_mode(monkeypatch):
    sensors = ["A", "B", "C", "D", "E"]
    candidate_frame = pd.DataFrame({
        "sensor": sensors,
        "peak_time": pd.to_datetime(["2024-01-01T00:00:00Z"] * len(sensors)),
    })
    cfg = GeometryConfig(min_sensors=1, min_events=1, n_shuffles=1)

    monkeypatch.setattr("msoup_pocket_test.geometry._compute_chi2", lambda *args, **kwargs: np.nan)
    with pytest.raises(ValueError):
        evaluate_geometry(candidate_frame, _geom_positions(sensors), cfg, run_mode="full")


def test_counts_section_present(tmp_path: Path):
    stats = GlobalStats(
        fano_obs=1.0, fano_null_mean=1.0, fano_null_std=0.1, fano_z=0.0, p_fano=0.5,
        c_obs=0.2, c_null_mean=0.1, c_null_std=0.05, c_excess=0.1, c_excess_std=0.05,
        c_z=2.0, p_clustering=0.01, p_clustering_two_sided=0.02, n_resamples=10, sensor_weights={"A": 1.0},
    )
    geom = GeometryResult(chi2_obs=None, null_chi2s=None, p_value=None, n_events=0, status="NOT_TESTABLE", reason="synthetic", speed_kmps=None, normal=None)
    windows = [Window(sensor="A", start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-01T00:10:00"), cadence_seconds=30, quality_score=1.0)]
    candidates = pd.DataFrame({"sensor": ["A"], "sigma": [4.0], "window_index": [0]})
    counts = compute_counts_coverage({"A": pd.DataFrame({"time": [pd.Timestamp("2024-01-01")], "sensor": ["A"], "value": [1.0]})}, windows, candidates, neighbor_time=30.0, neighbor_angle=25.0, thresholds=[2.0, 3.0, 4.0], geom=geom)
    report = generate_report(tmp_path, stats, geom, cross_modality_ok=False, run_label="test", config_path=None, counts_coverage=counts)
    content = Path(report).read_text()
    assert "Counts & Coverage" in content
    assert "Threshold counts" in content


def test_no_nan_in_summary(tmp_path: Path):
    stats = GlobalStats(
        fano_obs=1.0, fano_null_mean=1.0, fano_null_std=0.1, fano_z=0.0, p_fano=0.5,
        c_obs=0.2, c_null_mean=0.1, c_null_std=0.05, c_excess=0.1, c_excess_std=0.05,
        c_z=2.0, p_clustering=0.01, p_clustering_two_sided=0.02, n_resamples=10, sensor_weights={"A": 1.0},
    )
    geom = GeometryResult(chi2_obs=1.0, null_chi2s=np.array([1.0, 2.0]), p_value=0.5, n_events=5, status="COMPLETED", reason=None, speed_kmps=300.0, normal=np.ones(3))
    counts = {
        "input_coverage": {"n_sensors_total": 1, "n_sensors_used": 1, "n_windows_total": 1, "window_duration_s": {"min": 10, "median": 10, "max": 10, "p10": 10, "p50": 10, "p90": 10, "p99": 10}, "per_sensor_masked_fraction": {"min": 0.0, "median": 0.0, "max": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}},
        "candidate_catalog": {"n_candidates_total": 1, "per_sensor": {"min": 1, "median": 1, "max": 1, "p10": 1, "p50": 1, "p90": 1, "p99": 1}, "per_window": {"min": 1, "median": 1, "max": 1, "p10": 1, "p50": 1, "p90": 1, "p99": 1}, "sigma_distribution": {"min": 4, "median": 4, "max": 4, "p50": 4, "p90": 4, "p99": 4}, "threshold_counts": {">=2": 1}},
        "pair_statistics": {"n_pairs_total": 0, "n_sensors_with_pairs": 1, "pairs_per_sensor": {"min": 0, "median": 0, "max": 0, "p10": 0, "p50": 0, "p90": 0, "p99": 0}, "neighbor_time_s": 30.0, "neighbor_angle_deg": 25.0},
        "geometry_preconditions": {"n_candidate_events_total": 1, "n_coincidences_used": 1, "n_events_tested_geometry": 5, "reason": None},
    }
    summary_path = generate_summary_json(tmp_path, stats, geom, counts, run_mode="full")
    data = json.loads(Path(summary_path).read_text())
    def _walk_numbers(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                yield from _walk_numbers(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from _walk_numbers(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            yield float(obj)
    assert all(np.isfinite(x) for x in _walk_numbers(data))


def test_k1_reports_signed_and_two_sided(tmp_path: Path):
    stats = GlobalStats(
        fano_obs=1.0, fano_null_mean=1.0, fano_null_std=0.1, fano_z=0.0, p_fano=0.5,
        c_obs=0.2, c_null_mean=0.1, c_null_std=0.05, c_excess=0.1, c_excess_std=0.05,
        c_z=-2.0, p_clustering=0.8, p_clustering_two_sided=0.05, n_resamples=5, sensor_weights={"A": 1.0},
    )
    geom = GeometryResult(chi2_obs=None, null_chi2s=None, p_value=None, n_events=0, status="NOT_TESTABLE", reason="synthetic", speed_kmps=None, normal=None)
    counts = {
        "input_coverage": {"n_sensors_total": 1, "n_sensors_used": 1, "n_windows_total": 1, "window_duration_s": {"min": 10, "median": 10, "max": 10, "p10": 10, "p50": 10, "p90": 10, "p99": 10}, "per_sensor_masked_fraction": {"min": 0.0, "median": 0.0, "max": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "p99": 0.0}},
        "candidate_catalog": {"n_candidates_total": 1, "per_sensor": {"min": 1, "median": 1, "max": 1, "p10": 1, "p50": 1, "p90": 1, "p99": 1}, "per_window": {"min": 1, "median": 1, "max": 1, "p10": 1, "p50": 1, "p90": 1, "p99": 1}, "sigma_distribution": {"min": 4, "median": 4, "max": 4, "p50": 4, "p90": 4, "p99": 4}, "threshold_counts": {">=2": 1}},
        "pair_statistics": {"n_pairs_total": 0, "n_sensors_with_pairs": 1, "pairs_per_sensor": {"min": 0, "median": 0, "max": 0, "p10": 0, "p50": 0, "p90": 0, "p99": 0}, "neighbor_time_s": 30.0, "neighbor_angle_deg": 25.0},
        "geometry_preconditions": {"n_candidate_events_total": 1, "n_coincidences_used": 1, "n_events_tested_geometry": 0, "reason": "synthetic"},
    }
    summary_path = generate_summary_json(tmp_path, stats, geom, counts, run_mode="sanity")
    data = json.loads(Path(summary_path).read_text())
    assert "c_z" in data["statistics"]
    assert "p_clustering_two_sided" in data["statistics"]
