"""Data generation for baseline and extended models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from mverse_channel.config import SimulationConfig
from mverse_channel.metrics.anomaly_score import anomaly_score
from mverse_channel.metrics.correlations import (
    cross_correlation,
    cross_spectral_density,
    coherence_peak_magnitude,
)
from mverse_channel.metrics.memory import fit_ou_timescale
from mverse_channel.metrics.non_gaussian import excess_kurtosis, mardia_kurtosis, mardia_skewness
from mverse_channel.physics.extended_model import simulate_extended
from mverse_channel.physics.measurement_chain import apply_measurement_chain
from mverse_channel.rng import get_rng


def generate_time_series(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    series = simulate_extended(config, rng)
    ia, qa, ib, qb = apply_measurement_chain(
        series["ia"],
        series["qa"],
        series["ib"],
        series["qb"],
        config.measurement,
        rng,
    )
    series.update({"ia": ia, "qa": qa, "ib": ib, "qb": qb})
    return series


def compute_metrics(series: Dict[str, np.ndarray], config: SimulationConfig) -> Dict[str, float]:
    freqs, _, _, coherence = cross_spectral_density(
        series["ia"], series["ib"], config.measurement.sample_rate
    )
    coherence_peak = float(np.max(coherence))
    # Primary detection metric: phase-robust coherence magnitude
    coherence_mag = coherence_peak_magnitude(
        series["ia"], series["ib"], config.measurement.sample_rate
    )
    # Secondary diagnostic only (not used for detection)
    cross_corr = cross_correlation(series["ia"], series["ib"])
    cross_corr_peak = float(np.max(np.abs(cross_corr["corr"])))
    covariance = np.cov(
        np.vstack([series["ia"], series["qa"], series["ib"], series["qb"]]),
        rowvar=True,
    )
    non_gaussian = float(
        0.5
        * (
            excess_kurtosis(series["ia"])
            + excess_kurtosis(series["ib"])
        )
    )
    samples = np.vstack([series["ia"], series["qa"], series["ib"], series["qb"]]).T
    mardia_k = mardia_kurtosis(samples)
    mardia_s = mardia_skewness(samples)
    memory = float(
        0.5
        * (
            fit_ou_timescale(series["ia"], config.dt)
            + fit_ou_timescale(series["ib"], config.dt)
        )
    )
    metrics = {
        "coherence_peak": coherence_peak,
        "coherence_mag": coherence_mag,  # Primary detection metric
        "cross_corr_peak": cross_corr_peak,  # Secondary diagnostic
        "non_gaussian": non_gaussian,
        "mardia_kurtosis": mardia_k,
        "mardia_skewness": mardia_s,
        "memory": memory,
        "covariance": covariance.tolist(),
    }
    # Anomaly score uses coherence_mag (phase-robust) as primary
    metrics["anomaly_score"] = anomaly_score(
        {
            "coherence_mag": coherence_mag,
            "non_gaussian": non_gaussian,
            "memory": memory,
        },
        {
            "coherence_mag": 0.5,  # Primary weight on phase-robust metric
            "non_gaussian": 0.25,
            "memory": 0.25,
        },
    )
    metrics["freqs"] = float(freqs.size)
    return metrics


def generate_metrics(config: SimulationConfig, rng: np.random.Generator) -> Dict[str, float]:
    series = generate_time_series(config, rng)
    return compute_metrics(series, config)


def save_run(
    series: Dict[str, np.ndarray],
    metrics: Dict[str, float],
    config: SimulationConfig,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / "data.npz", **series)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "config.json").write_text(config.model_dump_json(indent=2))


def generate_and_save(config: SimulationConfig, out_dir: str | Path) -> Dict[str, float]:
    rng = get_rng(config.seed)
    series = generate_time_series(config, rng)
    metrics = compute_metrics(series, config)
    save_run(series, metrics, config, Path(out_dir))
    return metrics
