"""Fit baseline and extended models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from scipy import optimize

from mverse_channel.config import InferenceConfig, SimulationConfig
from mverse_channel.inference.likelihoods import gaussian_summary_loglike
from mverse_channel.metrics.anomaly_score import z_score
from mverse_channel.sim.generate_data import generate_metrics


@dataclass
class FitResult:
    params: Dict[str, float]
    loglike: float


def _summary_vector(metrics: Dict[str, float]) -> Dict[str, float]:
    return {
        "coherence_peak": metrics.get("coherence_peak", 0.0),
        "non_gaussian": metrics.get("non_gaussian", 0.0),
        "memory": metrics.get("memory", 0.0),
    }


def fit_baseline(
    data_metrics: Dict[str, float],
    config: SimulationConfig,
    inference: InferenceConfig,
    rng: np.random.Generator,
) -> FitResult:
    def objective(params: np.ndarray) -> float:
        amp_noise = params[0]
        config.measurement.amp_noise = float(amp_noise)
        sim_metrics = generate_metrics(config, rng)
        loglike = gaussian_summary_loglike(
            _summary_vector(data_metrics),
            _summary_vector(sim_metrics),
            inference.metric_sigma,
        )
        return -loglike

    best_params = None
    best_loglike = -np.inf
    for _ in range(inference.restarts):
        init = np.array([config.measurement.amp_noise * (0.5 + rng.random())])
        result = optimize.minimize(
            objective,
            init,
            method="Nelder-Mead",
            options={"maxiter": inference.max_iter},
        )
        loglike = -result.fun
        if loglike > best_loglike:
            best_loglike = loglike
            best_params = {"amp_noise": float(result.x[0])}

    return FitResult(params=best_params or {}, loglike=best_loglike)


def fit_extended(
    data_metrics: Dict[str, float],
    config: SimulationConfig,
    inference: InferenceConfig,
    rng: np.random.Generator,
) -> FitResult:
    def objective(params: np.ndarray) -> float:
        epsilon = params[0]
        rho = params[1]
        config.hidden_channel.enabled = True
        config.hidden_channel.epsilon_max = float(epsilon)
        config.hidden_channel.rho = float(rho)
        sim_metrics = generate_metrics(config, rng)
        loglike = gaussian_summary_loglike(
            _summary_vector(data_metrics),
            _summary_vector(sim_metrics),
            inference.metric_sigma,
        )
        return -loglike

    best_params = None
    best_loglike = -np.inf
    for _ in range(inference.restarts):
        init = np.array([config.hidden_channel.epsilon_max, config.hidden_channel.rho])
        init = init * (0.5 + rng.random(2))
        bounds = [(0.0, 1.0), (0.0, 0.99)]
        result = optimize.minimize(
            objective,
            init,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": inference.max_iter},
        )
        loglike = -result.fun
        if loglike > best_loglike:
            best_loglike = loglike
            best_params = {
                "epsilon_max": float(result.x[0]),
                "rho": float(result.x[1]),
            }

    return FitResult(params=best_params or {}, loglike=best_loglike)


def score_detection(
    data_metrics: Dict[str, float],
    baseline_metrics: Dict[str, float],
) -> float:
    return z_score(
        data_metrics.get("coherence_peak", 0.0),
        baseline_metrics.get("coherence_peak", 0.0),
        baseline_metrics.get("coherence_peak_std", 1.0),
    )
