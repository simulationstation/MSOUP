"""Anomaly scoring."""

from __future__ import annotations

from typing import Dict

import numpy as np


def anomaly_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    score = 0.0
    for key, weight in weights.items():
        score += weight * metrics.get(key, 0.0)
    return float(score)


def z_score(value: float, mean: float, std: float) -> float:
    return float((value - mean) / (std + 1e-12))
