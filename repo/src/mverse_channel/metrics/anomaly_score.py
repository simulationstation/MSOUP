"""Anomaly scoring.

Provides calibrated detection statistics based on null distributions.
Primary detection uses coherence magnitude (phase-robust), not correlation sign.
"""

from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass

import numpy as np


@dataclass
class CalibratedThresholds:
    """Thresholds calibrated from null distribution."""
    alpha: float
    n_null: int
    coherence_mag_threshold: float
    anomaly_score_threshold: float
    seeds_used: List[int]

    def to_dict(self) -> Dict:
        return {
            "alpha": self.alpha,
            "n_null": self.n_null,
            "coherence_mag_threshold": self.coherence_mag_threshold,
            "anomaly_score_threshold": self.anomaly_score_threshold,
            "seeds_used": self.seeds_used,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CalibratedThresholds":
        return cls(
            alpha=d["alpha"],
            n_null=d["n_null"],
            coherence_mag_threshold=d["coherence_mag_threshold"],
            anomaly_score_threshold=d["anomaly_score_threshold"],
            seeds_used=d["seeds_used"],
        )


def anomaly_score(metrics: Dict[str, float], weights: Dict[str, float]) -> float:
    """Weighted anomaly score from metrics.

    Uses coherence_mag (phase-robust) as primary, not cross_corr_peak.
    """
    score = 0.0
    for key, weight in weights.items():
        score += weight * metrics.get(key, 0.0)
    return float(score)


def z_score(value: float, mean: float, std: float) -> float:
    """Compute z-score relative to null distribution."""
    return float((value - mean) / (std + 1e-12))


def compute_null_quantile(values: List[float], alpha: float) -> float:
    """Compute (1-alpha) quantile threshold from null distribution.

    For alpha=0.01, returns 99th percentile.
    """
    return float(np.percentile(values, 100 * (1 - alpha)))


def detection_decision(
    coherence_mag: float,
    anomaly: float,
    thresholds: CalibratedThresholds,
) -> bool:
    """Make detection decision using calibrated thresholds.

    Detection requires BOTH:
    - coherence_mag > calibrated threshold
    - anomaly_score > calibrated threshold
    """
    return (
        coherence_mag > thresholds.coherence_mag_threshold
        and anomaly > thresholds.anomaly_score_threshold
    )
