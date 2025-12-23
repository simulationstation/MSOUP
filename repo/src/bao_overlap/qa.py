"""Quality assurance checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class QAResult:
    metrics: Dict[str, float]


def basic_checks(z: np.ndarray, weights: np.ndarray) -> QAResult:
    return QAResult(
        metrics={
            "z_min": float(np.min(z)),
            "z_max": float(np.max(z)),
            "weight_mean": float(np.mean(weights)),
        }
    )
