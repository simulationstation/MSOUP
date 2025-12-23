"""Mock handling and calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MockResult:
    beta_values: np.ndarray
    meta: dict


def summarize_null(beta_values: np.ndarray) -> dict:
    return {
        "mean": float(np.mean(beta_values)),
        "std": float(np.std(beta_values)),
        "n_mocks": int(len(beta_values)),
    }


def compute_p_value(beta_obs: float, beta_values: np.ndarray) -> float:
    return float((np.abs(beta_values) >= abs(beta_obs)).mean())
