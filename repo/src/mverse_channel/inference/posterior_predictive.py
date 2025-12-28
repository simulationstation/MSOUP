"""Posterior predictive simulation helpers."""

from __future__ import annotations

from typing import Dict

import numpy as np

from mverse_channel.config import SimulationConfig
from mverse_channel.sim.generate_data import generate_metrics


def posterior_predictive(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> Dict[str, float]:
    return generate_metrics(config, rng)
