"""Noise generation utilities."""

from __future__ import annotations

import numpy as np


def ou_process(
    n_steps: int,
    dt: float,
    tau: float,
    rng: np.random.Generator,
    sigma: float = 1.0,
) -> np.ndarray:
    """Generate Ornstein-Uhlenbeck process with mean zero."""
    x = np.zeros(n_steps)
    if tau <= 0:
        return x
    theta = 1.0 / tau
    for i in range(1, n_steps):
        dx = -theta * x[i - 1] * dt + sigma * np.sqrt(dt) * rng.normal()
        x[i] = x[i - 1] + dx
    return x
