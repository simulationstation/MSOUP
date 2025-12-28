"""Random number utilities."""

from __future__ import annotations

import numpy as np


def get_rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)
