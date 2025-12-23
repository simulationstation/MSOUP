"""BAO reconstruction placeholder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ReconstructionResult:
    displaced: np.ndarray
    meta: dict


def reconstruct_positions(xyz: np.ndarray, smoothing_radius: float) -> ReconstructionResult:
    """Simple placeholder: no displacement but store metadata."""
    return ReconstructionResult(displaced=xyz.copy(), meta={"smoothing_radius": smoothing_radius})
