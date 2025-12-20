"""
Quality-control filters and diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from .candidates import CandidateResult
from .preprocess import PreprocessResult


@dataclass
class QCDecision:
    keep: bool
    reasons: List[str]


def evaluate_qc(preprocess: PreprocessResult, candidates: CandidateResult, min_arc_pixels: int = 50) -> QCDecision:
    """Decide whether to keep a lens for analysis."""
    reasons: List[str] = []
    arc_pixels = int(np.sum(preprocess.arc_mask))
    if arc_pixels < min_arc_pixels:
        reasons.append(f"insufficient arc pixels ({arc_pixels})")
    if len(candidates.theta) == 0:
        reasons.append("no candidates above threshold")
    keep = len(reasons) == 0
    return QCDecision(keep=keep, reasons=reasons)


def summarize_counts(candidate_map: Dict[str, CandidateResult]) -> np.ndarray:
    """Array of candidate counts per lens for diagnostics."""
    return np.array([len(c.theta) for c in candidate_map.values()])
