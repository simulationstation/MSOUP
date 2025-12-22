"""
Guardrails for reporting: enforce finite statistics and safe null handling.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Set


def _iter_numbers(obj: Any, prefix: str = "") -> Iterable[tuple[str, float]]:
    """Yield key-paths and numeric values from nested dicts/lists."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _iter_numbers(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _iter_numbers(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


def assert_finite_metrics(metrics: dict, allow_null_keys: Set[str], mode: str = "full") -> None:
    """
    Ensure no NaN/Inf values leak into reports in full mode.

    Args:
        metrics: nested dictionary of metrics (e.g., summary.json contents)
        allow_null_keys: set of key paths allowed to be null (e.g., geometry chi2)
        mode: 'full' or 'sanity'; only full raises on NaN/Inf.
    """
    bad = []
    for keypath, value in _iter_numbers(metrics):
        if keypath in allow_null_keys:
            continue
        if math.isnan(value) or math.isinf(value):
            bad.append((keypath, value))
    if bad and mode == "full":
        msg = "; ".join(f"{k}={v}" for k, v in bad if k not in allow_null_keys)
        if msg:
            raise ValueError(f"Non-finite metric(s) encountered: {msg}")
