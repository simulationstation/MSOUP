"""Preregistration loader and schema validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable

import yaml


REQUIRED_TOP_LEVEL_KEYS: Iterable[str] = (
    "version",
    "status",
    "lock_date",
    "research_question",
    "primary_endpoint",
    "primary_dataset",
    "environment_metric",
    "correlation",
    "weights",
    "covariance",
    "bao_fitting",
    "reconstruction",
    "environment_binning",
    "hierarchical_inference",
    "blinding",
    "decision_criteria",
    "secondary_endpoints",
    "robustness_checks",
    "stopping_rules",
    "fiducial_cosmology",
    "random_seed",
    "mocks",
    "qa",
    "reporting",
)


def _ensure_mapping(payload: Any, path: Path) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError(f"Preregistration at {path} must be a YAML mapping.")
    return payload


def load_prereg(path: str | Path) -> Dict[str, Any]:
    """Load and validate preregistration YAML with top-level schema."""
    path = Path(path)
    with open(path, "r", encoding="utf-8") as handle:
        prereg = _ensure_mapping(yaml.safe_load(handle), path)

    if "analysis" in prereg:
        raise ValueError("Preregistration schema must not contain an 'analysis' block.")

    missing = [key for key in REQUIRED_TOP_LEVEL_KEYS if key not in prereg]
    if missing:
        raise ValueError(f"Preregistration missing required keys: {missing}")

    return prereg
