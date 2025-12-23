"""Reporting utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml

from .blinding import BlindState


def write_methods_snapshot(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)


def write_results(path: Path, results: Dict[str, Any], blind_state: BlindState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = results.copy()
    if not blind_state.unblind:
        payload["beta_significance"] = "BLINDED"
        payload["p_value"] = "BLINDED"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
