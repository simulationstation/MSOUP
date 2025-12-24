"""Reporting utilities."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable

import yaml

from .blinding import BlindState


FORBIDDEN_KEYS: Iterable[str] = (
    "beta",
    "sigma_beta",
    "p_value",
    "zscore",
    "percentile",
)


def _contains_forbidden_keys(payload: Dict[str, Any], forbidden: Iterable[str]) -> bool:
    for key, value in payload.items():
        if key in forbidden:
            return True
        if isinstance(value, dict):
            if _contains_forbidden_keys(value, forbidden):
                return True
    return False


def write_methods_snapshot(config: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=True)


def write_results(path: Path, results: Dict[str, Any], blind_state: BlindState) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = results.copy()
    if not blind_state.unblind:
        if _contains_forbidden_keys(payload, FORBIDDEN_KEYS):
            raise ValueError("Blinding violation: forbidden keys present in results payload.")
        payload["beta_significance"] = "BLINDED"
        payload["p_value"] = "BLINDED"
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def scan_forbidden_keys_in_dir(path: Path, forbidden: Iterable[str] = FORBIDDEN_KEYS) -> None:
    """Scan output directory for forbidden keys using ripgrep."""
    pattern = r'(\"(' + "|".join(forbidden) + r')\"\s*:|^\s*(' + "|".join(forbidden) + r')\s*:)'
    cmd = [
        "rg",
        "-n",
        "--glob",
        "*.json",
        "--glob",
        "*.yaml",
        "--glob",
        "*.yml",
        pattern,
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode == 0:
        raise ValueError(f"Forbidden keys detected in outputs:\n{result.stdout}")
    if result.returncode not in (1,):
        raise RuntimeError(f"rg failed: {result.stderr}")
