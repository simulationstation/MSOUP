#!/usr/bin/env python
"""Self-audit checks for preregistered BAO overlap pipeline."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import sys

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bao_overlap.prereg import load_prereg


FORBIDDEN_KEYS = {"beta", "sigma_beta", "p_value", "zscore", "percentile"}


def _scan_forbidden_keys(payload: Dict[str, Any], forbidden: Iterable[str]) -> bool:
    for key, value in payload.items():
        if key in forbidden:
            return True
        if isinstance(value, dict):
            if _scan_forbidden_keys(value, forbidden):
                return True
    return False


def _scan_text_forbidden(text: str) -> bool:
    pattern = re.compile(r"\b(beta|sigma_beta|p_value|zscore|percentile)\b")
    return bool(pattern.search(text))


def _load_payload(path: Path) -> Dict[str, Any] | None:
    try:
        if path.suffix == ".json":
            return json.loads(path.read_text(encoding="utf-8"))
        if path.suffix in {".yaml", ".yml"}:
            return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return None


def check_prereg_schema(prereg_path: Path) -> Tuple[bool, str]:
    try:
        prereg = load_prereg(prereg_path)
    except Exception as exc:
        return False, f"Preregistration schema load failed: {exc}"

    if "analysis" in prereg:
        return False, "Preregistration contains forbidden 'analysis' block."
    return True, "Preregistration schema loaded with top-level keys."


def check_prereg_analysis_usage(repo_root: Path) -> Tuple[bool, str]:
    offenders: List[Path] = []
    for path in (repo_root / "src").rglob("*.py"):
        content = path.read_text(encoding="utf-8")
        if "prereg[\"analysis\"]" in content or "prereg['analysis']" in content:
            offenders.append(path)
    for path in (repo_root / "scripts").rglob("*.py"):
        if path.name == "self_audit.py":
            continue
        content = path.read_text(encoding="utf-8")
        if "prereg[\"analysis\"]" in content or "prereg['analysis']" in content:
            offenders.append(path)
    if offenders:
        return False, f"Found prereg['analysis'] usage in: {', '.join(str(p) for p in offenders)}"
    return True, "No prereg['analysis'] access found."


def check_backend_imports() -> Tuple[bool, str]:
    for module in ("Corrfunc", "treecorr", "pycorr"):
        try:
            __import__(module)
            return True, f"Backend import available: {module}"
        except Exception:
            continue
    return False, "No correlation backend (Corrfunc/TreeCorr/pycorr) importable."


def check_wedge_bounds(prereg_path: Path, repo_root: Path) -> Tuple[bool, str]:
    prereg = load_prereg(prereg_path)
    try:
        wedge_cfg = prereg["correlation"]["wedges"]["tangential"]
        mu_min = wedge_cfg.get("mu_min")
        mu_max = wedge_cfg.get("mu_max")
        if not isinstance(mu_min, (int, float)) or not isinstance(mu_max, (int, float)):
            raise ValueError("Wedge bounds must be numeric.")
    except Exception as exc:
        return False, f"Wedge bounds invalid: {exc}"
    run_pipeline = (repo_root / "scripts" / "run_pipeline.py").read_text(encoding="utf-8")
    if "parse_wedge_bounds" not in run_pipeline:
        return False, "run_pipeline.py does not apply numeric wedge bounds."
    return True, "Wedge bounds are numeric and applied via parse_wedge_bounds."


def check_covariance(results_dir: Path) -> Tuple[bool, str]:
    cov_path = results_dir / "covariance" / "xi_wedge_covariance.npy"
    if not cov_path.exists():
        cov_path = results_dir / "xi_wedge_covariance.npy"
    if not cov_path.exists():
        return False, "Covariance file missing."
    cov = np.load(cov_path)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        return False, "Covariance matrix not square."
    if np.allclose(cov, np.eye(cov.shape[0])):
        return False, "Covariance matrix is identity (placeholder)."
    return True, f"Covariance saved and non-identity at {cov_path}."


def check_paper_package(results_dir: Path) -> Tuple[bool, str]:
    package_dir = results_dir / "paper_package"
    required = [
        package_dir / "metadata.json",
        package_dir / "xi_wedge.npz",
        package_dir / "figures" / "xi_tangential.png",
        package_dir / "xi_wedge_covariance.npy",
    ]
    if (results_dir / "blinded_results.json").exists():
        required.append(package_dir / "blinded_results.json")
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return False, f"Paper package missing artifacts: {missing}"
    return True, "Paper package directory exists with required artifacts."


def check_blinding_leaks(results_dir: Path) -> Tuple[bool, str]:
    blinded = (results_dir / "blinded_results.json").exists()
    if not blinded:
        return True, "No blinded_results.json found; skipping leak scan."
    offenders: List[str] = []
    for path in results_dir.rglob("*"):
        if path.is_dir():
            continue
        if path.suffix in {".json", ".yaml", ".yml"}:
            payload = _load_payload(path)
            if isinstance(payload, dict) and _scan_forbidden_keys(payload, FORBIDDEN_KEYS):
                offenders.append(str(path))
        elif path.suffix in {".txt", ".log"}:
            if _scan_text_forbidden(path.read_text(encoding="utf-8")):
                offenders.append(str(path))
    if offenders:
        return False, f"Forbidden keys leaked in: {offenders}"
    return True, "No forbidden keys found in blinded outputs."


def run_checks(results_dir: Path, prereg_path: Path, repo_root: Path) -> List[Tuple[str, bool, str]]:
    checks = [
        ("prereg_schema", *check_prereg_schema(prereg_path)),
        ("prereg_analysis_usage", *check_prereg_analysis_usage(repo_root)),
        ("backend_imports", *check_backend_imports()),
        ("wedge_bounds", *check_wedge_bounds(prereg_path, repo_root)),
        ("covariance", *check_covariance(results_dir)),
        ("paper_package", *check_paper_package(results_dir)),
        ("blinding_leaks", *check_blinding_leaks(results_dir)),
    ]
    return checks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument(
        "--prereg",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "preregistration.yaml",
    )
    args = parser.parse_args()

    results_dir = args.results_dir
    repo_root = Path(__file__).parent.parent

    checks = run_checks(results_dir, args.prereg, repo_root)
    failures = 0
    for name, passed, message in checks:
        status = "PASS" if passed else "FAIL"
        print(f"{status}: {name} - {message}")
        if not passed:
            failures += 1

    print("\nSummary:")
    print(f"PASS: {len(checks) - failures}")
    print(f"FAIL: {failures}")

    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
