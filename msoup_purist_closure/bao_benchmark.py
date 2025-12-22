from __future__ import annotations

import datetime as _dt
import pathlib
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from .config import DEFAULT_H_EARLY, DEFAULT_OMEGA_L0, DEFAULT_OMEGA_M0, DistanceConfig
from .constants import SPEED_OF_LIGHT_KM_S
from .observables import bao_predict


@dataclass(frozen=True)
class BaoBenchmarkPoint:
    z: float
    observable: str
    expected: float
    tolerance_frac: float
    note: str = ""


BENCHMARK_POINTS: List[BaoBenchmarkPoint] = [
    BaoBenchmarkPoint(0.15, "DV/rd", 4.47, 0.10, "MGS-like low-z DV/rd"),
    BaoBenchmarkPoint(0.38, "DM/rd", 10.48, 0.03, "BOSS DR12 consensus"),
    BaoBenchmarkPoint(0.38, "DH/rd", 24.73, 0.03, "BOSS DR12 consensus"),
    BaoBenchmarkPoint(0.51, "DM/rd", 13.40, 0.03, "BOSS DR12 consensus"),
    BaoBenchmarkPoint(0.51, "DH/rd", 22.87, 0.03, "BOSS DR12 consensus"),
    BaoBenchmarkPoint(0.70, "DM/rd", 17.29, 0.03, "eBOSS-like midpoint"),
    BaoBenchmarkPoint(0.70, "DH/rd", 20.36, 0.03, "eBOSS-like midpoint"),
    BaoBenchmarkPoint(2.33, "DH/rd", 8.67, 0.05, "Ly-alpha high-z check"),
]


def run_bao_benchmark(output_root: Optional[pathlib.Path] = None) -> tuple[pd.DataFrame, pathlib.Path]:
    """
    Evaluate LCDM baseline BAO predictions against embedded benchmarks.

    Returns:
        (results_df, output_dir)
    """
    distance_cfg = DistanceConfig()
    results_dir = pathlib.Path(output_root or "results/msoup_purist_closure") / "benchmarks"
    timestamp = _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    output_dir = results_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    cosmo_kwargs = dict(
        h_early=DEFAULT_H_EARLY,
        omega_m0=DEFAULT_OMEGA_M0,
        omega_L0=DEFAULT_OMEGA_L0,
        c_km_s=SPEED_OF_LIGHT_KM_S,
    )

    rows = []
    for point in BENCHMARK_POINTS:
        pred = bao_predict(
            point.z,
            point.observable,
            delta_m=0.0,
            rd_mpc=distance_cfg.rd_mpc,
            sanity_check=True,
            **cosmo_kwargs,
        )
        rel_err = (pred - point.expected) / point.expected
        within_tol = abs(rel_err) <= point.tolerance_frac
        rows.append(
            {
                "z": point.z,
                "observable": point.observable,
                "expected": point.expected,
                "predicted": pred,
                "relative_error": rel_err,
                "tolerance_frac": point.tolerance_frac,
                "within_tolerance": within_tol,
                "note": point.note,
            }
        )

    df = pd.DataFrame(rows)
    csv_path = output_dir / "bao_benchmark.csv"
    md_path = output_dir / "bao_benchmark.md"
    df.to_csv(csv_path, index=False, float_format="%.6f")

    # Write a compact markdown summary
    lines = [
        "# BAO Benchmark (embedded)",
        "",
        "| z | observable | expected | predicted | rel_error | tol | within | note |",
        "|---|------------|----------|-----------|-----------|-----|--------|------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['z']:.2f} | {r['observable']} | {r['expected']:.3f} | {r['predicted']:.3f} | "
            f"{r['relative_error']:+.3%} | {r['tolerance_frac']:.1%} | {r['within_tolerance']} | {r['note']} |"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")

    return df, output_dir
