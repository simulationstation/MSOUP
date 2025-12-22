"""
BAO diagnostic machinery for msoup_purist_closure distance-mode.

Provides:
- Per-row pulls audit table (CSV + markdown)
- Three-case sanity comparison (LCDM_BASELINE, MODEL_BEST, MODEL_WEAK)
- Per-observable chi2 breakdown
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .observables import bao_predict, canonicalize_bao_observable


@dataclass
class BaoRowResult:
    """Result for a single BAO data row."""
    z: float
    observable: str
    value: float
    sigma: float
    pred: float
    resid: float
    pull: float
    tracer: str
    paper_tag: str


@dataclass
class BaoCaseResult:
    """Result for a single delta_m case."""
    case_name: str
    delta_m: float
    total_chi2: float
    total_dof: int
    chi2_dof: float
    chi2_by_obs: Dict[str, Tuple[float, int]]  # {observable: (chi2, dof)}
    rows: List[BaoRowResult]
    worst_pulls: List[BaoRowResult]  # top 5 by |pull|


def compute_bao_pulls(
    df: pd.DataFrame,
    delta_m: float,
    rd_mpc: float,
    cosmo_kwargs: Dict,
    sanity_check: bool = False,
    rd_fid_conventions: dict | None = None,
) -> Tuple[List[BaoRowResult], Dict[str, Tuple[float, int]], float, int]:
    """
    Compute BAO pulls for all rows in a DataFrame.

    Args:
        df: DataFrame with columns z, observable, value, sigma, tracer, paper_tag
        delta_m: MSOUP delta_m parameter
        rd_mpc: Sound horizon in Mpc
        cosmo_kwargs: Cosmology parameters for bao_predict
        sanity_check: If True, perform sanity assertions in bao_predict

    Returns:
        (rows, chi2_by_obs, total_chi2, total_dof)
    """
    rows = []
    chi2_by_obs = {"DV/rd": (0.0, 0), "DM/rd": (0.0, 0), "DH/rd": (0.0, 0)}
    total_chi2 = 0.0
    total_dof = 0

    for _, row in df.iterrows():
        z = row['z']
        obs = canonicalize_bao_observable(row['observable'])
        val = row['value']
        sig = row['sigma']
        tracer = row.get('tracer', '')
        paper_tag = row.get('paper_tag', '')
        rd_fid_mpc = row.get('rd_fid_mpc')
        rd_scaling = row.get('rd_scaling')

        pred = bao_predict(
            z,
            obs,
            delta_m,
            rd_mpc,
            sanity_check=sanity_check,
            rd_fid_mpc=rd_fid_mpc,
            rd_scaling=rd_scaling,
            rd_fid_conventions=rd_fid_conventions,
            paper_tag=paper_tag,
            **cosmo_kwargs,
        )
        resid = val - pred
        pull = resid / sig
        chi2_i = pull ** 2

        result = BaoRowResult(
            z=z, observable=obs, value=val, sigma=sig,
            pred=pred, resid=resid, pull=pull,
            tracer=tracer, paper_tag=paper_tag
        )
        rows.append(result)

        total_chi2 += chi2_i
        total_dof += 1

        # Accumulate by observable type
        key = canonicalize_bao_observable(obs)
        old_chi2, old_dof = chi2_by_obs[key]
        chi2_by_obs[key] = (old_chi2 + chi2_i, old_dof + 1)

    return rows, chi2_by_obs, total_chi2, total_dof


def get_worst_pulls(rows: List[BaoRowResult], n: int = 5) -> List[BaoRowResult]:
    """Return top n rows by |pull|."""
    sorted_rows = sorted(rows, key=lambda r: abs(r.pull), reverse=True)
    return sorted_rows[:n]


def run_bao_sanity_cases(
    df: pd.DataFrame,
    delta_m_star: float,
    rd_mpc: float,
    cosmo_kwargs: Dict,
    rd_fid_conventions: dict | None = None,
) -> Dict[str, BaoCaseResult]:
    """
    Run BAO sanity diagnostics for three cases:
    - LCDM_BASELINE: delta_m = 0.0
    - MODEL_BEST: delta_m = delta_m_star
    - MODEL_WEAK: delta_m = 0.25 * delta_m_star

    Args:
        df: BAO DataFrame with required columns
        delta_m_star: Best-fit delta_m from kernel mode
        rd_mpc: Sound horizon in Mpc
        cosmo_kwargs: Cosmology parameters

    Returns:
        Dict mapping case_name to BaoCaseResult
    """
    cases = {
        "LCDM_BASELINE": 0.0,
        "MODEL_BEST": delta_m_star,
        "MODEL_WEAK": 0.25 * delta_m_star,
    }

    results = {}
    for case_name, delta_m in cases.items():
        # Run with sanity_check=True for first case only (to catch any issues)
        sanity_check = (case_name == "LCDM_BASELINE")

        rows, chi2_by_obs, total_chi2, total_dof = compute_bao_pulls(
            df, delta_m, rd_mpc, cosmo_kwargs, sanity_check=sanity_check, rd_fid_conventions=rd_fid_conventions
        )

        chi2_dof = total_chi2 / total_dof if total_dof > 0 else np.nan
        worst_pulls = get_worst_pulls(rows, n=5)

        results[case_name] = BaoCaseResult(
            case_name=case_name,
            delta_m=delta_m,
            total_chi2=total_chi2,
            total_dof=total_dof,
            chi2_dof=chi2_dof,
            chi2_by_obs=chi2_by_obs,
            rows=rows,
            worst_pulls=worst_pulls,
        )

    return results


def write_bao_audit_csv(rows: List[BaoRowResult], output_path: pathlib.Path) -> None:
    """Write BAO per-row audit table to CSV."""
    data = []
    for r in rows:
        data.append({
            'z': r.z,
            'observable': r.observable,
            'value': r.value,
            'sigma': r.sigma,
            'pred': r.pred,
            'resid': r.resid,
            'pull': r.pull,
            'tracer': r.tracer,
            'paper_tag': r.paper_tag,
        })
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False, float_format='%.6f')


def format_bao_sanity_markdown(results: Dict[str, BaoCaseResult]) -> str:
    """Format BAO sanity results as markdown for REPORT.md."""
    lines = []
    lines.append("## BAO Sanity Diagnostics")
    lines.append("")

    # Summary table
    lines.append("### Summary by Case")
    lines.append("")
    lines.append("| Case | delta_m | chi2 | dof | chi2/dof |")
    lines.append("|------|---------|------|-----|----------|")
    for case_name in ["LCDM_BASELINE", "MODEL_BEST", "MODEL_WEAK"]:
        if case_name not in results:
            continue
        r = results[case_name]
        lines.append(f"| {case_name} | {r.delta_m:.4f} | {r.total_chi2:.2f} | {r.total_dof} | {r.chi2_dof:.3f} |")
    lines.append("")

    # Per-observable breakdown
    lines.append("### Chi-square by Observable Type")
    lines.append("")
    lines.append("| Case | DV/rd chi2 (N) | DM/rd chi2 (N) | DH/rd chi2 (N) |")
    lines.append("|------|----------------|----------------|----------------|")
    for case_name in ["LCDM_BASELINE", "MODEL_BEST", "MODEL_WEAK"]:
        if case_name not in results:
            continue
        r = results[case_name]
        dv = r.chi2_by_obs.get("DV/rd", (0, 0))
        dm = r.chi2_by_obs.get("DM/rd", (0, 0))
        dh = r.chi2_by_obs.get("DH/rd", (0, 0))
        lines.append(f"| {case_name} | {dv[0]:.2f} ({dv[1]}) | {dm[0]:.2f} ({dm[1]}) | {dh[0]:.2f} ({dh[1]}) |")
    lines.append("")

    # Top 5 worst pulls for each case
    for case_name in ["LCDM_BASELINE", "MODEL_BEST", "MODEL_WEAK"]:
        if case_name not in results:
            continue
        r = results[case_name]
        lines.append(f"### Top 5 Worst Pulls ({case_name})")
        lines.append("")
        lines.append("| z | observable | value | pred | pull | tracer |")
        lines.append("|---|------------|-------|------|------|--------|")
        for wp in r.worst_pulls:
            lines.append(f"| {wp.z:.3f} | {wp.observable} | {wp.value:.3f} | {wp.pred:.3f} | {wp.pull:+.2f} | {wp.tracer} |")
        lines.append("")

    return "\n".join(lines)


def format_bao_pulls_markdown(rows: List[BaoRowResult], case_name: str = "MODEL_BEST") -> str:
    """Format full BAO per-row audit table as markdown."""
    lines = []
    lines.append(f"### BAO Per-Row Pulls ({case_name})")
    lines.append("")
    lines.append("| z | observable | value | sigma | pred | resid | pull | tracer | paper_tag |")
    lines.append("|---|------------|-------|-------|------|-------|------|--------|-----------|")
    for r in rows:
        lines.append(f"| {r.z:.3f} | {r.observable} | {r.value:.3f} | {r.sigma:.3f} | {r.pred:.3f} | {r.resid:+.3f} | {r.pull:+.2f} | {r.tracer} | {r.paper_tag} |")
    lines.append("")
    return "\n".join(lines)


def get_bao_sanity_summary_dict(results: Dict[str, BaoCaseResult]) -> Dict:
    """Get BAO sanity results as a dict for summary.json."""
    summary = {}
    for case_name, r in results.items():
        summary[case_name] = {
            "delta_m": r.delta_m,
            "total_chi2": r.total_chi2,
            "total_dof": r.total_dof,
            "chi2_dof": r.chi2_dof,
            "chi2_by_obs": {k: {"chi2": v[0], "dof": v[1]} for k, v in r.chi2_by_obs.items()},
            "worst_pulls": [
                {"z": wp.z, "observable": wp.observable, "pull": wp.pull}
                for wp in r.worst_pulls
            ],
        }
    return summary
