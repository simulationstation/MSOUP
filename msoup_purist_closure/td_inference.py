from __future__ import annotations

import json
import math
import pathlib
import time
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import norm

from .config import MsoupConfig, ProbeConfig
from .data_utils import compute_ddt_prediction
from .model import leak_fraction


DEFAULT_FB = (0.1571, 0.0017)


@dataclass
class HistogramLikelihood:
    bin_edges: np.ndarray
    pmf: np.ndarray
    support: str = "D_dt"  # or H0
    source: Optional[str] = None
    sample_mean: float = math.nan
    sample_std: float = math.nan

    def log_prob(self, value: float, floor: float = 1e-15) -> float:
        idx = np.searchsorted(self.bin_edges, value, side="right") - 1
        if idx < 0 or idx >= len(self.pmf):
            return math.log(floor)
        prob_density = self.pmf[idx] / max(floor, self.bin_edges[idx + 1] - self.bin_edges[idx])
        return math.log(max(floor, prob_density))


@dataclass
class TDLens:
    lens_id: str
    z_lens: float
    z_source: float
    observable_type: str
    kind: str  # "gaussian" or "hist"
    value: Optional[float] = None
    sigma: Optional[float] = None
    histogram: Optional[HistogramLikelihood] = None

    @property
    def label(self) -> str:
        return self.lens_id or self.observable_type


def _resolve_sample_column(df: pd.DataFrame, explicit: Optional[str]) -> str:
    if explicit and explicit in df.columns:
        return explicit
    for cand in ("D_dt", "ddt", "H0", "h0"):
        if cand in df.columns:
            return cand
    raise ValueError("No recognized sample column found in posterior file.")


def _stream_histogram(
    file_path: pathlib.Path,
    column: str,
    bins: int = 180,
    cache_dir: Optional[pathlib.Path] = None,
) -> HistogramLikelihood:
    cache_dir = cache_dir or (file_path.parent / ".td_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{file_path.name}.{column}.{bins}.json"
    cache_path = cache_dir / cache_key
    hist_path = cache_dir / cache_key.replace(".json", ".npz")

    if cache_path.exists() and hist_path.exists():
        try:
            meta = json.loads(cache_path.read_text())
            bin_edges = np.load(hist_path)["bin_edges"]
            pmf = np.load(hist_path)["pmf"]
            return HistogramLikelihood(
                bin_edges=bin_edges,
                pmf=pmf,
                support=meta.get("support", "D_dt"),
                source=str(file_path),
                sample_mean=float(meta.get("sample_mean", math.nan)),
                sample_std=float(meta.get("sample_std", math.nan)),
            )
        except Exception:
            pass

    mins, maxs, count = math.inf, -math.inf, 0
    for chunk in pd.read_csv(file_path, usecols=[column], chunksize=10000):
        vals = chunk[column].to_numpy()
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        vals = vals[finite]
        mins = min(mins, float(np.min(vals)))
        maxs = max(maxs, float(np.max(vals)))
        count += len(vals)

    if not np.isfinite(mins) or not np.isfinite(maxs) or count == 0:
        raise ValueError(f"No finite samples in {file_path}")

    span = maxs - mins
    lo = mins - 0.01 * span
    hi = maxs + 0.01 * span
    edges = np.linspace(lo, hi, bins + 1)
    counts = np.zeros(bins, dtype=float)
    sample_sum = 0.0
    sample_sq = 0.0
    for chunk in pd.read_csv(file_path, usecols=[column], chunksize=20000):
        vals = chunk[column].to_numpy()
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        vals = vals[finite]
        hist, _ = np.histogram(vals, bins=edges)
        counts += hist
        sample_sum += float(np.sum(vals))
        sample_sq += float(np.sum(vals ** 2))

    pmf = counts / max(1.0, np.sum(counts))
    sample_mean = sample_sum / count if count else math.nan
    sample_var = max(0.0, sample_sq / count - sample_mean ** 2) if count else math.nan
    sample_std = math.sqrt(sample_var) if count else math.nan

    np.savez_compressed(hist_path, bin_edges=edges, pmf=pmf)
    cache_path.write_text(
        json.dumps(
            {"support": "H0" if column.lower().startswith("h0") else "D_dt", "sample_mean": sample_mean, "sample_std": sample_std}
        ),
        encoding="utf-8",
    )
    return HistogramLikelihood(
        bin_edges=edges,
        pmf=pmf,
        support="H0" if column.lower().startswith("h0") else "D_dt",
        source=str(file_path),
        sample_mean=sample_mean,
        sample_std=sample_std,
    )


def _resolve_td_paths(base: pathlib.Path, ref: str) -> pathlib.Path:
    ref_path = pathlib.Path(ref)
    if ref_path.is_absolute():
        return ref_path
    return (base / ref_path).expanduser()


def load_td_master(probe: ProbeConfig, data_dir: Optional[pathlib.Path] = None, cache_dir: Optional[pathlib.Path] = None) -> Tuple[List[TDLens], str]:
    path = pathlib.Path(probe.path)
    if not path.exists():
        return [], f"file not found: {probe.path}"
    df = pd.read_csv(path)
    if probe.filter_column and probe.filter_column in df.columns:
        if probe.filter_value is not None:
            df = df[df[probe.filter_column] == probe.filter_value]
    if df.empty:
        return [], "no matching TD rows"

    z_lens_col = probe.z_column or "z_lens"
    if z_lens_col not in df.columns:
        for cand in ["z_lens", "zlens", "z"]:
            if cand in df.columns:
                z_lens_col = cand
                break
    z_source_col = None
    for cand in ["z_source", "zsrc", "zs"]:
        if cand in df.columns:
            z_source_col = cand
            break
    value_col = probe.obs_column or "value"
    sigma_col = probe.sigma_column or "sigma"
    obs_type_col = probe.observable_column or "observable_type"
    file_ref_col = probe.file_ref_column or "file_ref"
    sample_col = probe.sample_column

    missing_cols = [c for c in [z_lens_col, z_source_col, obs_type_col] if c and c not in df.columns]
    if missing_cols:
        return [], f"missing columns: {', '.join(missing_cols)}"

    lenses: List[TDLens] = []
    base_dir = data_dir or path.parent
    for idx, row in df.iterrows():
        zl = row.get(z_lens_col, math.nan)
        zs = row.get(z_source_col, math.nan) if z_source_col else math.nan
        obs_type = row.get(obs_type_col, "D_dt")
        file_ref = row.get(file_ref_col) if file_ref_col in df.columns else None
        lens_id = str(row.get("lens_id", f"lens_{idx}"))

        if not (np.isfinite(zl) and np.isfinite(zs)):
            continue

        if isinstance(file_ref, str) and len(file_ref) > 0:
            posterior_path = _resolve_td_paths(base_dir, file_ref)
            if not posterior_path.exists():
                continue
            preview_df = pd.read_csv(posterior_path, nrows=1)
            try:
                col = _resolve_sample_column(preview_df, sample_col)
            except ValueError:
                continue
            hist = _stream_histogram(posterior_path, col, cache_dir=cache_dir)
            lenses.append(
                TDLens(
                    lens_id=lens_id,
                    z_lens=float(zl),
                    z_source=float(zs),
                    observable_type=str(obs_type),
                    kind="hist",
                    histogram=hist,
                )
            )
            continue

        if value_col not in df.columns or sigma_col not in df.columns:
            continue
        value = row[value_col]
        sigma = row[sigma_col]
        if not (np.isfinite(value) and np.isfinite(sigma) and sigma > 0):
            continue
        lenses.append(
            TDLens(
                lens_id=lens_id,
                z_lens=float(zl),
                z_source=float(zs),
                observable_type=str(obs_type),
                kind="gaussian",
                value=float(value),
                sigma=float(sigma),
            )
        )

    if not lenses:
        return [], "no usable time-delay rows"
    return lenses, "OK"


def compute_h0_from_ddt(ddt_pred: np.ndarray, cfg: MsoupConfig, z_lens: float, z_source: float) -> np.ndarray:
    ddt_lcdm = compute_ddt_prediction(z_lens, z_source, 0.0, cfg)
    base_h0 = cfg.h_early
    with np.errstate(divide="ignore", invalid="ignore"):
        return base_h0 * (ddt_lcdm / np.maximum(1e-30, ddt_pred))


def evaluate_lens_loglike(lens: TDLens, delta_m_grid: np.ndarray, cfg: MsoupConfig) -> np.ndarray:
    preds = np.array([compute_ddt_prediction(lens.z_lens, lens.z_source, float(dm), cfg) for dm in delta_m_grid])
    if lens.kind == "gaussian":
        resid = (preds - lens.value) / lens.sigma
        return -0.5 * resid ** 2 - math.log(math.sqrt(2 * math.pi) * lens.sigma)

    assert lens.histogram is not None
    support = lens.histogram.support
    values = preds
    if support.lower().startswith("h0"):
        values = compute_h0_from_ddt(preds, cfg, lens.z_lens, lens.z_source)
    return np.array([lens.histogram.log_prob(float(v)) for v in values], dtype=float)


def summarize_posterior(delta_m_grid: np.ndarray, loglike: np.ndarray) -> Dict[str, float]:
    logpost = loglike - logsumexp(loglike)
    weights = np.exp(logpost)
    mean = float(np.sum(delta_m_grid * weights))
    median = float(np.interp(0.5, np.cumsum(weights), delta_m_grid))
    cdf = np.cumsum(weights)
    cdf /= cdf[-1]
    low68 = float(np.interp(0.16, cdf, delta_m_grid))
    high68 = float(np.interp(0.84, cdf, delta_m_grid))
    low95 = float(np.interp(0.025, cdf, delta_m_grid))
    high95 = float(np.interp(0.975, cdf, delta_m_grid))
    var = float(np.sum(weights * (delta_m_grid - mean) ** 2))
    sigma = math.sqrt(max(0.0, var))
    return {
        "mean": mean,
        "median": median,
        "low68": low68,
        "high68": high68,
        "low95": low95,
        "high95": high95,
        "sigma": sigma,
    }


def f0_from_delta_m(delta_m: np.ndarray | float, omega_m0: float, omega_L0: float) -> np.ndarray:
    dm_arr = np.asarray(delta_m, dtype=float).reshape(-1)
    f_list = [
        float(leak_fraction(np.array([0.0]), float(dm), omega_m0, omega_L0)[0])
        for dm in dm_arr
    ]
    return np.asarray(f_list, dtype=float).reshape(np.shape(delta_m))


def _information_from_weights(delta_m_grid: np.ndarray, weights: np.ndarray) -> float:
    mean = float(np.sum(delta_m_grid * weights))
    var = float(np.sum(weights * (delta_m_grid - mean) ** 2))
    if var <= 0:
        return 0.0
    return 1.0 / var


def _write_plots(output_dir: pathlib.Path, delta_m_grid: np.ndarray, weights: np.ndarray, f0_weights: np.ndarray, f0_grid: np.ndarray):
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots()
    ax.plot(delta_m_grid, weights, lw=2)
    ax.set_xlabel(r"$\delta_m$")
    ax.set_ylabel("Posterior")
    ax.set_title("TD-only δm posterior")
    (output_dir / "delta_m_posterior.png").parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_dir / "delta_m_posterior.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(f0_grid, f0_weights, lw=2, color="tab:orange")
    ax.set_xlabel(r"$f_0$")
    ax.set_ylabel("Posterior")
    ax.set_title("TD-only $f_0$ posterior")
    fig.tight_layout()
    fig.savefig(output_dir / "f0_posterior.png")
    plt.close(fig)


def _write_report(
    cfg: MsoupConfig,
    output_dir: pathlib.Path,
    summary: Dict[str, float],
    f0_summary: Dict[str, float],
    fb_stats: Dict[str, float],
    lens_rows: List[Dict[str, object]],
    loo: List[Dict[str, object]],
    dominance: List[Dict[str, object]],
    status: str,
    elapsed_s: float,
    peak_rss: float,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "REPORT.md"
    lines = [
        "# TD-only Purist Closure Inference",
        "",
        f"- Status: {status}",
        f"- Elapsed: {elapsed_s:.1f} s",
        f"- Peak RSS: {peak_rss:.1f} MB",
        "",
        "## δm Posterior",
        "",
        f"- mean = {summary['mean']:.4f}, median = {summary['median']:.4f}",
        f"- 68% CI = [{summary['low68']:.4f}, {summary['high68']:.4f}]",
        f"- 95% CI = [{summary['low95']:.4f}, {summary['high95']:.4f}]",
        "",
        "## f0 Posterior",
        "",
        f"- mean = {f0_summary['mean']:.4f}, median = {f0_summary['median']:.4f}",
        f"- 68% CI = [{f0_summary['low68']:.4f}, {f0_summary['high68']:.4f}]",
        f"- 95% CI = [{f0_summary['low95']:.4f}, {f0_summary['high95']:.4f}]",
        "",
        "## Comparison to Planck f_b",
        "",
        f"- f_b (Planck) = {fb_stats['f_b']:.4f} ± {fb_stats['sigma_fb']:.4f}",
        f"- Z = {fb_stats['Z']:.2f}, p-value = {fb_stats['p_value']:.3f}",
        "",
        "## Leave-one-out Diagnostics",
        "",
    ]
    for row in loo:
        lines.append(f"- {row['lens_id']}: median δm = {row['median']:.4f}, sigma = {row['sigma']:.4f}")
    lines.append("")
    lines.append("## Dominance Check")
    for row in dominance:
        flag = " (dominant)" if row["fraction"] >= 0.5 else ""
        lines.append(f"- {row['lens_id']}: info fraction = {row['fraction']:.3f}{flag}")
    lines.append("")
    lines.append("## Per-lens Audit")
    lines.append("")
    lines.append("| lens | z_lens | z_source | obs | pred_map | resid | pull | loglike |")
    lines.append("|------|--------|----------|-----|----------|-------|------|---------|")
    for row in lens_rows:
        lines.append(
            f"| {row['lens_id']} | {row['z_lens']:.3f} | {row['z_source']:.3f} | {row['observable_type']} | "
            f"{row['pred_map']:.2f} | {row['resid']:.2f} | {row['pull']:.2f} | {row['loglike_at_map']:.3f} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    summary_json = {
        "delta_m": summary,
        "f0": f0_summary,
        "fb_comparison": fb_stats,
        "lens_audit": lens_rows,
        "loo": loo,
        "dominance": dominance,
        "status": status,
        "elapsed_s": elapsed_s,
        "peak_rss_mb": peak_rss,
        "config": asdict(cfg),
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary_json, f, indent=2, default=str)

    audit_df = pd.DataFrame(lens_rows)
    audit_df.to_csv(output_dir / "td_audit.csv", index=False)


def run_td_inference(
    cfg: MsoupConfig,
    output_dir: pathlib.Path,
    td_probe: ProbeConfig,
    run_tdlmc: bool = False,
) -> Dict[str, object]:
    start = time.time()
    try:
        import psutil

        start_rss = psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception:
        start_rss = 0.0

    cache_dir = output_dir / ".cache"
    lenses, status = load_td_master(td_probe, cfg.data_dir, cache_dir)
    if not lenses:
        raise RuntimeError(status)

    delta_m_grid = np.linspace(cfg.f0.delta_m_min, cfg.f0.delta_m_max, int(cfg.f0.num_points))
    per_lens_loglike = []
    lens_audit_rows: List[Dict[str, object]] = []
    for lens in lenses:
        ll = evaluate_lens_loglike(lens, delta_m_grid, cfg)
        per_lens_loglike.append(ll)
    total_loglike = np.sum(per_lens_loglike, axis=0)
    summary = summarize_posterior(delta_m_grid, total_loglike)
    weights = np.exp(total_loglike - logsumexp(total_loglike))

    f0_grid = f0_from_delta_m(delta_m_grid, cfg.omega_m0, cfg.omega_L0)
    f0_weights = weights / max(1e-30, np.sum(weights))
    f0_summary = summarize_posterior(f0_grid, np.log(f0_weights + 1e-30))
    fb_mean, fb_sigma = DEFAULT_FB
    fb_diff = f0_summary["mean"] - fb_mean
    Z = fb_diff / math.sqrt(f0_summary["sigma"] ** 2 + fb_sigma ** 2)
    fb_stats = {"f_b": fb_mean, "sigma_fb": fb_sigma, "Z": Z, "p_value": 2 * norm.sf(abs(Z))}

    map_idx = int(np.argmax(weights))
    map_dm = float(delta_m_grid[map_idx])
    for lens, ll in zip(lenses, per_lens_loglike):
        pred_map = compute_ddt_prediction(lens.z_lens, lens.z_source, map_dm, cfg)
        if lens.kind == "gaussian":
            resid = pred_map - lens.value
            pull = resid / lens.sigma
        else:
            hist = lens.histogram
            resid = pred_map - (hist.sample_mean if hist and np.isfinite(hist.sample_mean) else pred_map)
            pull = resid / (hist.sample_std if hist and hist.sample_std > 0 else np.nan)
        lens_audit_rows.append(
            {
                "lens_id": lens.lens_id,
                "z_lens": lens.z_lens,
                "z_source": lens.z_source,
                "observable_type": lens.observable_type,
                "pred_map": pred_map,
                "resid": resid,
                "pull": pull,
                "loglike_at_map": float(ll[map_idx]),
            }
        )

    loo_entries: List[Dict[str, object]] = []
    total_loglike_array = np.array(per_lens_loglike)
    for i, lens in enumerate(lenses):
        loo_loglike = np.sum(np.delete(total_loglike_array, i, axis=0), axis=0)
        loo_summary = summarize_posterior(delta_m_grid, loo_loglike)
        loo_entries.append(
            {"lens_id": lens.lens_id, "median": loo_summary["median"], "sigma": loo_summary["sigma"]}
        )

    info_total = _information_from_weights(delta_m_grid, weights)
    dominance_rows: List[Dict[str, object]] = []
    for i, lens in enumerate(lenses):
        loo_loglike = np.sum(np.delete(total_loglike_array, i, axis=0), axis=0)
        loo_weights = np.exp(loo_loglike - logsumexp(loo_loglike))
        info_loo = _information_from_weights(delta_m_grid, loo_weights)
        contribution = max(0.0, info_total - info_loo)
        frac = contribution / info_total if info_total > 0 else 0.0
        dominance_rows.append({"lens_id": lens.lens_id, "fraction": frac})

    try:
        import psutil

        peak_rss = max(start_rss, psutil.Process().memory_info().rss / (1024 * 1024))
    except Exception:
        peak_rss = float("nan")

    _write_plots(output_dir, delta_m_grid, weights, f0_weights, f0_grid)
    _write_report(
        cfg,
        output_dir,
        summary,
        f0_summary,
        fb_stats,
        lens_audit_rows,
        loo_entries,
        dominance_rows,
        status,
        time.time() - start,
        peak_rss,
    )

    results = {
        "delta_m_grid": delta_m_grid,
        "weights": weights,
        "summary": summary,
        "f0_summary": f0_summary,
        "fb_stats": fb_stats,
        "lens_audit": lens_audit_rows,
        "loo": loo_entries,
        "dominance": dominance_rows,
    }

    if run_tdlmc and cfg.tdlmc:
        tdlmc_path = pathlib.Path(cfg.tdlmc.get("path", ""))
        if tdlmc_path.exists():
            tdlmc_probe = ProbeConfig(
                name="tdlmc",
                path=str(tdlmc_path),
                type="td",
                z_column="z_lens",
                obs_column="value",
                sigma_column="sigma",
                file_ref_column="file_ref",
                observable_column="observable_type",
            )
            tdlmc_lenses, msg = load_td_master(tdlmc_probe, cfg.data_dir, cache_dir)
            if tdlmc_lenses:
                # Compute bias estimate
                tdlmc_ll = [evaluate_lens_loglike(l, delta_m_grid, cfg) for l in tdlmc_lenses]
                tdlmc_total = np.sum(tdlmc_ll, axis=0)
                tdlmc_summary = summarize_posterior(delta_m_grid, tdlmc_total)
                results["tdlmc"] = {"summary": tdlmc_summary, "status": "OK"}
            else:
                results["tdlmc"] = {"summary": None, "status": msg}
        else:
            results["tdlmc"] = {"summary": None, "status": "tdlmc data not found"}

    return results
