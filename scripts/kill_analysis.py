"""
COWLS Field-Level Detection Kill Analysis

Systematic stress-testing of the reported Z_corr ≈ 3.67 signal.
"""

from __future__ import annotations

import json
import re
import sys
import argparse
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter, laplace

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cowls_field_study.io import discover_lenses, read_fits
from cowls_field_study.products import list_band_products, choose_noise_psf, choose_residual_product
from cowls_field_study.preprocess import load_preprocess, build_arc_mask
from cowls_field_study.ring_profile import compute_ring_profile
from cowls_field_study.stats_field import compute_field_stats
from cowls_field_study.nulls import draw_null_statistics
from cowls_field_study.window import build_window
from cowls_field_study.config import FieldStudyConfig
from cowls_field_study.kill_calibration import (
    LensNullContext,
    build_global_null_distribution,
    compute_low_m_ratio,
    empirical_p_value,
    prepare_null_baseline,
)
from cowls_field_study.highpass import HighpassConfig, mask_ring_profile


@dataclass
class LensData:
    """Parsed per-lens results."""
    lens_id: str
    band: str
    score_bin: str
    mode: str
    t_corr: float
    t_pow: float
    z_corr: float
    z_pow: float
    t_corr_hp: float | None = None
    t_pow_hp: float | None = None
    z_corr_hp: float | None = None
    z_pow_hp: float | None = None


def parse_report(report_path: Path) -> Tuple[Dict, List[LensData]]:
    """Parse the report.md file to extract per-lens data."""
    text = report_path.read_text()

    # Extract global stats
    global_stats = {}
    for line in text.split('\n'):
        if 'Global Z_corr:' in line:
            global_stats['global_z_corr'] = float(line.split(':')[1].strip())
        elif 'Global Z_pow:' in line:
            global_stats['global_z_pow'] = float(line.split(':')[1].strip())
        elif 'Global Z_corr_hp:' in line:
            global_stats['global_z_corr_hp'] = float(line.split(':')[1].strip())
        elif 'Global Z_pow_hp:' in line:
            global_stats['global_z_pow_hp'] = float(line.split(':')[1].strip())
        elif 'Mean Z_corr (model):' in line:
            global_stats['mean_z_corr'] = float(line.split(':')[1].strip())
        elif 'Mean Z_pow  (model):' in line:
            global_stats['mean_z_pow'] = float(line.split(':')[1].strip())
        elif 'Mean Z_corr_hp (model):' in line:
            global_stats['mean_z_corr_hp'] = float(line.split(':')[1].strip())
        elif 'Mean Z_pow_hp  (model):' in line:
            global_stats['mean_z_pow_hp'] = float(line.split(':')[1].strip())
        elif 'processed lenses:' in line:
            global_stats['n_lenses'] = int(line.split(':')[1].strip())

    # Parse per-lens lines
    lenses = []
    for line in text.splitlines():
        if not line.startswith("- "):
            continue
        header_match = re.match(r'- (\S+) \(([^,]+), ([^,]+), ([^)]+)\):', line)
        if not header_match:
            continue
        lens_id, band, score_bin, mode = header_match.groups()
        metrics_part = line.split(":", 1)[1]
        metrics = dict(re.findall(r'([A-Za-z_]+)=([-\d.nan]+)', metrics_part))
        lenses.append(LensData(
            lens_id=lens_id,
            band=band,
            score_bin=score_bin,
            mode=mode,
            t_corr=float(metrics.get("T_corr", "nan")),
            t_pow=float(metrics.get("T_pow", "nan")),
            z_corr=float(metrics.get("Z_corr", "nan")),
            z_pow=float(metrics.get("Z_pow", "nan")),
            t_corr_hp=float(metrics.get("T_corr_hp", "nan")) if "T_corr_hp" in metrics else None,
            t_pow_hp=float(metrics.get("T_pow_hp", "nan")) if "T_pow_hp" in metrics else None,
            z_corr_hp=float(metrics.get("Z_corr_hp", "nan")) if "Z_corr_hp" in metrics else None,
            z_pow_hp=float(metrics.get("Z_pow_hp", "nan")) if "Z_pow_hp" in metrics else None,
        ))

    return global_stats, lenses


def compute_global_z(z_values: np.ndarray) -> float:
    """Compute global Z from individual Z scores."""
    if len(z_values) == 0:
        return 0.0
    return float(np.nanmean(z_values) * np.sqrt(len(z_values)))


def section_a_reproduce_and_dominance(lenses: List[LensData], plot_dir: Path) -> Dict:
    """Section A: Reproduce headline numbers and check dominance."""
    print("\n" + "="*60)
    print("SECTION A: Reproduce Headline Numbers & Dominance Analysis")
    print("="*60)

    z_corr = np.array([l.z_corr for l in lenses])
    z_pow = np.array([l.z_pow for l in lenses])
    z_corr_hp_full = np.array([l.z_corr_hp if l.z_corr_hp is not None else np.nan for l in lenses])
    z_corr_hp = np.array([l.z_corr_hp for l in lenses if l.z_corr_hp is not None])
    n = len(lenses)

    mean_z_corr = np.mean(z_corr)
    mean_z_pow = np.mean(z_pow)
    global_z_corr = mean_z_corr * np.sqrt(n)
    global_z_pow = mean_z_pow * np.sqrt(n)

    global_z_corr_hp = None
    mean_z_corr_hp = None
    if z_corr_hp.size:
        mean_z_corr_hp = np.nanmean(z_corr_hp)
        global_z_corr_hp = mean_z_corr_hp * np.sqrt(z_corr_hp.size)

    print(f"\nReproduced from per-lens data:")
    print(f"  n_lenses = {n}")
    print(f"  mean(Z_corr) = {mean_z_corr:.4f}")
    print(f"  mean(Z_pow) = {mean_z_pow:.4f}")
    print(f"  Global Z_corr = {global_z_corr:.3f}")
    print(f"  Global Z_pow = {global_z_pow:.3f}")

    print(f"\nZ_corr distribution:")
    print(f"  min = {np.min(z_corr):.3f}")
    print(f"  median = {np.median(z_corr):.3f}")
    print(f"  max = {np.max(z_corr):.3f}")
    print(f"  std = {np.std(z_corr):.3f}")
    if mean_z_corr_hp is not None:
        print(f"  mean(Z_corr_hp) = {mean_z_corr_hp:.3f}, Global Z_corr_hp = {global_z_corr_hp:.3f}")

    # Top 10 by Z_corr
    sorted_idx = np.argsort(z_corr)[::-1]
    print(f"\nTop 10 lenses by Z_corr:")
    for i in range(min(10, n)):
        idx = sorted_idx[i]
        l = lenses[idx]
        print(f"  {i+1}. {l.lens_id} ({l.score_bin}): Z_corr={l.z_corr:.3f}, Z_pow={l.z_pow:.3f}")

    # Top 10 by |Z_corr|
    sorted_idx_abs = np.argsort(np.abs(z_corr))[::-1]
    print(f"\nTop 10 lenses by |Z_corr|:")
    for i in range(min(10, n)):
        idx = sorted_idx_abs[i]
        l = lenses[idx]
        print(f"  {i+1}. {l.lens_id} ({l.score_bin}): Z_corr={l.z_corr:.3f}, Z_pow={l.z_pow:.3f}")

    # Dominance analysis
    # Global Z = mean(Z) * sqrt(n), so contribution of lens i is z_i / n * sqrt(n) = z_i / sqrt(n)
    # Total is sum(z_i) / sqrt(n) = mean(z) * sqrt(n)
    # Fraction from top k: sum of top k z_i / sum of all z_i

    total_z_sum = np.sum(z_corr)

    def dominance_fraction(k):
        top_k_sum = np.sum(np.sort(z_corr)[-k:])
        return top_k_sum / total_z_sum if total_z_sum != 0 else 0.0

    def mean_without_top_k(k):
        sorted_z = np.sort(z_corr)
        remaining = sorted_z[:-k] if k < len(sorted_z) else []
        return np.mean(remaining) if len(remaining) > 0 else 0.0

    dom_1 = dominance_fraction(1)
    dom_3 = dominance_fraction(3)
    dom_5 = dominance_fraction(5)

    mean_without_1 = mean_without_top_k(1)
    mean_without_3 = mean_without_top_k(3)
    mean_without_5 = mean_without_top_k(5)

    global_without_1 = mean_without_1 * np.sqrt(n - 1)
    global_without_3 = mean_without_3 * np.sqrt(n - 3)
    global_without_5 = mean_without_5 * np.sqrt(n - 5)

    print(f"\nDominance analysis:")
    print(f"  Fraction of Z_sum from top 1: {dom_1:.3f}")
    print(f"  Fraction of Z_sum from top 3: {dom_3:.3f}")
    print(f"  Fraction of Z_sum from top 5: {dom_5:.3f}")
    print(f"\n  mean(Z_corr) without top 1: {mean_without_1:.4f} (Global: {global_without_1:.3f})")
    print(f"  mean(Z_corr) without top 3: {mean_without_3:.4f} (Global: {global_without_3:.3f})")
    print(f"  mean(Z_corr) without top 5: {mean_without_5:.4f} (Global: {global_without_5:.3f})")

    # Plot Z_corr distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(z_corr, bins=15, edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', label='Null expectation')
    ax.axvline(np.mean(z_corr), color='blue', linestyle='-', label=f'Mean = {np.mean(z_corr):.3f}')
    ax.set_xlabel('Z_corr')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of per-lens Z_corr')
    ax.legend()

    ax = axes[1]
    colors = {'M25': 'blue', 'S10': 'green', 'S11': 'orange', 'S12': 'red'}
    for l in lenses:
        ax.scatter(l.z_corr, l.z_pow, c=colors.get(l.score_bin, 'gray'),
                   s=50, alpha=0.7, label=l.score_bin)
    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Z_corr')
    ax.set_ylabel('Z_pow')
    ax.set_title('Z_corr vs Z_pow by score bin')

    plt.tight_layout()
    plt.savefig(plot_dir / 'a_zcorr_distribution.png', dpi=150)
    plt.close()

    return {
        'n_lenses': n,
        'mean_z_corr': float(mean_z_corr),
        'mean_z_pow': float(mean_z_pow),
        'global_z_corr': float(global_z_corr),
        'global_z_pow': float(global_z_pow),
        'mean_z_corr_hp': float(mean_z_corr_hp) if mean_z_corr_hp is not None else None,
        'global_z_corr_hp': float(global_z_corr_hp) if global_z_corr_hp is not None else None,
        'z_corr_min': float(np.min(z_corr)),
        'z_corr_median': float(np.median(z_corr)),
        'z_corr_max': float(np.max(z_corr)),
        'z_corr_std': float(np.std(z_corr)),
        'dominance_top1': float(dom_1),
        'dominance_top3': float(dom_3),
        'dominance_top5': float(dom_5),
        'global_without_top1': float(global_without_1),
        'global_without_top3': float(global_without_3),
        'global_without_top5': float(global_without_5),
    }


def section_b_jackknife(lenses: List[LensData], plot_dir: Path) -> Dict:
    """Section B: Leave-one-out and jackknife analysis."""
    print("\n" + "="*60)
    print("SECTION B: Leave-One-Out and Jackknife Analysis")
    print("="*60)

    z_corr = np.array([l.z_corr for l in lenses])
    n = len(lenses)

    # Leave-one-out global Z_corr
    loo_global = []
    for i in range(n):
        z_without_i = np.delete(z_corr, i)
        global_without_i = np.mean(z_without_i) * np.sqrt(n - 1)
        loo_global.append(global_without_i)

    loo_global = np.array(loo_global)
    original_global = np.mean(z_corr) * np.sqrt(n)

    print(f"\nLeave-one-out Global Z_corr:")
    print(f"  Original: {original_global:.3f}")
    print(f"  Min: {np.min(loo_global):.3f}")
    print(f"  Median: {np.median(loo_global):.3f}")
    print(f"  Max: {np.max(loo_global):.3f}")
    print(f"  Std: {np.std(loo_global):.3f}")

    # Find which lens removal causes biggest drop
    drops = original_global - loo_global
    max_drop_idx = np.argmax(drops)
    max_drop = drops[max_drop_idx]

    print(f"\n  Biggest drop: {max_drop:.3f} (removing {lenses[max_drop_idx].lens_id})")

    # Dominance alarm: drop > 0.5
    dominance_alarm = max_drop > 0.5
    print(f"  DOMINANCE ALARM (drop > 0.5): {'YES' if dominance_alarm else 'NO'}")

    # Jackknife estimate of mean(Z_corr) and SE
    jackknife_means = []
    for i in range(n):
        z_without_i = np.delete(z_corr, i)
        jackknife_means.append(np.mean(z_without_i))

    jackknife_means = np.array(jackknife_means)
    jackknife_mean = np.mean(jackknife_means)
    jackknife_se = np.sqrt((n - 1) / n * np.sum((jackknife_means - jackknife_mean)**2))

    print(f"\nJackknife estimate of mean(Z_corr):")
    print(f"  Mean: {jackknife_mean:.4f}")
    print(f"  SE: {jackknife_se:.4f}")
    print(f"  95% CI: [{jackknife_mean - 1.96*jackknife_se:.4f}, {jackknife_mean + 1.96*jackknife_se:.4f}]")

    # Plot LOO distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(loo_global, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(original_global, color='red', linestyle='-', linewidth=2, label=f'Original = {original_global:.3f}')
    ax.axvline(3.0, color='green', linestyle='--', label='3σ threshold')
    ax.set_xlabel('Global Z_corr (leave-one-out)')
    ax.set_ylabel('Count')
    ax.set_title('Leave-One-Out Global Z_corr Distribution')
    ax.legend()

    ax = axes[1]
    ax.scatter(range(n), loo_global, c='blue', alpha=0.6)
    ax.axhline(original_global, color='red', linestyle='-', label='Original')
    ax.axhline(3.0, color='green', linestyle='--', label='3σ threshold')
    for i in np.where(drops > 0.3)[0]:
        ax.annotate(lenses[i].lens_id[-8:], (i, loo_global[i]), fontsize=7, rotation=45)
    ax.set_xlabel('Lens index')
    ax.set_ylabel('Global Z_corr')
    ax.set_title('LOO Z_corr by lens')
    ax.legend()

    plt.tight_layout()
    plt.savefig(plot_dir / 'b_leave_one_out.png', dpi=150)
    plt.close()

    return {
        'loo_min': float(np.min(loo_global)),
        'loo_median': float(np.median(loo_global)),
        'loo_max': float(np.max(loo_global)),
        'loo_std': float(np.std(loo_global)),
        'max_drop_lens': lenses[max_drop_idx].lens_id,
        'max_drop_value': float(max_drop),
        'dominance_alarm': dominance_alarm,
        'jackknife_mean': float(jackknife_mean),
        'jackknife_se': float(jackknife_se),
    }


def _summarize_exclusions(exclusions: List[Tuple[str, str]]) -> str:
    if not exclusions:
        return "None"
    header = ["Lens", "Reason"]
    rows = [header] + [[lens, reason] for lens, reason in exclusions]
    col_widths = [max(len(str(row[i])) for row in rows) for i in range(2)]
    lines = [" | ".join(cell.ljust(col_widths[i]) for i, cell in enumerate(row)) for row in rows]
    return "\n".join(lines)


def _enforce_usage(expected: Sequence[str], seen: Sequence[str], exclusions: List[Tuple[str, str]], allow_reduction: bool) -> None:
    missing = set(expected) - set(seen)
    if missing and not allow_reduction:
        table = _summarize_exclusions(exclusions)
        raise RuntimeError(f"Excluded lenses without allow_reduction flag:\n{table}")
    if missing:
        print("\nExcluded lenses:")
        print(_summarize_exclusions(exclusions))


def section_c_null_adequacy(
    lenses: List[LensData],
    data_root: Path,
    cache_dir: Path,
    plot_dir: Path,
    cfg: FieldStudyConfig,
) -> Dict:
    """Section C: Null adequacy / false positive rate tests."""
    print("\n" + "="*60)
    print("SECTION C: Null Adequacy / False Positive Rate Tests")
    print("="*60)
    theta_bins = np.asarray(cfg.theta_bin_edges())

    all_lenses_discovered = discover_lenses(data_root, subset=None, score_bins={'M25', 'S10', 'S11', 'S12'})
    lens_map = {l.lens_id: l for l in all_lenses_discovered}

    contexts: List[LensNullContext] = []
    exclusions: List[Tuple[str, str]] = []

    per_lens_draws = cfg.per_lens_null_B or cfg.null_draws
    hp_cfg = HighpassConfig(m_cut=cfg.m_cut, m_max=cfg.fourier_m_max)

    for lens_data in lenses:
        lens_entry = lens_map.get(lens_data.lens_id)
        if lens_entry is None:
            exclusions.append((lens_data.lens_id, "lens_not_found"))
            continue

        band = lens_data.band
        products = list_band_products(lens_entry.path, band)
        data_path = products.get("data")
        if data_path is None:
            exclusions.append((lens_data.lens_id, "missing_data"))
            continue

        try:
            data, header = read_fits(data_path)
            noise_path, _ = choose_noise_psf(products)

            if noise_path is None:
                from cowls_field_study.run import _robust_std
                noise = np.full_like(data, _robust_std(data))
            else:
                noise = read_fits(noise_path)[0]

            preprocess_path = cache_dir / lens_data.lens_id / f"{band}_preprocess.npz"
            pre = load_preprocess(preprocess_path)
            if pre is None:
                exclusions.append((lens_data.lens_id, "missing_preprocess"))
                continue

            residual_products, mode_label = choose_residual_product(products, prefer_model=True)
            from cowls_field_study.run import _construct_residual
            residual, _ = _construct_residual(residual_products, data)

            s_theta, _ = build_window(pre.arc_mask, data=data, noise=noise, center=pre.center, theta_bins=theta_bins)
            profile = compute_ring_profile(residual, noise, pre.arc_mask, pre.center, theta_bins, window_weights=s_theta)
            valid_mask = np.isfinite(profile.r_theta) & (profile.coverage > 0)

            residual_samples = (residual / (noise + 1e-6))[pre.arc_mask]

            ctx = LensNullContext(
                lens_id=lens_data.lens_id,
                band=band,
                profile=profile,
                residual_samples=residual_samples,
                valid_mask=valid_mask,
            )

            for method in ("resample", "shift"):
                null_vals, mean, std, mean_hp, std_hp = prepare_null_baseline(
                    profile=profile,
                    residual_samples=residual_samples,
                    method=method,
                    lag_max=cfg.lag_max,
                    hf_fraction=cfg.hf_fraction,
                    draws=per_lens_draws,
                    default_draws=FieldStudyConfig().null_draws,
                    allow_reduction=cfg.allow_reduction,
                    highpass_config=hp_cfg,
                    valid_mask=valid_mask,
                )
                ctx.null_means[method] = mean
                ctx.null_stds[method] = std
                if mean_hp is not None:
                    ctx.null_hp_means[method] = mean_hp
                    ctx.null_hp_stds[method] = std_hp
            contexts.append(ctx)
        except Exception as exc:
            exclusions.append((lens_data.lens_id, f"error:{exc}"))

    _enforce_usage([l.lens_id for l in lenses], [c.lens_id for c in contexts], exclusions, cfg.allow_reduction)

    observed_global_z = np.mean([l.z_corr for l in lenses]) * np.sqrt(len(lenses))

    print(f"\nRunning global-null draws: resample={cfg.G_resample_global}, shift={cfg.G_shift_global}")
    synth_global_z_resample, synth_global_z_resample_hp = build_global_null_distribution(
        contexts=contexts,
        method="resample",
        lag_max=cfg.lag_max,
        hf_fraction=cfg.hf_fraction,
        draws=cfg.G_resample_global,
        default_draws=FieldStudyConfig().G_resample_global,
        allow_reduction=cfg.allow_reduction,
        max_workers=cfg.max_workers,
        highpass_config=hp_cfg,
    )
    synth_global_z_shift, synth_global_z_shift_hp = build_global_null_distribution(
        contexts=contexts,
        method="shift",
        lag_max=cfg.lag_max,
        hf_fraction=cfg.hf_fraction,
        draws=cfg.G_shift_global,
        default_draws=FieldStudyConfig().G_shift_global,
        allow_reduction=cfg.allow_reduction,
        max_workers=cfg.max_workers,
        highpass_config=hp_cfg,
    )

    p_resample = empirical_p_value(synth_global_z_resample, observed_global_z)
    p_shift = empirical_p_value(synth_global_z_shift, observed_global_z)
    count_resample = int(np.sum(synth_global_z_resample >= observed_global_z))
    count_shift = int(np.sum(synth_global_z_shift >= observed_global_z))
    hp_vals = [l.z_corr_hp for l in lenses if l.z_corr_hp is not None and np.isfinite(l.z_corr_hp)]
    if hp_vals:
        observed_global_z_hp = float(np.nanmean(hp_vals) * math.sqrt(len(hp_vals)))
    else:
        observed_global_z_hp = math.nan
    p_resample_hp = empirical_p_value(synth_global_z_resample_hp, observed_global_z_hp) if np.isfinite(observed_global_z_hp) else math.nan
    p_shift_hp = empirical_p_value(synth_global_z_shift_hp, observed_global_z_hp) if np.isfinite(observed_global_z_hp) else math.nan

    def _binomial_ci(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
        if n <= 0:
            return float("nan"), float("nan")
        lower, upper = stats.beta.interval(1 - alpha, k + 1, n - k + 1)
        return float(lower), float(upper)

    ci_resample = _binomial_ci(count_resample, len(synth_global_z_resample))
    ci_shift = _binomial_ci(count_shift, len(synth_global_z_shift))

    print(f"\nNull self-test results:")
    print(f"  Observed Global Z_corr: {observed_global_z:.3f}")
    if np.isfinite(observed_global_z_hp):
        print(f"  Observed Global Z_corr_hp: {observed_global_z_hp:.3f}")
    print(f"\n  Resample null ({len(synth_global_z_resample)} simulations):")
    print(f"    Mean synth Global Z: {np.nanmean(synth_global_z_resample):.3f}")
    print(f"    Std synth Global Z: {np.nanstd(synth_global_z_resample):.3f}")
    print(f"    p_empirical: {p_resample:.4f}")
    print(f"\n  Shift null ({len(synth_global_z_shift)} simulations):")
    print(f"    Mean synth Global Z: {np.nanmean(synth_global_z_shift):.3f}")
    print(f"    Std synth Global Z: {np.nanstd(synth_global_z_shift):.3f}")
    print(f"    p_empirical: {p_shift:.4f}")

    if np.isfinite(observed_global_z_hp):
        print(f"\n  Resample null hp (p={p_resample_hp:.4f}), Shift hp (p={p_shift_hp:.4f})")

    null_warning = (p_resample > 0.01) or (p_shift > 0.01)
    if np.isfinite(observed_global_z_hp):
        null_warning = null_warning or (p_resample_hp > 0.01) or (p_shift_hp > 0.01)
    if null_warning:
        print(f"\n  *** WARNING: p > 0.01 - significance may not be trustworthy! ***")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(synth_global_z_resample, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(observed_global_z, color='red', linestyle='-', linewidth=2, label=f'Observed = {observed_global_z:.2f}')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Global Z_corr (resample null)')
    ax.set_ylabel('Count')
    ax.set_title(f'Resample Null (p={p_resample:.3f})')
    ax.legend()

    ax = axes[1]
    ax.hist(synth_global_z_shift, bins=30, edgecolor='black', alpha=0.7)
    ax.axvline(observed_global_z, color='red', linestyle='-', linewidth=2, label=f'Observed = {observed_global_z:.2f}')
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Global Z_corr (shift null)')
    ax.set_ylabel('Count')
    ax.set_title(f'Shift Null (p={p_shift:.3f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(plot_dir / 'c_null_adequacy.png', dpi=150)
    plt.close()

    return {
        'observed_global_z': float(observed_global_z),
        'n_synth_resample': len(synth_global_z_resample),
        'mean_synth_resample': float(np.nanmean(synth_global_z_resample)),
        'std_synth_resample': float(np.nanstd(synth_global_z_resample)),
        'p_resample': float(p_resample),
        'p_resample_ci': ci_resample,
        'observed_global_z_hp': observed_global_z_hp,
        'p_resample_hp': p_resample_hp,
        'n_synth_shift': len(synth_global_z_shift),
        'mean_synth_shift': float(np.nanmean(synth_global_z_shift)),
        'std_synth_shift': float(np.nanstd(synth_global_z_shift)),
        'p_shift': float(p_shift),
        'p_shift_ci': ci_shift,
        'p_shift_hp': p_shift_hp,
        'null_warning': null_warning,
        'n_lenses_used': len(contexts),
        'excluded_lenses': exclusions,
    }


def section_d_artifact_proxies(lenses: List[LensData], data_root: Path, cache_dir: Path, plot_dir: Path) -> Dict:
    """Section D: Modeling-artifact hypothesis tests."""
    print("\n" + "="*60)
    print("SECTION D: Modeling-Artifact Hypothesis Tests")
    print("="*60)

    cfg = FieldStudyConfig()
    all_lenses_discovered = discover_lenses(data_root, subset=None, score_bins={'M25', 'S10', 'S11', 'S12'})
    lens_map = {l.lens_id: l for l in all_lenses_discovered}

    # Compute proxies for each lens
    proxies = {
        'lens_id': [],
        'z_corr': [],
        'z_pow': [],
        'score_bin': [],
        'residual_rms': [],
        'coverage': [],
        'texture_metric': [],
        'psf_fwhm': [],
    }

    for lens_data in lenses:
        lens_entry = lens_map.get(lens_data.lens_id)
        if lens_entry is None:
            continue

        band = lens_data.band
        products = list_band_products(lens_entry.path, band)
        data_path = products.get("data")
        if data_path is None:
            continue

        data, header = read_fits(data_path)
        noise_path, psf_path = choose_noise_psf(products)

        if noise_path is None:
            from cowls_field_study.run import _robust_std
            noise = np.full_like(data, _robust_std(data))
        else:
            noise = read_fits(noise_path)[0]

        # Load preprocessed data
        preprocess_path = cache_dir / lens_data.lens_id / f"{band}_preprocess.npz"
        pre = load_preprocess(preprocess_path)
        if pre is None:
            continue

        # Construct residual
        residual_products, _ = choose_residual_product(products, prefer_model=True)
        from cowls_field_study.run import _construct_residual
        residual, _ = _construct_residual(residual_products, data)

        # Proxy 1: Residual RMS in arc mask
        residual_normalized = residual / (noise + 1e-6)
        arc_residuals = residual_normalized[pre.arc_mask]
        residual_rms = np.std(arc_residuals) if len(arc_residuals) > 0 else np.nan

        # Proxy 2: Coverage (fraction of arc mask that's valid)
        coverage = np.sum(pre.arc_mask) / pre.arc_mask.size

        # Proxy 3: Texture metric (gradient variance in data within arc)
        grad_y, grad_x = np.gradient(data)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        arc_grad = grad_mag[pre.arc_mask]
        texture_metric = np.var(arc_grad) if len(arc_grad) > 0 else np.nan

        # Proxy 4: PSF FWHM (if available)
        psf_fwhm = np.nan
        if psf_path is not None:
            try:
                psf, _ = read_fits(psf_path)
                # Estimate FWHM from PSF
                psf_norm = psf / np.max(psf)
                half_max_mask = psf_norm >= 0.5
                fwhm_pixels = 2 * np.sqrt(np.sum(half_max_mask) / np.pi)
                psf_fwhm = fwhm_pixels
            except:
                pass

        proxies['lens_id'].append(lens_data.lens_id)
        proxies['z_corr'].append(lens_data.z_corr)
        proxies['z_pow'].append(lens_data.z_pow)
        proxies['score_bin'].append(lens_data.score_bin)
        proxies['residual_rms'].append(residual_rms)
        proxies['coverage'].append(coverage)
        proxies['texture_metric'].append(texture_metric)
        proxies['psf_fwhm'].append(psf_fwhm)

    # Convert to arrays
    z_corr = np.array(proxies['z_corr'])
    z_pow = np.array(proxies['z_pow'])
    residual_rms = np.array(proxies['residual_rms'])
    coverage = np.array(proxies['coverage'])
    texture = np.array(proxies['texture_metric'])
    psf_fwhm = np.array(proxies['psf_fwhm'])

    # Compute correlations
    correlations = {}

    for name, values in [('residual_rms', residual_rms), ('coverage', coverage),
                         ('texture', texture), ('psf_fwhm', psf_fwhm)]:
        valid = ~np.isnan(values) & ~np.isnan(z_corr)
        if np.sum(valid) > 3:
            r_pearson, p_pearson = stats.pearsonr(z_corr[valid], values[valid])
            r_spearman, p_spearman = stats.spearmanr(z_corr[valid], values[valid])
            correlations[name] = {
                'pearson_r': float(r_pearson),
                'pearson_p': float(p_pearson),
                'spearman_r': float(r_spearman),
                'spearman_p': float(p_spearman),
                'n_valid': int(np.sum(valid)),
            }
            print(f"\n{name} vs Z_corr:")
            print(f"  Pearson r = {r_pearson:.3f} (p = {p_pearson:.4f})")
            print(f"  Spearman ρ = {r_spearman:.3f} (p = {p_spearman:.4f})")
        else:
            correlations[name] = {'n_valid': 0}

    # Score bin analysis
    score_bins = np.array(proxies['score_bin'])
    print(f"\nZ_corr by score bin:")
    for sb in ['M25', 'S10', 'S11', 'S12']:
        mask = score_bins == sb
        if np.sum(mask) > 0:
            print(f"  {sb}: n={np.sum(mask)}, mean Z_corr = {np.mean(z_corr[mask]):.3f}, std = {np.std(z_corr[mask]):.3f}")

    # Plot correlations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    proxy_names = ['residual_rms', 'coverage', 'texture', 'psf_fwhm']
    proxy_values = [residual_rms, coverage, texture, psf_fwhm]
    proxy_labels = ['Residual RMS', 'Arc Coverage', 'Texture Metric', 'PSF FWHM']

    for ax, name, values, label in zip(axes.flat, proxy_names, proxy_values, proxy_labels):
        valid = ~np.isnan(values) & ~np.isnan(z_corr)
        if np.sum(valid) > 0:
            colors = [{'M25': 'blue', 'S10': 'green', 'S11': 'orange', 'S12': 'red'}.get(sb, 'gray')
                      for sb in score_bins[valid]]
            ax.scatter(values[valid], z_corr[valid], c=colors, alpha=0.6)

            # Add regression line
            if np.sum(valid) > 2:
                slope, intercept, r, p, se = stats.linregress(values[valid], z_corr[valid])
                x_line = np.linspace(np.min(values[valid]), np.max(values[valid]), 100)
                ax.plot(x_line, slope * x_line + intercept, 'r--', alpha=0.5,
                        label=f'r={r:.2f}, p={p:.3f}')
                ax.legend()

        ax.set_xlabel(label)
        ax.set_ylabel('Z_corr')
        ax.axhline(0, color='gray', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_dir / 'd_artifact_proxies.png', dpi=150)
    plt.close()

    # Find strongest correlation
    strongest = None
    strongest_r = 0
    for name, corr in correlations.items():
        if corr.get('n_valid', 0) > 0:
            r = abs(corr.get('spearman_r', 0))
            if r > abs(strongest_r):
                strongest_r = corr.get('spearman_r', 0)
                strongest = name

    return {
        'correlations': correlations,
        'strongest_proxy': strongest,
        'strongest_rho': float(strongest_r) if strongest else None,
    }


def section_e_frequency_structure(lenses: List[LensData], data_root: Path, cache_dir: Path, plot_dir: Path, cfg: FieldStudyConfig) -> Dict:
    """Section E: Frequency-structure consistency."""
    print("\n" + "="*60)
    print("SECTION E: Frequency-Structure Consistency")
    print("="*60)

    theta_bins = np.asarray(cfg.theta_bin_edges())

    all_lenses_discovered = discover_lenses(data_root, subset=None, score_bins={'M25', 'S10', 'S11', 'S12'})
    lens_map = {l.lens_id: l for l in all_lenses_discovered}

    # Compute low-m dominance for each lens
    low_m_dominance = []
    profiles_data = []
    exclusions: List[Tuple[str, str]] = []

    for lens_data in lenses:
        lens_entry = lens_map.get(lens_data.lens_id)
        if lens_entry is None:
            low_m_dominance.append(np.nan)
            continue

        band = lens_data.band
        products = list_band_products(lens_entry.path, band)
        data_path = products.get("data")
        if data_path is None:
            low_m_dominance.append(np.nan)
            continue

        try:
            data, header = read_fits(data_path)
            noise_path, _ = choose_noise_psf(products)

            if noise_path is None:
                from cowls_field_study.run import _robust_std
                noise = np.full_like(data, _robust_std(data))
            else:
                noise = read_fits(noise_path)[0]

            preprocess_path = cache_dir / lens_data.lens_id / f"{band}_preprocess.npz"
            pre = load_preprocess(preprocess_path)
            if pre is None:
                exclusions.append((lens_data.lens_id, "missing_preprocess"))
                low_m_dominance.append(np.nan)
                continue

            residual_products, _ = choose_residual_product(products, prefer_model=True)
            from cowls_field_study.run import _construct_residual
            residual, _ = _construct_residual(residual_products, data)

            s_theta, _ = build_window(pre.arc_mask, data=data, noise=noise, center=pre.center, theta_bins=theta_bins)
            profile = compute_ring_profile(residual, noise, pre.arc_mask, pre.center, theta_bins, window_weights=s_theta)

            low_m_ratio = compute_low_m_ratio(profile)
            low_m_dominance.append(low_m_ratio)

            fft = np.fft.fft(profile.r_theta) if len(profile.r_theta) else np.array([])
            power = np.abs(fft) ** 2 if fft.size else np.array([])

            profiles_data.append({
                'lens_id': lens_data.lens_id,
                'z_corr': lens_data.z_corr,
                'z_pow': lens_data.z_pow,
                'r_theta': profile.r_theta,
                'power': power,
                'low_m_ratio': low_m_ratio,
            })
        except Exception as exc:
            exclusions.append((lens_data.lens_id, f"error:{exc}"))
            low_m_dominance.append(np.nan)

    low_m_dominance = np.array(low_m_dominance)
    z_corr = np.array([l.z_corr for l in lenses])
    z_pow = np.array([l.z_pow for l in lenses])
    z_corr_hp_full = np.array([l.z_corr_hp if l.z_corr_hp is not None else np.nan for l in lenses])

    valid = ~np.isnan(low_m_dominance)
    if np.sum(valid) > 3:
        r_low_m, p_low_m = stats.spearmanr(z_corr[valid], low_m_dominance[valid])
        pearson_r, pearson_p = stats.pearsonr(z_corr[valid], low_m_dominance[valid])
        print(f"\nLow-m (m<=3) dominance vs Z_corr:")
        print(f"  Spearman ρ = {r_low_m:.3f} (p = {p_low_m:.4f})")
        print(f"  Pearson r = {pearson_r:.3f} (p = {pearson_p:.4f})")
    else:
        r_low_m, p_low_m = np.nan, np.nan
        pearson_r, pearson_p = np.nan, np.nan

    r_low_m_hp = np.nan
    p_low_m_hp = np.nan
    if np.sum(valid & np.isfinite(z_corr_hp_full)) > 3:
        r_low_m_hp, p_low_m_hp = stats.spearmanr(z_corr_hp_full[valid], low_m_dominance[valid])
        print(f"  Spearman (hp) ρ = {r_low_m_hp:.3f} (p = {p_low_m_hp:.4f})")

    # Identify high Z_corr / negative Z_pow lenses
    high_corr_low_pow = [(l, lmd) for l, lmd in zip(lenses, low_m_dominance)
                         if l.z_corr > 0.9 and l.z_pow < 0]

    print(f"\nLenses with Z_corr > 0.9 and Z_pow < 0:")
    for l, lmd in high_corr_low_pow[:5]:
        print(f"  {l.lens_id}: Z_corr={l.z_corr:.2f}, Z_pow={l.z_pow:.2f}, low_m_ratio={lmd:.3f}")

    if exclusions:
        print("\nLow-m computation exclusions:")
        print(_summarize_exclusions(exclusions))

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Z_corr vs Z_pow scatter
    ax = axes[0, 0]
    ax.scatter(z_corr, z_pow, c=low_m_dominance, cmap='viridis', alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Z_corr')
    ax.set_ylabel('Z_pow')
    ax.set_title('Z_corr vs Z_pow (color = low-m ratio)')
    plt.colorbar(ax.collections[0], ax=ax, label='Low-m ratio')

    # Low-m dominance vs Z_corr
    ax = axes[0, 1]
    valid = ~np.isnan(low_m_dominance)
    ax.scatter(low_m_dominance[valid], z_corr[valid], alpha=0.7)
    ax.set_xlabel('Low-m (m≤3) Power Ratio')
    ax.set_ylabel('Z_corr')
    ax.set_title(f'Low-m Dominance vs Z_corr (ρ={r_low_m:.2f})')

    # Example r(θ) profiles for high Z_corr lenses
    ax = axes[1, 0]
    for pd in sorted(profiles_data, key=lambda x: -x['z_corr'])[:5]:
        theta_centers = 0.5 * (theta_bins[:-1] + theta_bins[1:])
        if len(pd['r_theta']) == len(theta_centers):
            ax.plot(theta_centers, pd['r_theta'], alpha=0.7,
                    label=f"{pd['lens_id'][-8:]}: Z={pd['z_corr']:.2f}")
    ax.set_xlabel('θ (deg)')
    ax.set_ylabel('r(θ)')
    ax.set_title('Ring profiles for top 5 Z_corr lenses')
    ax.legend(fontsize=8)

    # Power spectrum for a high Z_corr lens
    ax = axes[1, 1]
    if profiles_data:
        top_lens = max(profiles_data, key=lambda x: x['z_corr'])
        power = top_lens['power']
        n = len(power)
        ax.semilogy(range(n//2), power[:n//2], 'b-', alpha=0.7)
        ax.axvline(3, color='red', linestyle='--', label='m=3 cutoff')
        ax.set_xlabel('Mode m')
        ax.set_ylabel('Power')
        ax.set_title(f"Power spectrum: {top_lens['lens_id'][-12:]}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(plot_dir / 'e_frequency_structure.png', dpi=150)
    plt.close()

    return {
        'low_m_corr_rho': float(r_low_m) if not np.isnan(r_low_m) else None,
        'low_m_corr_p': float(p_low_m) if not np.isnan(p_low_m) else None,
        'low_m_corr_pearson': float(pearson_r) if not np.isnan(pearson_r) else None,
        'low_m_corr_pearson_p': float(pearson_p) if not np.isnan(pearson_p) else None,
        'low_m_corr_hp_rho': float(r_low_m_hp) if not np.isnan(r_low_m_hp) else None,
        'low_m_corr_hp_p': float(p_low_m_hp) if not np.isnan(p_low_m_hp) else None,
        'mean_low_m_ratio': float(np.nanmean(low_m_dominance)),
        'n_high_corr_neg_pow': len(high_corr_low_pow),
        'excluded': exclusions,
    }


def section_f_band_consistency(lenses: List[LensData], data_root: Path, cache_dir: Path, plot_dir: Path, cfg: FieldStudyConfig) -> Dict:
    """Section F: Band-consistency checks."""
    print("\n" + "="*60)
    print("SECTION F: Band-Consistency Checks")
    print("="*60)

    all_lenses_discovered = discover_lenses(data_root, subset=None, score_bins={'M25', 'S10', 'S11', 'S12'})
    lens_map = {l.lens_id: l for l in all_lenses_discovered}

    multi_band_lenses = []
    for lens_data in lenses:
        lens_entry = lens_map.get(lens_data.lens_id)
        if lens_entry is None:
            continue
        if len(lens_entry.bands) > 1:
            multi_band_lenses.append((lens_data, lens_entry))

    print(f"\nLenses with multiple bands: {len(multi_band_lenses)}")

    if len(multi_band_lenses) == 0:
        print("  No multi-band lenses available for consistency check.")
        return {'n_multi_band': 0, 'band_consistency': None}

    band_results = []

    theta_bins = np.asarray(cfg.theta_bin_edges())
    hp_cfg = HighpassConfig(m_cut=cfg.m_cut, m_max=cfg.fourier_m_max)
    mask_mode = (cfg.band_common_mask or "all_bands").lower()
    band_pair = [b.strip() for b in cfg.band_pair.split(",")] if cfg.band_pair else []

    exclusions: List[Tuple[str, str]] = []

    for lens_data, lens_entry in multi_band_lenses:
        print(f"\n  {lens_data.lens_id}: bands = {lens_entry.bands}")

        band_profiles = []

        for band in lens_entry.bands:
            products = list_band_products(lens_entry.path, band)
            data_path = products.get("data")
            if data_path is None:
                exclusions.append((lens_data.lens_id, f"{band}:missing_data"))
                continue

            try:
                data, header = read_fits(data_path)
                noise_path, _ = choose_noise_psf(products)

                if noise_path is None:
                    from cowls_field_study.run import _robust_std
                    noise = np.full_like(data, _robust_std(data))
                else:
                    noise = read_fits(noise_path)[0]

                preprocess_path = cache_dir / lens_data.lens_id / f"{band}_preprocess.npz"
                pre = load_preprocess(preprocess_path)

                if pre is None:
                    meta = {k.lower(): header.get(k) for k in ("X_CENTER", "Y_CENTER", "THETA_E", "R_EIN")} if header else {}
                    pre = build_arc_mask(
                        image=data,
                        noise=noise,
                        mode='model_residual',
                        metadata=meta,
                        snr_threshold=cfg.snr_threshold,
                        annulus_width=cfg.annulus_width,
                    )

                residual_products, _ = choose_residual_product(products, prefer_model=True)
                from cowls_field_study.run import _construct_residual
                residual, _ = _construct_residual(residual_products, data)

                s_theta, _ = build_window(pre.arc_mask, data=data, noise=noise, center=pre.center, theta_bins=theta_bins)
                profile = compute_ring_profile(residual, noise, pre.arc_mask, pre.center, theta_bins, window_weights=s_theta)
                valid_mask = np.isfinite(profile.r_theta) & (profile.coverage > 0)

                residual_samples = (residual / (noise + 1e-6))[pre.arc_mask]
                band_profiles.append(
                    {
                        "band": band,
                        "profile": profile,
                        "residual_samples": residual_samples,
                        "valid_mask": valid_mask,
                    }
                )
            except Exception as e:
                exclusions.append((lens_data.lens_id, f"{band}:error:{e}"))

        if len(band_profiles) < 2:
            continue

        if mask_mode == "pairwise" and len(band_pair) == 2:
            selected = [bp for bp in band_profiles if bp["band"] in band_pair]
        else:
            selected = band_profiles

        if len(selected) < 2:
            exclusions.append((lens_data.lens_id, "insufficient_band_pair"))
            continue

        common_mask = np.logical_and.reduce([bp["valid_mask"] for bp in selected])
        if np.count_nonzero(common_mask) < 2:
            exclusions.append((lens_data.lens_id, "insufficient_common_theta"))
            continue

        lens_band_z = {'lens_id': lens_data.lens_id}

        for bp in selected:
            masked_profile = mask_ring_profile(bp["profile"], common_mask)
            field_stats = compute_field_stats(
                masked_profile,
                lag_max=cfg.lag_max,
                hf_fraction=cfg.hf_fraction,
                highpass_config=hp_cfg,
                highpass_mask=common_mask,
            )
            nulls = draw_null_statistics(
                masked_profile,
                mode='both',
                residual_samples=bp["residual_samples"],
                lag_max=cfg.lag_max,
                hf_fraction=cfg.hf_fraction,
                draws=cfg.null_draws,
                highpass_config=hp_cfg,
                valid_mask=common_mask,
            )

            z_corr = (field_stats.t_corr - np.nanmean(nulls.t_corr)) / (np.nanstd(nulls.t_corr) + 1e-9)
            z_pow = (field_stats.t_pow - np.nanmean(nulls.t_pow)) / (np.nanstd(nulls.t_pow) + 1e-9)

            if nulls.t_corr_hp is not None and field_stats.t_corr_hp is not None:
                z_corr_hp = (field_stats.t_corr_hp - np.nanmean(nulls.t_corr_hp)) / (np.nanstd(nulls.t_corr_hp) + 1e-9)
                z_pow_hp = (field_stats.t_pow_hp - np.nanmean(nulls.t_pow_hp)) / (np.nanstd(nulls.t_pow_hp) + 1e-9)
            else:
                z_corr_hp = float('nan')
                z_pow_hp = float('nan')

            lens_band_z[bp["band"]] = {'z_corr': float(z_corr), 'z_pow': float(z_pow), 'z_corr_hp': float(z_corr_hp), 'z_pow_hp': float(z_pow_hp)}
            print(f"    {bp['band']}: Z_corr={z_corr:.2f}, Z_pow={z_pow:.2f}, Z_corr_hp={z_corr_hp:.2f}")

        if len(lens_band_z) > 2:
            band_results.append(lens_band_z)

    sign_consistent = 0
    sign_consistent_hp = 0
    per_lens_variance = []
    per_lens_variance_hp = []
    f150 = []
    f277 = []
    f150_hp = []
    f277_hp = []

    for br in band_results:
        bands = [k for k in br if k != 'lens_id']
        z_vals = [br[b]['z_corr'] for b in bands if np.isfinite(br[b]['z_corr'])]
        z_vals_hp = [br[b]['z_corr_hp'] for b in bands if np.isfinite(br[b]['z_corr_hp'])]
        signs = [np.sign(v) for v in z_vals]
        signs_hp = [np.sign(v) for v in z_vals_hp]
        if z_vals and len(set(signs)) == 1:
            sign_consistent += 1
        if z_vals_hp and len(set(signs_hp)) == 1:
            sign_consistent_hp += 1
        if z_vals:
            per_lens_variance.append(np.var(z_vals))
        if z_vals_hp:
            per_lens_variance_hp.append(np.var(z_vals_hp))
        if 'F150W' in br and 'F277W' in br:
            f150.append(br['F150W']['z_corr'])
            f277.append(br['F277W']['z_corr'])
            f150_hp.append(br['F150W']['z_corr_hp'])
            f277_hp.append(br['F277W']['z_corr_hp'])

    consistency_frac = sign_consistent / len(band_results) if band_results else float('nan')
    consistency_frac_hp = sign_consistent_hp / len(band_results) if band_results else float('nan')
    band_var_mean = float(np.nanmean(per_lens_variance)) if per_lens_variance else float('nan')
    band_var_mean_hp = float(np.nanmean(per_lens_variance_hp)) if per_lens_variance_hp else float('nan')
    band_corr = float('nan')
    band_corr_hp = float('nan')
    if len(f150) > 1:
        try:
            band_corr = stats.pearsonr(f150, f277)[0]
        except ValueError:
            band_corr = float('nan')
    if len(f150_hp) > 1:
        try:
            band_corr_hp = stats.pearsonr(f150_hp, f277_hp)[0]
        except ValueError:
            band_corr_hp = float('nan')

    print(f"\n  Sign consistency across bands (base): {sign_consistent}/{len(band_results)} = {consistency_frac:.2f}")
    print(f"  Sign consistency across bands (hp): {sign_consistent_hp}/{len(band_results)} = {consistency_frac_hp:.2f}")
    print(f"  Mean per-lens Z_corr variance across bands: {band_var_mean:.3f}")
    print(f"  Mean per-lens Z_corr_hp variance across bands: {band_var_mean_hp:.3f}")
    if not np.isnan(band_corr):
        print(f"  F150W vs F277W correlation: {band_corr:.3f}")
    if not np.isnan(band_corr_hp):
        print(f"  F150W vs F277W correlation (hp): {band_corr_hp:.3f}")
    if exclusions:
        print("\nBand consistency exclusions:")
        print(_summarize_exclusions(exclusions))

    return {
        'n_multi_band': len(multi_band_lenses),
        'n_tested': len(band_results),
        'band_consistency': float(consistency_frac) if not np.isnan(consistency_frac) else None,
        'band_consistency_hp': float(consistency_frac_hp) if not np.isnan(consistency_frac_hp) else None,
        'band_variance_mean': band_var_mean,
        'band_variance_mean_hp': band_var_mean_hp,
        'band_corr_f150_f277': band_corr if not np.isnan(band_corr) else None,
        'band_corr_f150_f277_hp': band_corr_hp if not np.isnan(band_corr_hp) else None,
        'band_results': band_results,
        'excluded': exclusions,
    }


def write_kill_report(results: Dict, plot_dir: Path, output_path: Path):
    """Write the KILL_REPORT.md."""

    # Determine verdict
    verdict_points = 0
    kill_conditions = []

    # Check null adequacy
    if results.get('section_c', {}).get('null_warning', False):
        kill_conditions.append("TRIGGERED: Null p-value > 0.01 - significance not trustworthy")
    else:
        verdict_points += 1
        kill_conditions.append("NOT TRIGGERED: Null p-values < 0.01")

    # Check dominance
    if results.get('section_b', {}).get('dominance_alarm', False):
        kill_conditions.append("TRIGGERED: Single lens dominates signal (drop > 0.5σ)")
    else:
        verdict_points += 1
        kill_conditions.append("NOT TRIGGERED: No single lens dominates")

    # Check LOO stability
    loo_min = results.get('section_b', {}).get('loo_min', 0)
    if loo_min < 3.0:
        kill_conditions.append(f"TRIGGERED: LOO min ({loo_min:.2f}) falls below 3σ")
    else:
        verdict_points += 1
        kill_conditions.append(f"NOT TRIGGERED: LOO min ({loo_min:.2f}) stays above 3σ")

    # Check proxy correlations
    strongest_proxy = results.get('section_d', {}).get('strongest_proxy')
    strongest_rho = results.get('section_d', {}).get('strongest_rho', 0)
    if strongest_rho and abs(strongest_rho) > 0.5:
        kill_conditions.append(f"TRIGGERED: Strong proxy correlation ({strongest_proxy}: ρ={strongest_rho:.2f})")
    else:
        verdict_points += 1
        kill_conditions.append(f"NOT TRIGGERED: Weak proxy correlations (max |ρ| = {abs(strongest_rho) if strongest_rho else 0:.2f})")

    # Low-m dominance
    low_m_rho = results.get('section_e', {}).get('low_m_corr_rho')
    if low_m_rho and abs(low_m_rho) > 0.6:
        kill_conditions.append(f"TRIGGERED: Low-m dominance correlates with Z_corr (ρ={low_m_rho:.2f})")
    else:
        verdict_points += 1
        kill_conditions.append(f"NOT TRIGGERED: Low-m dominance uncorrelated (ρ={low_m_rho if low_m_rho else 0:.2f})")

    # Overall verdict
    if verdict_points >= 4:
        verdict = "SIGNAL SURVIVES - No kill conditions triggered"
        verdict_prob = "High confidence (4-5/5 checks passed)"
    elif verdict_points >= 3:
        verdict = "SIGNAL WEAKENED - Some concerns but not killed"
        verdict_prob = "Moderate confidence (3/5 checks passed)"
    else:
        verdict = "SIGNAL KILLED - Multiple kill conditions triggered"
        verdict_prob = "Low confidence (≤2/5 checks passed)"

    sec_a = results['section_a']
    fmt = lambda v: f"{v:.3f}" if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)) else "N/A"

    # Build report
    lines = [
        "# COWLS Field-Level Detection: Kill Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"**VERDICT: {verdict}**",
        f"**Confidence: {verdict_prob}**",
        "",
        f"- Original Global Z_corr: {sec_a['global_z_corr']:.3f}",
        f"- Original Global Z_corr_hp: {fmt(sec_a.get('global_z_corr_hp'))}",
        f"- Original Global Z_pow: {sec_a['global_z_pow']:.3f}",
        f"- Number of lenses: {sec_a['n_lenses']}",
        "",
        "---",
        "",
        "## Kill Condition Checklist",
        "",
    ]

    for kc in kill_conditions:
        status = "❌" if "TRIGGERED" in kc and "NOT TRIGGERED" not in kc else "✓"
        lines.append(f"- {status} {kc}")

    lines.extend([
        "",
        "---",
        "",
        "## Section A: Headline Numbers & Dominance",
        "",
        "### Reproduced Statistics",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| n_lenses | {sec_a['n_lenses']} |",
        f"| mean(Z_corr) | {sec_a['mean_z_corr']:.4f} |",
        f"| mean(Z_corr_hp) | {fmt(sec_a.get('mean_z_corr_hp'))} |",
        f"| Global Z_corr | {sec_a['global_z_corr']:.3f} |",
        f"| Global Z_corr_hp | {fmt(sec_a.get('global_z_corr_hp'))} |",
        f"| Z_corr min/median/max | {sec_a['z_corr_min']:.2f} / {sec_a['z_corr_median']:.2f} / {sec_a['z_corr_max']:.2f} |",
        "",
        "### Dominance Analysis",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Top 1 lens contribution | {results['section_a']['dominance_top1']*100:.1f}% |",
        f"| Top 3 lens contribution | {results['section_a']['dominance_top3']*100:.1f}% |",
        f"| Top 5 lens contribution | {results['section_a']['dominance_top5']*100:.1f}% |",
        f"| Global Z without top 1 | {results['section_a']['global_without_top1']:.3f} |",
        f"| Global Z without top 3 | {results['section_a']['global_without_top3']:.3f} |",
        f"| Global Z without top 5 | {results['section_a']['global_without_top5']:.3f} |",
        "",
        "![Z_corr Distribution](kill_plots/a_zcorr_distribution.png)",
        "",
        "---",
        "",
        "## Section B: Leave-One-Out & Jackknife",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| LOO min | {results['section_b']['loo_min']:.3f} |",
        f"| LOO median | {results['section_b']['loo_median']:.3f} |",
        f"| LOO max | {results['section_b']['loo_max']:.3f} |",
        f"| Max drop lens | {results['section_b']['max_drop_lens']} |",
        f"| Max drop value | {results['section_b']['max_drop_value']:.3f} |",
        f"| Dominance alarm | {'YES' if results['section_b']['dominance_alarm'] else 'NO'} |",
        f"| Jackknife mean | {results['section_b']['jackknife_mean']:.4f} |",
        f"| Jackknife SE | {results['section_b']['jackknife_se']:.4f} |",
        "",
        "![Leave-One-Out](kill_plots/b_leave_one_out.png)",
        "",
        "---",
        "",
        "## Section C: Null Adequacy Tests",
        "",
    ]) 

    sec_c = results.get('section_c', {})
    if sec_c:
        lines.extend([
            f"| Null Method | G Used | Mean Synth Z | Std Synth Z | p_emp |",
            f"|-------------|--------|--------------|-------------|-------|",
            f"| Resample | {sec_c.get('n_synth_resample', 'N/A')} | {sec_c.get('mean_synth_resample', 'N/A')} | {sec_c.get('std_synth_resample', 'N/A')} | {sec_c.get('p_resample', 'N/A')} |",
            f"| Shift | {sec_c.get('n_synth_shift', 'N/A')} | {sec_c.get('mean_synth_shift', 'N/A')} | {sec_c.get('std_synth_shift', 'N/A')} | {sec_c.get('p_shift', 'N/A')} |",
            f"| Resample (hp) | {sec_c.get('n_synth_resample', 'N/A')} | — | — | {sec_c.get('p_resample_hp', 'N/A')} |",
            f"| Shift (hp) | {sec_c.get('n_synth_shift', 'N/A')} | — | — | {sec_c.get('p_shift_hp', 'N/A')} |",
            "",
            f"- Resample CI (95%): {sec_c.get('p_resample_ci')}",
            f"- Shift CI (95%): {sec_c.get('p_shift_ci')}",
            "",
            f"**Null Warning: {'YES - significance may not be trustworthy!' if sec_c.get('null_warning') else 'NO - nulls are well-calibrated'}**",
            "",
            f"- Observed Z_corr: {sec_c.get('observed_global_z', 'N/A')}",
            f"- Observed Z_corr_hp: {sec_c.get('observed_global_z_hp', 'N/A')}",
            f"- Lenses used: {sec_c.get('n_lenses_used', 'N/A')}",
            "",
            "Histograms:",
            "",
            "![Null Adequacy](kill_plots/c_null_adequacy.png)",
        ])
        excluded = sec_c.get('excluded_lenses') or []
        if excluded:
            lines.extend([
                "",
                "Excluded lenses (null adequacy):",
                "",
                "Lens | Reason",
                "--- | ---",
            ])
            for lens_id, reason in excluded:
                lines.append(f"{lens_id} | {reason}")

    lines.extend([
        "",
        "---",
        "",
        "## Section D: Artifact Proxy Correlations",
        "",
        "| Proxy | Spearman ρ | p-value | N |",
        "|-------|------------|---------|---|",
    ])

    for name, corr in results.get('section_d', {}).get('correlations', {}).items():
        if corr.get('n_valid', 0) > 0:
            lines.append(f"| {name} | {corr.get('spearman_r', 'N/A'):.3f} | {corr.get('spearman_p', 'N/A'):.4f} | {corr.get('n_valid')} |")

    lines.extend([
        "",
        f"**Strongest proxy: {results.get('section_d', {}).get('strongest_proxy') or 'None'} (ρ = {results.get('section_d', {}).get('strongest_rho') or 0:.3f})**",
        "",
        "![Artifact Proxies](kill_plots/d_artifact_proxies.png)",
        "",
        "---",
        "",
        "## Section E: Frequency Structure",
        "",
    ])

    sec_e = results.get('section_e', {})
    low_m_rho_val = sec_e.get('low_m_corr_rho')
    low_m_pearson_val = sec_e.get('low_m_corr_pearson')
    low_m_rho_hp_val = sec_e.get('low_m_corr_hp_rho')
    mean_low_m_val = sec_e.get('mean_low_m_ratio')
    low_m_str = f"{low_m_rho_val:.3f}" if low_m_rho_val is not None else "N/A"
    low_m_hp_str = f"{low_m_rho_hp_val:.3f}" if low_m_rho_hp_val is not None else "N/A"
    low_m_pearson_str = f"{low_m_pearson_val:.3f}" if low_m_pearson_val is not None else "N/A"
    mean_low_m_str = f"{mean_low_m_val:.3f}" if (mean_low_m_val is not None and not np.isnan(mean_low_m_val)) else "N/A"
    lines.extend([
        f"- Low-m (m≤3) dominance correlation with Z_corr: ρ = {low_m_str}",
        f"- Low-m (m≤3) dominance correlation with Z_corr_hp: ρ = {low_m_hp_str}",
        f"- Low-m (m≤3) Pearson correlation with Z_corr: r = {low_m_pearson_str}",
        f"- Mean low-m ratio: {mean_low_m_str}",
        f"- Lenses with high Z_corr but negative Z_pow: {sec_e.get('n_high_corr_neg_pow', 0)}",
        "",
        "![Frequency Structure](kill_plots/e_frequency_structure.png)",
        "",
        "---",
        "",
        "## Section F: Band Consistency",
        "",
    ])

    sec_f = results.get('section_f', {})
    lines.extend([
        f"- Multi-band lenses available: {sec_f.get('n_multi_band', 0)}",
        f"- Lenses tested: {sec_f.get('n_tested', 0)}",
        f"- Sign consistency: {sec_f.get('band_consistency', 'N/A')}",
        f"- Sign consistency (hp): {sec_f.get('band_consistency_hp', 'N/A')}",
        f"- Mean per-lens band variance: {sec_f.get('band_variance_mean', 'N/A')}",
        f"- Mean per-lens band variance (hp): {sec_f.get('band_variance_mean_hp', 'N/A')}",
        f"- F150W vs F277W correlation: {sec_f.get('band_corr_f150_f277', 'N/A')}",
        f"- F150W vs F277W correlation (hp): {sec_f.get('band_corr_f150_f277_hp', 'N/A')}",
        "",
        "---",
        "",
        "## Top 5 Mundane Explanations (Ranked by Evidence)",
        "",
    ])

    # Rank explanations by evidence
    explanations = []

    if sec_e.get('n_high_corr_neg_pow', 0) > 10:
        explanations.append(("Low-mode (m≤3) macro-model mismatch", 0.8,
            "High Z_corr but Z_pow~0 suggests large-scale smooth residuals, not localized structure"))

    if abs(results.get('section_d', {}).get('correlations', {}).get('texture', {}).get('spearman_r', 0)) > 0.3:
        explanations.append(("Source texture/complexity systematics", 0.6,
            "Z_corr correlates with texture metric, suggesting unmodeled source structure"))

    if abs(results.get('section_d', {}).get('correlations', {}).get('residual_rms', {}).get('spearman_r', 0)) > 0.3:
        explanations.append(("Model quality variation", 0.5,
            "Z_corr correlates with residual RMS, suggesting worse models produce higher Z"))

    if results['section_a']['dominance_top3'] > 0.3:
        explanations.append(("Outlier-driven signal", 0.4,
            f"Top 3 lenses contribute {results['section_a']['dominance_top3']*100:.0f}% of signal"))

    explanations.append(("Selection effects", 0.3,
        "Arc selection may preferentially select regions with correlated residuals"))

    explanations.sort(key=lambda x: -x[1])

    for i, (name, score, desc) in enumerate(explanations[:5], 1):
        lines.append(f"{i}. **{name}** (evidence score: {score:.1f})")
        lines.append(f"   - {desc}")
        lines.append("")

    lines.extend([
        "---",
        "",
        "## What Would Still Make It Pockets?",
        "",
        "For the signal to be attributed to dark substructure perturbations, we would need:",
        "",
        "1. **Mass-scale consistency**: Z_corr should correlate with predicted subhalo abundance for each lens",
        "2. **Spectral signature**: Power should peak at specific angular scales corresponding to subhalo masses",
        "3. **Band independence**: Same Z_corr in all bands (wavelength-independent)",
        "4. **Model improvement test**: Z_corr should decrease when using higher-fidelity lens models",
        "5. **Mock injection recovery**: Injected subhalos should produce similar Z_corr patterns",
        "",
        "---",
        "",
        "*Generated by kill_analysis.py*",
    ])

    output_path.write_text("\n".join(lines))
    print(f"\nWrote KILL_REPORT.md to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the COWLS kill analysis pipeline.")
    parser.add_argument("--report-path", type=Path, default=Path("results/cowls_field_study/full_M25_S10-S12/report.md"))
    parser.add_argument("--data-root", type=Path, default=FieldStudyConfig().data_root)
    parser.add_argument("--cache-dir", type=Path, default=Path("results/cowls_field_study/full_M25_S10-S12/cache"))
    parser.add_argument("--results-dir", type=Path, default=Path("results/cowls_field_study"))
    parser.add_argument("--G-resample-global", type=int, default=FieldStudyConfig().G_resample_global)
    parser.add_argument("--G-shift-global", type=int, default=FieldStudyConfig().G_shift_global)
    parser.add_argument("--null-B", dest="null_draws", type=int, default=FieldStudyConfig().null_draws)
    parser.add_argument("--allow-reduction", action="store_true", help="Allow using fewer draws or lenses without error.")
    parser.add_argument("--max-workers", type=int, default=None, help="Max workers for parallel null draws.")
    parser.add_argument("--m-cut", type=int, default=FieldStudyConfig().m_cut, help="Low-m cutoff for high-pass filter")
    parser.add_argument("--fourier-m-max", type=int, default=None, help="Maximum Fourier mode for high-pass fit")
    parser.add_argument("--band-common-mask", type=str, default="all_bands", choices=["all_bands", "pairwise"], help="Common-mask strategy for band consistency")
    parser.add_argument("--band-pair", type=str, default=FieldStudyConfig().band_pair, help="Band pair to test when band-common-mask=pairwise")
    return parser.parse_args()


def write_kill_summary(results: Dict, output_path: Path):
    """Write the kill_summary.json."""
    summary = {
        'global_z_corr': results['section_a']['global_z_corr'],
        'global_z_corr_hp': results['section_a'].get('global_z_corr_hp'),
        'global_z_pow': results['section_a']['global_z_pow'],
        'n_lenses': results['section_a']['n_lenses'],
        'empirical_p_null_resample': results.get('section_c', {}).get('p_resample'),
        'empirical_p_null_shift': results.get('section_c', {}).get('p_shift'),
        'p_emp_resample': results.get('section_c', {}).get('p_resample'),
        'p_emp_shift': results.get('section_c', {}).get('p_shift'),
        'p_emp_resample_hp': results.get('section_c', {}).get('p_resample_hp'),
        'p_emp_shift_hp': results.get('section_c', {}).get('p_shift_hp'),
        'G_resample': results.get('section_c', {}).get('n_synth_resample'),
        'G_shift': results.get('section_c', {}).get('n_synth_shift'),
        'n_lenses_used': results.get('section_c', {}).get('n_lenses_used'),
        'leave_one_out_min': results['section_b']['loo_min'],
        'leave_one_out_median': results['section_b']['loo_median'],
        'leave_one_out_max': results['section_b']['loo_max'],
        'dominance_top1': results['section_a']['dominance_top1'],
        'dominance_top3': results['section_a']['dominance_top3'],
        'dominance_top5': results['section_a']['dominance_top5'],
        'strongest_proxy': results.get('section_d', {}).get('strongest_proxy'),
        'strongest_proxy_rho': results.get('section_d', {}).get('strongest_rho'),
        'low_m_dominance_rho': results.get('section_e', {}).get('low_m_corr_rho'),
        'low_m_dominance_p': results.get('section_e', {}).get('low_m_corr_p'),
        'low_m_dominance_pearson_r': results.get('section_e', {}).get('low_m_corr_pearson'),
        'low_m_dominance_pearson_p': results.get('section_e', {}).get('low_m_corr_pearson_p'),
        'low_m_dominance_hp_rho': results.get('section_e', {}).get('low_m_corr_hp_rho'),
        'low_m_dominance_hp_p': results.get('section_e', {}).get('low_m_corr_hp_p'),
        'jackknife_mean': results['section_b']['jackknife_mean'],
        'jackknife_se': results['section_b']['jackknife_se'],
        'band_sign_consistency': results.get('section_f', {}).get('band_consistency'),
        'band_sign_consistency_hp': results.get('section_f', {}).get('band_consistency_hp'),
        'band_variance_mean': results.get('section_f', {}).get('band_variance_mean'),
        'band_variance_mean_hp': results.get('section_f', {}).get('band_variance_mean_hp'),
        'band_corr_f150_f277': results.get('section_f', {}).get('band_corr_f150_f277'),
        'band_corr_f150_f277_hp': results.get('section_f', {}).get('band_corr_f150_f277_hp'),
    }

    output_path.write_text(json.dumps(summary, indent=2))
    print(f"Wrote kill_summary.json to {output_path}")


def main():
    args = parse_args()
    cfg = FieldStudyConfig(
        data_root=args.data_root,
        results_root=args.results_dir,
        null_draws=args.null_draws,
        G_resample_global=args.G_resample_global,
        G_shift_global=args.G_shift_global,
        allow_reduction=args.allow_reduction,
        max_workers=args.max_workers,
        per_lens_null_B=args.null_draws,
        m_cut=args.m_cut,
        fourier_m_max=args.fourier_m_max,
        band_common_mask=args.band_common_mask,
        band_pair=args.band_pair,
    )
    # Paths
    project_root = Path(__file__).parent.parent
    report_path = project_root / args.report_path
    data_root = project_root / args.data_root
    cache_dir = project_root / args.cache_dir
    output_dir = project_root / args.results_dir
    plot_dir = output_dir / "kill_plots"

    # Create directories
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Parse existing results
    print("Loading results from report...")
    global_stats, lenses = parse_report(report_path)
    print(f"  Loaded {len(lenses)} lenses")

    results = {}

    # Section A
    results['section_a'] = section_a_reproduce_and_dominance(lenses, plot_dir)

    # Section B
    results['section_b'] = section_b_jackknife(lenses, plot_dir)

    # Section C (computationally intensive - reduce K if needed)
    results['section_c'] = section_c_null_adequacy(lenses, data_root, cache_dir, plot_dir, cfg)

    # Section D
    results['section_d'] = section_d_artifact_proxies(lenses, data_root, cache_dir, plot_dir)

    # Section E
    results['section_e'] = section_e_frequency_structure(lenses, data_root, cache_dir, plot_dir, cfg)

    # Section F
    results['section_f'] = section_f_band_consistency(lenses, data_root, cache_dir, plot_dir, cfg)

    # Write outputs
    write_kill_report(results, plot_dir, output_dir / "KILL_REPORT.md")
    write_kill_summary(results, output_dir / "kill_summary.json")

    print("\n" + "="*60)
    print("KILL ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
