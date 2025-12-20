"""CLI entry point for the COWLS field-level study."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from .config import FieldStudyConfig
from .io import LensEntry, discover_lenses, read_fits
from .products import choose_noise_psf, choose_residual_product, list_band_products
from .preprocess import build_arc_mask, load_preprocess, save_preprocess
from .ring_profile import compute_ring_profile
from .stats_field import compute_field_stats, zscore
from .nulls import draw_null_statistics
from .window import build_window
from .report import LensResult, ReportBundle, write_report


def _robust_std(img: np.ndarray) -> float:
    med = np.nanmedian(img)
    mad = np.nanmedian(np.abs(img - med))
    return float(1.4826 * mad + 1e-6)


def _choose_band(lens: LensEntry, requested: str) -> str | None:
    if requested != "auto":
        return requested if requested in lens.bands else None
    # Simple heuristic: prefer redder bands when available
    for candidate in ("F277W", "F444W", "F150W", "F115W"):
        if candidate in lens.bands:
            return candidate
    return lens.bands[0] if lens.bands else None


def _load_noise(path: Path | None, data: np.ndarray) -> np.ndarray:
    if path is None:
        return np.full_like(data, _robust_std(data))
    return read_fits(path)[0]


def _construct_residual(
    residual_products: Dict[str, Path], data: np.ndarray
) -> Tuple[np.ndarray, str]:
    """Construct residual from model products or approximate."""
    from scipy.ndimage import gaussian_filter

    # Direct residual product
    if "residual" in residual_products:
        residual = read_fits(residual_products["residual"])[0]
        return residual, "model_residual"

    # Compute residual by subtracting source_light and lens_light
    if "source_light" in residual_products or "lens_light" in residual_products:
        model = np.zeros_like(data)
        if "source_light" in residual_products:
            source_light = read_fits(residual_products["source_light"])[0]
            if source_light.shape == data.shape:
                model += source_light
        if "lens_light" in residual_products:
            lens_light = read_fits(residual_products["lens_light"])[0]
            if lens_light.shape == data.shape:
                model += lens_light
        if np.any(model != 0):
            return data - model, "model_residual"

    # Approximate residual as data minus smooth background
    background = gaussian_filter(data, sigma=3.0)
    residual = data - background
    return residual, "approx_residual"


def process_lens(
    lens: LensEntry, cfg: FieldStudyConfig, cache_dir: Path
) -> Tuple[LensResult | None, str]:
    band = _choose_band(lens, cfg.band)
    if band is None:
        return None, "no_band"
    products = list_band_products(lens.path, band)
    data_path = products.get("data")
    if data_path is None:
        return None, "no_data"
    data, header = read_fits(data_path)
    noise_path, _ = choose_noise_psf(products)
    noise = _load_noise(noise_path, data)

    residual_products, mode_label = choose_residual_product(products, prefer_model=cfg.prefer_model_residuals)
    residual, mode_label = _construct_residual(residual_products, data)

    cache_path = cache_dir / lens.lens_id / f"{band}_preprocess.npz"
    cache = load_preprocess(cache_path)
    if cache is None:
        meta = {k.lower(): header.get(k) for k in ("X_CENTER", "Y_CENTER", "THETA_E", "R_EIN")} if header else {}
        pre = build_arc_mask(
            image=data,
            noise=noise,
            mode=mode_label,
            metadata=meta,
            snr_threshold=cfg.snr_threshold,
            annulus_width=cfg.annulus_width,
        )
        save_preprocess(cache_path, pre)
    else:
        pre = cache

    theta_bins = np.asarray(cfg.theta_bin_edges())
    s_theta, _ = build_window(pre.arc_mask, data=data, noise=noise, center=pre.center, theta_bins=theta_bins)
    profile = compute_ring_profile(residual, noise, pre.arc_mask, pre.center, theta_bins, window_weights=s_theta)

    stats = compute_field_stats(profile, lag_max=cfg.lag_max, hf_fraction=cfg.hf_fraction)

    residual_samples = (residual / (noise + 1e-6))[pre.arc_mask]
    t_corr_null, t_pow_null = draw_null_statistics(
        profile, mode=cfg.null_mode, residual_samples=residual_samples, lag_max=cfg.lag_max, hf_fraction=cfg.hf_fraction, draws=cfg.null_draws
    )
    z_corr, _, _ = zscore(stats.t_corr, t_corr_null)
    z_pow, _, _ = zscore(stats.t_pow, t_pow_null)

    result = LensResult(
        lens_id=lens.lens_id,
        band=band,
        score_bin=lens.score_bin,
        mode=mode_label,
        t_corr=stats.t_corr,
        t_pow=stats.t_pow,
        z_corr=z_corr,
        z_pow=z_pow,
    )
    return result, mode_label


def aggregate_results(results: List[LensResult]) -> ReportBundle:
    model_results = [r for r in results if r.mode == "model_residual"]
    approx_results = [r for r in results if r.mode != "model_residual"]

    def _mean_z(res: List[LensResult], attr: str) -> float:
        if not res:
            return 0.0
        return float(np.nanmean([getattr(r, attr) for r in res]))

    z_corr_mean = _mean_z(model_results, "z_corr")
    z_pow_mean = _mean_z(model_results, "z_pow")
    z_corr_mean_approx = _mean_z(approx_results, "z_corr")
    z_pow_mean_approx = _mean_z(approx_results, "z_pow")
    z_corr_global = _mean_z(results, "z_corr") * np.sqrt(max(len(results), 1))
    z_pow_global = _mean_z(results, "z_pow") * np.sqrt(max(len(results), 1))

    return ReportBundle(
        subset_label=",".join(sorted({r.score_bin for r in results})) if results else "none",
        n_processed=len(results),
        n_used=len(results),
        model_count=len(model_results),
        approx_count=len(approx_results),
        z_corr_mean=z_corr_mean,
        z_pow_mean=z_pow_mean,
        z_corr_mean_approx=z_corr_mean_approx,
        z_pow_mean_approx=z_pow_mean_approx,
        z_corr_global=z_corr_global,
        z_pow_global=z_pow_global,
        lens_results=results,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="COWLS field-level residual study")
    parser.add_argument("--subset", type=str, default=None, help="Comma-separated lens IDs or score bins")
    parser.add_argument("--score-bins", type=str, default=None, help="Comma-separated score bins to include")
    parser.add_argument("--band", type=str, default="auto", help="Band to use (or auto)")
    parser.add_argument("--null-B", dest="null_draws", type=int, default=300, help="Number of null draws")
    parser.add_argument("--null-mode", type=str, default="both", choices=["shift", "resample", "both"], help="Null generation mode")
    parser.add_argument("--out", type=Path, default=FieldStudyConfig().results_root, help="Output directory")
    parser.add_argument("--lag-max", type=int, default=6)
    parser.add_argument("--hf-frac", type=float, default=0.35)
    parser.add_argument("--snr-thresh", type=float, default=1.5)
    parser.add_argument("--theta-bins", type=str, default=None, help="Comma-separated theta bin edges")
    parser.add_argument("--annulus-width", type=float, default=0.5)
    parser.add_argument("--prefer-model-residuals", action="store_true", help="Prefer explicit model residual products")
    parser.add_argument("--data-root", type=Path, default=FieldStudyConfig().data_root)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = FieldStudyConfig.parse_subset(args.subset.split(",") if args.subset else None)
    score_bins = FieldStudyConfig.parse_subset(args.score_bins.split(",") if args.score_bins else None)
    theta_bins = (
        [float(t) for t in args.theta_bins.split(",") if t.strip()]
        if args.theta_bins
        else FieldStudyConfig().theta_bin_edges()
    )
    cfg = FieldStudyConfig(
        data_root=args.data_root,
        results_root=args.out,
        subset=subset,
        score_bins=score_bins,
        band=args.band,
        prefer_model_residuals=args.prefer_model_residuals,
        snr_threshold=args.snr_thresh,
        annulus_width=args.annulus_width,
        theta_bins=theta_bins,
        lag_max=args.lag_max,
        hf_fraction=args.hf_frac,
        null_mode=args.null_mode,
        null_draws=args.null_draws,
    )

    lenses = discover_lenses(cfg.data_root, subset=cfg.subset, score_bins=cfg.score_bins)
    results: List[LensResult] = []
    cache_dir = cfg.results_root / "cache"
    for lens in lenses:
        res, reason = process_lens(lens, cfg, cache_dir)
        if res is None:
            continue
        results.append(res)

    bundle = aggregate_results(results)
    write_report(cfg.results_root, bundle)


if __name__ == "__main__":
    main()
