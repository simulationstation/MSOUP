#!/usr/bin/env python
"""Run full BAO overlap pipeline."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
from joblib import Parallel, delayed

# Number of parallel workers
N_JOBS = min(50, max(1, int(os.environ.get("BAO_N_JOBS", 50))))


def status(msg: str, output_dir: Path | None = None) -> None:
    """Print status message with timestamp and optionally write to status file."""
    timestamp = time.strftime("%H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    if output_dir is not None:
        status_file = output_dir / "pipeline_status.log"
        with open(status_file, "a", encoding="utf-8") as f:
            f.write(line + "\n")

from bao_overlap.blinding import BlindState, compute_prereg_hash, initialize_blinding, save_blinded_results
from bao_overlap.correlation import (
    compute_xi_s_mu,
    parse_wedge_bounds,
    wedge_xi,
)
from bao_overlap.covariance import (
    assign_jackknife_regions,
    covariance_from_jackknife,
    covariance_from_mocks,
    StreamingJackknifeCovariance,
)
from bao_overlap.density_field import build_density_field, build_grid_spec, gaussian_smooth, save_density_field, trilinear_sample
from bao_overlap.bao_template import bao_template
from bao_overlap.fitting import fit_wedge
from bao_overlap.diagnostic_bao import FitFailure, fit_wedge_diagnostic, fit_wiggle_only_monopole
from bao_overlap.geometry import radec_to_cartesian
from bao_overlap.hierarchical import infer_beta_blinded
from bao_overlap.io import load_catalog, load_yaml, save_metadata
from bao_overlap.overlap_metric import compute_per_galaxy_mean_e1, normalize
from bao_overlap.prereg import load_prereg
from bao_overlap.reporting import scan_forbidden_keys_in_dir, write_methods_snapshot
from bao_overlap.selection import (
    apply_f_b_weights,
    build_f_b_grid,
    compute_cell_keys,
    compute_selection_function_f_b,
)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _jk_iteration(
    idx: int,
    data_xyz_all: np.ndarray,
    rand_xyz_all: np.ndarray,
    env_bin_all: np.ndarray,
    rand_bin_all: np.ndarray,
    data_w_all: np.ndarray,
    rand_w_all: np.ndarray,
    rand_cell_keys_all: np.ndarray,
    f_b_grid: np.ndarray,
    jk_data_regions: np.ndarray,
    jk_rand_regions: np.ndarray,
    n_bins: int,
    s_edges: np.ndarray,
    mu_edges: np.ndarray,
    tangential_bounds: tuple,
) -> np.ndarray:
    """Compute xi wedge vector for a single JK iteration (for parallel execution)."""
    data_mask = jk_data_regions != idx
    rand_mask = jk_rand_regions != idx
    rand_cell_keys = rand_cell_keys_all[rand_mask]

    rand_weights_by_bin = np.vstack(
        [
            apply_f_b_weights(rand_w_all[rand_mask], rand_cell_keys, f_b_grid, b)
            for b in range(n_bins)
        ]
    )

    vecs = []
    for b in range(n_bins):
        bin_mask = data_mask & (env_bin_all == b)
        if not np.any(bin_mask):
            xi_bin = np.zeros((len(s_edges) - 1, len(mu_edges) - 1))
        else:
            xi_bin = compute_xi_s_mu(
                data_xyz_all[bin_mask],
                rand_xyz_all[rand_mask],
                s_edges=s_edges,
                mu_edges=mu_edges,
                data_weights=data_w_all[bin_mask],
                rand_weights=rand_weights_by_bin[b],
                verbose=False,
            ).xi
        vecs.append(np.mean(xi_bin, axis=1))
    return np.concatenate(vecs)


def run_pipeline(config_path: Path, dry_run: bool = False) -> None:
    cfg = load_yaml(config_path)
    prereg = load_prereg(cfg["preregistration"])
    datasets = load_yaml(cfg["datasets"])
    cfg["_preregistration"] = prereg
    cfg["_datasets"] = datasets

    output_dir = Path(cfg["output_dir"])
    _ensure_dir(output_dir)

    status("STAGE 0: Pipeline initialization", output_dir)

    seed = prereg["random_seed"]
    rng = np.random.default_rng(seed)
    prereg_hash = compute_prereg_hash(Path(cfg["preregistration"]))
    (output_dir / "prereg_hash.txt").write_text(prereg_hash, encoding="utf-8")

    dry_run_fraction = cfg.get("runtime", {}).get("dry_run_fraction") if dry_run else None
    regions = prereg["primary_dataset"]["regions"]

    cosmo_cfg = prereg["fiducial_cosmology"]
    env_primary = prereg["environment_metric"]["primary"]
    env_params = env_primary["parameters"]
    normalization_method = env_primary["normalization"]["method"]
    smoothing_radius = env_params["smoothing_radius"]
    line_step = env_params["line_integral_step"]
    pair_subsample_fraction = env_params["pair_subsample_fraction"]

    fit_cfg = prereg["bao_fitting"]
    fit_range = (fit_cfg["fit_range"]["s_min"], fit_cfg["fit_range"]["s_max"])
    nuisance_terms = fit_cfg["nuisance"]["terms"]

    corr_cfg = prereg["correlation"]["binning"]
    s_edges = np.arange(corr_cfg["s_min"], corr_cfg["s_max"] + corr_cfg["s_bin"], corr_cfg["s_bin"])
    mu_edges = np.arange(corr_cfg["mu_min"], corr_cfg["mu_max"] + corr_cfg["mu_bin"], corr_cfg["mu_bin"])
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    fit_mask = (s_centers >= fit_range[0]) & (s_centers <= fit_range[1])
    if not np.any(fit_mask):
        raise ValueError("Fit range excludes all s bins; check preregistration fit_range.")
    fit_s_min = float(np.min(s_centers[fit_mask]))
    fit_s_max = float(np.max(s_centers[fit_mask]))
    n_fit_bins = int(np.sum(fit_mask))
    assert fit_s_min >= fit_range[0], "Fit range lower bound falls below preregistration s_min."

    wedge_cfg = prereg["correlation"]["wedges"]
    tangential_bounds = parse_wedge_bounds(wedge_cfg["tangential"])
    radial_bounds = parse_wedge_bounds(wedge_cfg["radial"])

    n_bins = prereg["environment_binning"]["n_bins"]
    quantile_edges = np.asarray(prereg["environment_binning"]["quantile_edges"], dtype="f8")

    max_outside_fraction = 0.2
    grid_cfg = env_primary.get("grid", {})
    target_cell_size = float(env_params.get("target_cell_size", grid_cfg.get("target_cell_size", 10.0)))
    padding = float(env_params.get("padding", grid_cfg.get("padding", max(3.0 * smoothing_radius, 50.0))))
    max_n_per_axis = int(env_params.get("max_n_per_axis", grid_cfg.get("max_n_per_axis", 512)))

    per_region = {}
    all_env_raw = []
    all_ra = []
    all_dec = []
    all_z = []
    all_region = []
    all_xyz = []
    all_rand_xyz = []
    all_data_w = []
    all_rand_w = []

    # Check for existing E1 results (resume capability)
    e_by_galaxy_path = output_dir / "E_by_galaxy.npz"
    resume_from_e1 = e_by_galaxy_path.exists()

    if resume_from_e1:
        status("STAGE 1: RESUMING from existing E_by_galaxy.npz", output_dir)
        saved_e1 = np.load(e_by_galaxy_path, allow_pickle=True)
        saved_regions = saved_e1["region"]
        saved_e1_raw = saved_e1["E1_raw"]
        saved_e1_norm = saved_e1["E1_value"]
        saved_e_bin = saved_e1["E_bin"]
        env_bin_edges = saved_e1["E_bin_edges"]
        env_bin_centers = saved_e1["E_bin_centers"]
        norm_method_arr = saved_e1["normalization_method"]
        normalization_method = str(norm_method_arr[0]) if len(norm_method_arr) else "median_mad"
        norm_stats = {
            "median": float(saved_e1["normalization_median"][0]),
            "mad": float(saved_e1["normalization_mad"][0]),
            "scale": float(saved_e1["normalization_scale"][0]),
        }

        def _assign_bins(values: np.ndarray) -> np.ndarray:
            bins = np.full_like(values, -1, dtype=int)
            valid = np.isfinite(values)
            if np.any(valid):
                bins[valid] = np.clip(
                    np.digitize(values[valid], env_bin_edges[1:-1], right=False),
                    0,
                    n_bins - 1,
                )
            return bins

        # Load catalogs (quick) and reconstruct per_region
        offset = 0
        for region in regions:
            status(f"  Loading catalog for {region} (resume mode)...", output_dir)
            data_cat, rand_cat = load_catalog(
                datasets_cfg=datasets,
                catalog_key=cfg["catalog"],
                region=region,
                dry_run_fraction=dry_run_fraction,
                seed=seed,
            )
            status(f"  {region}: {len(data_cat.ra)} data, {len(rand_cat.ra)} randoms", output_dir)
            data_w = data_cat.w if data_cat.w is not None else np.ones(len(data_cat.ra))
            rand_w = rand_cat.w if rand_cat.w is not None else np.ones(len(rand_cat.ra))
            geom_cosmo = {"omega_m": cosmo_cfg["omega_m"], "h": cosmo_cfg["h"]}
            data_xyz = radec_to_cartesian(data_cat.ra, data_cat.dec, data_cat.z, **geom_cosmo)
            rand_xyz = radec_to_cartesian(rand_cat.ra, rand_cat.dec, rand_cat.z, **geom_cosmo)

            # Load density field for rand_local
            density_path = output_dir / f"density_field_{region}.npz"
            if density_path.exists():
                from bao_overlap.density_field import load_density_field
                smooth = load_density_field(density_path)
                rand_local = trilinear_sample(smooth, rand_xyz)
            else:
                rand_local = np.zeros(len(rand_xyz))

            n_reg = len(data_cat.ra)
            reg_slice = slice(offset, offset + n_reg)
            env_raw = saved_e1_raw[reg_slice]
            env_norm = saved_e1_norm[reg_slice]
            env_bin = saved_e_bin[reg_slice]

            # Normalize rand_local and assign bins
            scale = norm_stats.get("scale") or 1.0
            center = norm_stats.get("median", 0.0)
            rand_norm = (rand_local - center) / scale
            rand_bin = _assign_bins(rand_norm)

            per_region[region] = {
                "data": data_cat,
                "randoms": rand_cat,
                "data_xyz": data_xyz,
                "rand_xyz": rand_xyz,
                "env_raw": env_raw,
                "env_meta": {},  # Not needed for pair counts
                "env_norm": env_norm,
                "env_bin": env_bin,
                "rand_local": rand_local,
                "rand_bin": rand_bin,
                "data_w": data_w,
                "rand_w": rand_w,
            }
            all_env_raw.append(env_raw)
            all_ra.append(data_cat.ra)
            all_dec.append(data_cat.dec)
            all_z.append(data_cat.z)
            all_region.append(np.full(len(data_cat.ra), region))
            all_xyz.append(data_xyz)
            all_rand_xyz.append(rand_xyz)
            all_data_w.append(data_w)
            all_rand_w.append(rand_w)
            offset += n_reg

        env_raw_all = np.concatenate(all_env_raw)
        env_norm_all = saved_e1_norm
        env_bins_all = saved_e_bin
        status("  Resume complete - skipping to Stage 2", output_dir)

    else:
        status(f"STAGE 1: Processing {len(regions)} regions: {regions}", output_dir)
        for region in regions:
            status(f"  Loading catalog for {region}...", output_dir)
            data_cat, rand_cat = load_catalog(
                datasets_cfg=datasets,
                catalog_key=cfg["catalog"],
                region=region,
                dry_run_fraction=dry_run_fraction,
                seed=seed,
            )
            status(f"  {region}: {len(data_cat.ra)} data, {len(rand_cat.ra)} randoms", output_dir)
            data_w = data_cat.w if data_cat.w is not None else np.ones(len(data_cat.ra))
            rand_w = rand_cat.w if rand_cat.w is not None else np.ones(len(rand_cat.ra))
            geom_cosmo = {"omega_m": cosmo_cfg["omega_m"], "h": cosmo_cfg["h"]}
            data_xyz = radec_to_cartesian(data_cat.ra, data_cat.dec, data_cat.z, **geom_cosmo)
            rand_xyz = radec_to_cartesian(rand_cat.ra, rand_cat.dec, rand_cat.z, **geom_cosmo)

            status(f"  Building density field for {region}...", output_dir)
            grid_spec = build_grid_spec(
                data_xyz=data_xyz,
                random_xyz=rand_xyz,
                target_cell_size=target_cell_size,
                padding=padding,
                max_n_per_axis=max_n_per_axis,
            )
            status(f"  {region} grid: {grid_spec.grid_shape}, cell_sizes: {grid_spec.cell_sizes}", output_dir)
            density = build_density_field(
                data_xyz,
                rand_xyz,
                data_cat.w,
                rand_cat.w,
                grid_spec=grid_spec,
            )
            smooth = gaussian_smooth(density, radius=smoothing_radius)
            save_density_field(
                output_dir / f"density_field_{region}.npz",
                smooth,
                meta={
                    "region": region,
                    "grid_shape": grid_spec.grid_shape,
                    "cell_sizes": grid_spec.cell_sizes.tolist(),
                    "padding": padding,
                    "target_cell_size": target_cell_size,
                    "smoothing_radius": smoothing_radius,
                    "line_integral_step": line_step,
                },
            )

            status(f"  Computing E1 environment metric for {region} ({len(data_xyz)} galaxies)...", output_dir)
            t0 = time.time()
            env_raw, env_meta = compute_per_galaxy_mean_e1(
                field=smooth,
                galaxy_xyz=data_xyz,
                s_min=fit_range[0],
                s_max=fit_range[1],
                step=line_step,
                rng=rng,
                pair_subsample_fraction=pair_subsample_fraction,
                max_outside_fraction=max_outside_fraction,
            )
            valid_e1 = np.sum(np.isfinite(env_raw))
            status(f"  {region} E1 done: {valid_e1}/{len(env_raw)} valid ({time.time()-t0:.1f}s)", output_dir)

            rand_local = trilinear_sample(smooth, rand_xyz)

            per_region[region] = {
                "data": data_cat,
                "randoms": rand_cat,
                "data_xyz": data_xyz,
                "rand_xyz": rand_xyz,
                "env_raw": env_raw,
                "env_meta": env_meta,
                "rand_local": rand_local,
                "data_w": data_w,
                "rand_w": rand_w,
            }
            all_env_raw.append(env_raw)
            all_ra.append(data_cat.ra)
            all_dec.append(data_cat.dec)
            all_z.append(data_cat.z)
            all_region.append(np.full(len(data_cat.ra), region))
            all_xyz.append(data_xyz)
            all_rand_xyz.append(rand_xyz)
            all_data_w.append(data_w)
            all_rand_w.append(rand_w)

        env_raw_all = np.concatenate(all_env_raw)
        valid_mask = np.isfinite(env_raw_all)
        env_norm_all = np.full_like(env_raw_all, np.nan, dtype="f8")
        norm_stats: Dict[str, float] = {}
        if np.any(valid_mask):
            env_norm_all[valid_mask], norm_stats = normalize(env_raw_all[valid_mask], method=normalization_method)

        env_bin_edges = np.quantile(env_norm_all[valid_mask], quantile_edges)
        env_bin_centers = 0.5 * (env_bin_edges[:-1] + env_bin_edges[1:])

        def _assign_bins(values: np.ndarray) -> np.ndarray:
            bins = np.full_like(values, -1, dtype=int)
            valid = np.isfinite(values)
            if np.any(valid):
                bins[valid] = np.clip(
                    np.digitize(values[valid], env_bin_edges[1:-1], right=False),
                    0,
                    n_bins - 1,
                )
            return bins

        env_bins_all = _assign_bins(env_norm_all)

        offset = 0
        for region in regions:
            n_reg = len(per_region[region]["env_raw"])
            reg_slice = slice(offset, offset + n_reg)
            per_region[region]["env_norm"] = env_norm_all[reg_slice]
            per_region[region]["env_bin"] = env_bins_all[reg_slice]
            rand_norm = np.full_like(per_region[region]["rand_local"], np.nan, dtype="f8")
            if np.any(valid_mask):
                scale = norm_stats.get("scale") or norm_stats.get("std") or 1.0
                center = norm_stats.get("median", norm_stats.get("mean", 0.0))
                rand_norm = (per_region[region]["rand_local"] - center) / scale
            per_region[region]["rand_bin"] = _assign_bins(rand_norm)
            offset += n_reg

        galaxy_id = np.arange(len(env_raw_all))
        np.savez(
            e_by_galaxy_path,
            galaxy_id=galaxy_id,
            region=np.concatenate(all_region),
            z=np.concatenate(all_z),
            ra=np.concatenate(all_ra),
            dec=np.concatenate(all_dec),
            weight=np.concatenate(all_data_w),
            E1_raw=env_raw_all,
            E1_value=env_norm_all,
            E_bin=env_bins_all,
            E_bin_edges=env_bin_edges,
            E_bin_centers=env_bin_centers,
            normalization_method=np.array([normalization_method], dtype=object),
            normalization_median=np.array([norm_stats.get("median", np.nan)], dtype="f8"),
            normalization_mad=np.array([norm_stats.get("mad", np.nan)], dtype="f8"),
            normalization_scale=np.array([norm_stats.get("scale", np.nan)], dtype="f8"),
        )

    # Environment diagnostics (skip detailed stats in resume mode)
    env_diag: Dict[str, Any] = {}
    if not resume_from_e1:
        for region in regions:
            meta = per_region[region]["env_meta"]
            attempted = meta["attempted_pairs"]
            valid = meta["valid_pairs"]
            invalid = meta["invalid_pairs"]
            total_attempted = int(attempted.sum())
            total_invalid = int(invalid.sum())
            env_diag[region] = {
                "mean_valid_pairs": float(np.mean(valid)) if len(valid) else 0.0,
                "median_valid_pairs": float(np.median(valid)) if len(valid) else 0.0,
                "min_valid_pairs": int(np.min(valid)) if len(valid) else 0,
                "max_valid_pairs": int(np.max(valid)) if len(valid) else 0,
                "invalid_fraction": float(total_invalid / max(total_attempted, 1)),
            }
    with open(output_dir / "environment_assignment_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(env_diag, handle, indent=2)

    status("  Building selection function f_b(theta, z)...", output_dir)
    selection_path = output_dir / "selection_function_f_b.npz"
    if selection_path.exists():
        selection = np.load(selection_path)
        pixel_ids = selection["pixel_ids"]
        zbin_ids = selection["zbin_ids"]
        f_b = selection["f_b"]
        n_all = selection["n_all"]
        z_edges = selection["z_edges"]
        nside = int(np.atleast_1d(selection["nside"])[0])
    else:
        if resume_from_e1 and "weight" in saved_e1:
            data_weights_all = saved_e1["weight"]
        else:
            data_weights_all = np.concatenate(all_data_w)
        selection = compute_selection_function_f_b(
            ra=np.concatenate(all_ra),
            dec=np.concatenate(all_dec),
            z=np.concatenate(all_z),
            e_bin=env_bins_all,
            weights=data_weights_all,
            n_bins=n_bins,
            nside=32,
            dz=0.05,
            z_min=0.6,
            z_max=1.0,
        )
        pixel_ids = selection["pixel_ids"]
        zbin_ids = selection["zbin_ids"]
        f_b = selection["f_b"]
        n_all = selection["n_all"]
        z_edges = selection["z_edges"]
        nside = int(selection["nside"])
        np.savez(
            selection_path,
            pixel_ids=pixel_ids,
            zbin_ids=zbin_ids,
            f_b=f_b,
            n_all=n_all,
            z_edges=z_edges,
            nside=np.array([nside], dtype=int),
        )

    f_b_grid, _ = build_f_b_grid(pixel_ids, zbin_ids, f_b, nside, z_edges)
    for region in regions:
        rand_cat = per_region[region]["randoms"]
        per_region[region]["rand_cell_keys"] = compute_cell_keys(
            rand_cat.ra,
            rand_cat.dec,
            rand_cat.z,
            nside=nside,
            z_edges=z_edges,
        )

    # Aggressive memory cleanup before pair counts
    status("  Memory cleanup before pair counts...", output_dir)
    import gc
    gc.collect()

    # Terminate any lingering joblib workers
    try:
        from joblib.externals.loky import get_reusable_executor
        get_reusable_executor().shutdown(wait=True)
        status("  Worker pool terminated", output_dir)
    except Exception:
        pass

    gc.collect()

    # Check available memory
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemAvailable:'):
                    available_gb = int(line.split()[1]) / 1e6
                    status(f"  Available memory: {available_gb:.1f} GB", output_dir)
                    break
    except FileNotFoundError:
        pass

    status("STAGE 2: Computing xi by environment bin", output_dir)

    xi_wedge_path = output_dir / "xi_wedge_by_Ebin.npz"
    stage2_complete = xi_wedge_path.exists()

    if stage2_complete:
        status("  STAGE 2 outputs exist - skipping to STAGE 3", output_dir)

    if stage2_complete:
        xi_data = np.load(xi_wedge_path)
        xi_tangential = xi_data["xi_tangential"]
        xi_radial = xi_data["xi_radial"]
        if "xi_monopole" in xi_data:
            xi_monopole = xi_data["xi_monopole"]
        else:
            status("  Warning: xi_monopole missing; falling back to tangential wedge for fitting.", output_dir)
            xi_monopole = xi_tangential
        env_bin_edges = xi_data["E_bin_edges"]
        env_bin_centers = xi_data["E_bin_centers"]
    else:
        data_xyz_all = np.vstack(all_xyz)
        rand_xyz_all = np.vstack(all_rand_xyz)
        data_w_all = np.concatenate(all_data_w)
        rand_w_all = np.concatenate(all_rand_w)
        rand_cell_keys_all = np.concatenate([per_region[r]["rand_cell_keys"] for r in regions])

        status("  Computing xi(s,mu) for all data...", output_dir)
        xi_all = compute_xi_s_mu(
            data_xyz_all,
            rand_xyz_all,
            s_edges=s_edges,
            mu_edges=mu_edges,
            data_weights=data_w_all,
            rand_weights=rand_w_all,
            verbose=False,
        ).xi
        xi_tangential_all = wedge_xi(xi_all, mu_edges, tangential_bounds)
        xi_radial_all = wedge_xi(xi_all, mu_edges, radial_bounds)
        xi_monopole_all = np.mean(xi_all, axis=1)

        tail_mask = s_centers > 155
        if np.any(tail_mask):
            tail_max = max(
                float(np.max(np.abs(xi_tangential_all[tail_mask]))),
                float(np.max(np.abs(xi_monopole_all[tail_mask]))),
            )
            if tail_max > 0.02:
                raise RuntimeError(
                    "Validation FAIL: xi_all tail max|xi| at s>155 exceeds 0.02 "
                    f"(max={tail_max:.3f}). Aborting before JK/covariance."
                )

        xi_tangential = np.zeros((n_bins, len(s_centers)))
        xi_radial = np.zeros((n_bins, len(s_centers)))
        xi_monopole = np.zeros((n_bins, len(s_centers)))

        for b in range(n_bins):
            data_mask = env_bins_all == b
            if not np.any(data_mask):
                xi_bin = np.zeros((len(s_edges) - 1, len(mu_edges) - 1))
            else:
                rand_weights_b = apply_f_b_weights(
                    rand_w_all,
                    rand_cell_keys_all,
                    f_b_grid,
                    b,
                )
                xi_bin = compute_xi_s_mu(
                    data_xyz_all[data_mask],
                    rand_xyz_all,
                    s_edges=s_edges,
                    mu_edges=mu_edges,
                    data_weights=data_w_all[data_mask],
                    rand_weights=rand_weights_b,
                    verbose=False,
                ).xi

            xi_tangential[b] = wedge_xi(xi_bin, mu_edges, tangential_bounds)
            xi_radial[b] = wedge_xi(xi_bin, mu_edges, radial_bounds)
            xi_monopole[b] = np.mean(xi_bin, axis=1)

        paper_package_dir = Path(__file__).resolve().parents[1] / "paper_package"
        _ensure_dir(paper_package_dir)

        s_targets = [102.5, 177.5]
        s_indices = [int(np.argmin(np.abs(s_centers - target))) for target in s_targets]
        s_centers_actual = [float(s_centers[idx]) for idx in s_indices]
        xi_report = {
            "s_targets": s_targets,
            "s_indices": s_indices,
            "s_centers": s_centers_actual,
            "xi_all": {
                str(target): float(xi_monopole_all[idx])
                for target, idx in zip(s_targets, s_indices)
            },
            "xi_by_bin": {},
        }
        for b in range(n_bins):
            xi_report["xi_by_bin"][str(b)] = {
                str(target): float(xi_monopole[b][idx])
                for target, idx in zip(s_targets, s_indices)
            }

        with open(paper_package_dir / "report_per_bin_xi_sanity.json", "w", encoding="utf-8") as handle:
            json.dump(xi_report, handle, indent=2)

        np.savez(
            xi_wedge_path,
            xi_tangential=xi_tangential,
            xi_radial=xi_radial,
            xi_monopole=xi_monopole,
            xi_tangential_all=xi_tangential_all,
            xi_radial_all=xi_radial_all,
            xi_monopole_all=xi_monopole_all,
            s_centers=s_centers,
            mu_wedge_defs=np.array([
                tangential_bounds[0],
                tangential_bounds[1],
                radial_bounds[0],
                radial_bounds[1],
            ]),
            E_bin_edges=env_bin_edges,
            E_bin_centers=env_bin_centers,
        )

    cov_cfg = prereg["covariance"]
    cov_dir = output_dir / "covariance"
    cov_dir.mkdir(parents=True, exist_ok=True)
    mock_cov_path = cov_dir / "mock_xi_wedges.npy"

    cov_method = cov_cfg["primary_method"]
    cov_meta: Dict[str, Any] = {"method": cov_method}

    status("STAGE 3: Computing covariance matrix", output_dir)
    if cov_method == "mocks" and mock_cov_path.exists():
        status("  Using mock covariance", output_dir)
        mock_wedges = np.load(mock_cov_path)
        if mock_wedges.ndim == 3:
            mock_data = mock_wedges.reshape(mock_wedges.shape[0], -1)
        else:
            mock_data = mock_wedges
        cov_result = covariance_from_mocks(mock_data)
        cov_meta.update(cov_result.meta)
    else:
        cov_method = cov_cfg["fallback_method"]
        cov_meta["method"] = cov_method
        status(f"  Using jackknife covariance (method={cov_method})", output_dir)
        jk_cfg = cov_cfg["jackknife"]
        data_xyz_all = np.vstack(all_xyz)
        rand_xyz_all = np.vstack(all_rand_xyz)
        data_ra_all = np.concatenate(all_ra)
        data_dec_all = np.concatenate(all_dec)
        rand_ra_all = np.concatenate([per_region[r]["randoms"].ra for r in regions])
        rand_dec_all = np.concatenate([per_region[r]["randoms"].dec for r in regions])
        data_w_all = np.concatenate(all_data_w)
        rand_w_all = np.concatenate(all_rand_w)
        env_bin_all = env_bins_all
        rand_bin_all = np.concatenate([per_region[r]["rand_bin"] for r in regions])
        rand_cell_keys_all = np.concatenate([per_region[r]["rand_cell_keys"] for r in regions])

        jk_data_regions = assign_jackknife_regions(
            data_ra_all,
            data_dec_all,
            n_regions=jk_cfg["n_regions"],
            scheme=jk_cfg["scheme"],
            nside=jk_cfg.get("nside"),
        )
        jk_rand_regions = assign_jackknife_regions(
            rand_ra_all,
            rand_dec_all,
            n_regions=jk_cfg["n_regions"],
            scheme=jk_cfg["scheme"],
            nside=jk_cfg.get("nside"),
        )

        n_jk = jk_cfg["n_regions"]
        status(f"  Jackknife loop: {n_jk} regions", output_dir)

        # --- MEMORY FIX: Use streaming covariance instead of storing all JK vectors ---
        # Previous code stored jk_vectors list that grew each iteration.
        # Now we use StreamingJackknifeCovariance which maintains only sum_x and sum_xx.
        # This is mathematically equivalent but uses O(M^2) memory instead of O(N*M).
        import gc
        import os

        # Get memory info function
        def get_rss_gb():
            """Get resident set size in GB."""
            try:
                import psutil
                return psutil.Process(os.getpid()).memory_info().rss / 1e9
            except ImportError:
                import resource
                return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # KB to GB

        # Initialize memory debug log
        mem_log_path = output_dir / "jk_memory_debug.log"
        with open(mem_log_path, "w") as f:
            f.write("# JK Memory Debug Log\n")
            f.write("# Memory fix: streaming covariance accumulator\n")
            f.write("# Objects previously growing: jk_vectors list (100 x M floats)\n")
            f.write("# Now using: StreamingJackknifeCovariance (sum_x: M, sum_xx: M x M)\n")
            f.write(f"# Vector size M = n_bins * n_s = {n_bins} * {len(s_edges)-1}\n")
            f.write("#\n")
            f.write("# iter, RSS_GB, accum_bytes, iter_time_s\n")

        # Set aggressive GC thresholds for this stage
        gc.set_threshold(700, 10, 10)

        # Determine vector size: n_bins * n_s_bins (tangential wedge only)
        n_s = len(s_edges) - 1
        vector_size = n_bins * n_s
        jk_accum = StreamingJackknifeCovariance(vector_size)

        jk_start = time.time()
        rss_start = get_rss_gb()

        # --- RESUMABLE JK ITERATIONS WITH CHECKPOINTING ---
        jk_checkpoint_path = output_dir / "jk_checkpoint.npz"
        start_idx = 0

        if jk_checkpoint_path.exists():
            status("  Resuming JK from checkpoint...", output_dir)
            jk_ckpt = np.load(jk_checkpoint_path)
            start_idx = int(jk_ckpt["completed_iterations"].item())
            jk_accum.sum_x = jk_ckpt["sum_x"].copy()
            jk_accum.sum_xx = jk_ckpt["sum_xx"].copy()
            jk_accum.n = int(jk_ckpt["n"].item())
            status(f"  Resumed from iteration {start_idx}/{n_jk}", output_dir)

        # Process in batches for checkpointing (batch_size iterations, then save)
        # SMOKE TEST: Set to 5 for quick validation, then restore to 50 for full run
        jk_batch_size = int(os.environ.get("BAO_JK_BATCH", "50"))

        for batch_start in range(start_idx, n_jk, jk_batch_size):
            batch_end = min(batch_start + jk_batch_size, n_jk)
            batch_indices = list(range(batch_start, batch_end))

            status(f"  [JK] Processing iterations {batch_start}-{batch_end-1}/{n_jk-1}...", output_dir)

            jk_results = Parallel(n_jobs=N_JOBS, verbose=10, batch_size=1)(
                delayed(_jk_iteration)(
                    idx=idx,
                    data_xyz_all=data_xyz_all,
                    rand_xyz_all=rand_xyz_all,
                    env_bin_all=env_bin_all,
                    rand_bin_all=rand_bin_all,
                    data_w_all=data_w_all,
                    rand_w_all=rand_w_all,
                    rand_cell_keys_all=rand_cell_keys_all,
                    f_b_grid=f_b_grid,
                    jk_data_regions=jk_data_regions,
                    jk_rand_regions=jk_rand_regions,
                    n_bins=n_bins,
                    s_edges=s_edges,
                    mu_edges=mu_edges,
                    tangential_bounds=tangential_bounds,
                )
                for idx in batch_indices
            )

            # Update streaming accumulator with batch results
            for xi_jk in jk_results:
                jk_accum.update(xi_jk)

            # Save checkpoint after each batch
            np.savez(
                jk_checkpoint_path,
                completed_iterations=np.array([batch_end]),
                sum_x=jk_accum.sum_x,
                sum_xx=jk_accum.sum_xx,
                n=np.array([jk_accum.n]),
            )

            # Memory cleanup and status
            gc.collect()
            try:
                with open('/proc/meminfo') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            avail = int(line.split()[1]) / 1e6
                            status(f"  [JK] Checkpoint saved ({batch_end}/{n_jk}), Memory: {avail:.1f} GB", output_dir)
                            break
            except:
                status(f"  [JK] Checkpoint saved ({batch_end}/{n_jk})", output_dir)

        rss_now = get_rss_gb()
        with open(mem_log_path, "a") as f:
            f.write(f"parallel_complete, {rss_now:.3f}, {jk_accum.memory_bytes()}, {time.time()-jk_start:.1f}\n")

        # Finalize covariance from streaming accumulator
        cov_result = jk_accum.finalize()
        status(f"  Jackknife complete ({time.time()-jk_start:.1f}s total) RSS_final={get_rss_gb():.2f}GB", output_dir)
        cov_meta.update(cov_result.meta)

    cov_path = cov_dir / "xi_wedge_covariance.npy"
    np.save(cov_path, cov_result.covariance)
    with open(cov_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(cov_meta, handle, indent=2)

    template_params = {
        "r_d": prereg["fiducial_cosmology"]["r_d"],
        "sigma_nl": prereg["reconstruction"]["parameters"]["smoothing"],
        "omega_m": prereg["fiducial_cosmology"]["omega_m"],
        "omega_b": prereg["fiducial_cosmology"]["omega_b"],
        "h": prereg["fiducial_cosmology"]["h"],
        "n_s": prereg["fiducial_cosmology"]["n_s"],
        "sigma8": prereg["fiducial_cosmology"]["sigma8"],
    }

    paper_package_dir = Path(__file__).resolve().parents[1] / "paper_package"
    _ensure_dir(paper_package_dir)
    template_values = bao_template(
        s_centers,
        r_d=template_params["r_d"],
        sigma_nl=template_params["sigma_nl"],
        omega_m=template_params["omega_m"],
        omega_b=template_params["omega_b"],
        h=template_params["h"],
        n_s=template_params["n_s"],
        sigma8=template_params["sigma8"],
    )
    with open(paper_package_dir / "template_values.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "s_centers": s_centers.tolist(),
                "xi_template": template_values.tolist(),
                "fit_range": {"s_min": fit_range[0], "s_max": fit_range[1]},
            },
            handle,
            indent=2,
        )

    status("STAGE 4: BAO fitting per environment bin", output_dir)
    diagnostic_cfg = cfg.get("diagnostic_bao", {})
    diagnostic_enabled = bool(diagnostic_cfg.get("enabled", False))
    diagnostic_alpha_points = int(diagnostic_cfg.get("alpha_scan_points", 41))
    diagnostic_sigma_bounds = tuple(diagnostic_cfg.get("sigma_nl_bounds", (0.0, 15.0)))
    wiggle_cfg = cfg.get("diagnostic_wiggle_only", {})
    wiggle_enabled = bool(wiggle_cfg.get("enabled", False))
    wiggle_alpha_points = int(wiggle_cfg.get("alpha_scan_points", 41))
    wiggle_sigma_bounds = wiggle_cfg.get("sigma_nl_bounds")
    wiggle_sigma_bounds = (
        tuple(wiggle_sigma_bounds)
        if wiggle_sigma_bounds is not None
        else None
    )
    alpha_records = []
    alpha_values = []
    alpha_sigmas = []
    for b in range(n_bins):
        status(f"  Fitting bin {b+1}/{n_bins}...", output_dir)
        idx_start = b * len(s_centers)
        idx_end = (b + 1) * len(s_centers)
        cov_block = cov_result.covariance[idx_start:idx_end, idx_start:idx_end]
        # Alpha bounds widened from (0.8, 1.2) to (0.6, 1.4) as numerical safety fix
        # See AUDIT.md section "Alpha Bounds Widening" for justification
        alpha_bounds = fit_cfg.get("alpha_bounds", (0.6, 1.4))
        fit_result = fit_wedge(
            s=s_centers,
            xi=xi_monopole[b],
            covariance=cov_block,
            fit_range=fit_range,
            nuisance_terms=nuisance_terms,
            template_params=template_params,
            optimizer=fit_cfg["optimizer"],
            alpha_bounds=alpha_bounds,
        )
        alpha_values.append(fit_result.alpha)
        alpha_sigmas.append(fit_result.sigma_alpha)
        alpha_records.append(
            {
                "bin": b,
                "alpha_perp_hat": fit_result.alpha,
                "alpha_perp_sigma": fit_result.sigma_alpha,
                "chi2": fit_result.chi2,
                "dof": fit_result.meta.get("dof"),
                "nuisance": {
                    "bias_coeff": fit_result.meta.get("bias_coeff"),
                    "coeffs": fit_result.meta.get("nuisance_coeffs"),
                },
                "fit_range": {"s_min": fit_range[0], "s_max": fit_range[1]},
                "alpha_bounds": list(alpha_bounds),
                "template_params": template_params,
                "nuisance_terms": nuisance_terms,
                "optimizer": fit_cfg["optimizer"],
            }
        )

    if diagnostic_enabled:
        status("  Diagnostic mode enabled: profiling Sigma_nl per alpha.", output_dir)
        diagnostic_records = []
        for b in range(n_bins):
            status(f"  [Diagnostic] Fitting bin {b+1}/{n_bins}...", output_dir)
            idx_start = b * len(s_centers)
            idx_end = (b + 1) * len(s_centers)
            cov_block = cov_result.covariance[idx_start:idx_end, idx_start:idx_end]
            alpha_bounds = fit_cfg.get("alpha_bounds", (0.6, 1.4))
            diagnostic_fit = fit_wedge_diagnostic(
                s=s_centers,
                xi=xi_monopole[b],
                covariance=cov_block,
                fit_range=fit_range,
                nuisance_terms=nuisance_terms,
                template_params=template_params,
                optimizer=fit_cfg["optimizer"],
                alpha_bounds=alpha_bounds,
                sigma_nl_bounds=diagnostic_sigma_bounds,
                alpha_scan_points=diagnostic_alpha_points,
            )
            diagnostic_records.append(
                {
                    "bin": b,
                    "alpha_hat": diagnostic_fit.alpha,
                    "sigma_nl_hat": diagnostic_fit.sigma_nl,
                    "chi2": diagnostic_fit.chi2,
                    "dof": diagnostic_fit.dof,
                    "chi2_per_dof": diagnostic_fit.chi2 / max(diagnostic_fit.dof, 1),
                    "sigma_nl_hit_bounds": diagnostic_fit.sigma_nl_hit_bounds,
                    "fit_range": {"s_min": fit_range[0], "s_max": fit_range[1]},
                    "alpha_bounds": list(alpha_bounds),
                    "sigma_nl_bounds": list(diagnostic_sigma_bounds),
                    "template_params": template_params,
                    "nuisance_terms": nuisance_terms,
                    "optimizer": fit_cfg["optimizer"],
                    "diagnostic_meta": diagnostic_fit.meta,
                }
            )

        diagnostic_path = output_dir / "alpha_by_Ebin_diagnostic.json"
        with open(diagnostic_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "bins": diagnostic_records,
                    "bin_edges": env_bin_edges.tolist(),
                    "bin_centers": env_bin_centers.tolist(),
                    "fit_range": {"s_min": fit_range[0], "s_max": fit_range[1]},
                    "template": "bao_template_damped",
                    "nuisance": fit_cfg["nuisance"],
                    "diagnostic_bao": {
                        "enabled": diagnostic_enabled,
                        "alpha_scan_points": diagnostic_alpha_points,
                        "sigma_nl_bounds": list(diagnostic_sigma_bounds),
                        "multipoles": ["xi0"],
                    },
                },
                handle,
                indent=2,
            )

    if wiggle_enabled:
        status("  Diagnostic mode enabled: wiggle-only monopole fit.", output_dir)
        wiggle_records = []
        alpha_boundary_hits = []
        for b in range(n_bins):
            status(f"  [Wiggle-only] Fitting bin {b+1}/{n_bins}...", output_dir)
            idx_start = b * len(s_centers)
            idx_end = (b + 1) * len(s_centers)
            cov_block = cov_result.covariance[idx_start:idx_end, idx_start:idx_end]
            alpha_bounds = fit_cfg.get("alpha_bounds", (0.6, 1.4))
            wiggle_fit = fit_wiggle_only_monopole(
                s=s_centers,
                xi=xi_monopole[b],
                covariance=cov_block,
                fit_range=fit_range,
                template_params=template_params,
                alpha_bounds=alpha_bounds,
                alpha_scan_points=wiggle_alpha_points,
                sigma_nl_bounds=wiggle_sigma_bounds,
                optimizer=fit_cfg["optimizer"],
            )
            alpha_boundary_hits.append(wiggle_fit.alpha_hit_bounds)
            wiggle_records.append(
                {
                    "bin": b,
                    "alpha_hat": wiggle_fit.alpha,
                    "sigma_nl_hat": wiggle_fit.sigma_nl,
                    "chi2": wiggle_fit.chi2,
                    "dof": wiggle_fit.dof,
                    "chi2_per_dof": wiggle_fit.chi2 / max(wiggle_fit.dof, 1),
                    "alpha_hit_bounds": wiggle_fit.alpha_hit_bounds,
                    "sigma_nl_hit_bounds": wiggle_fit.sigma_nl_hit_bounds,
                    "fit_range": {"s_min": fit_range[0], "s_max": fit_range[1]},
                    "alpha_bounds": list(alpha_bounds),
                    "sigma_nl_bounds": list(wiggle_sigma_bounds) if wiggle_sigma_bounds else None,
                    "template_params": template_params,
                    "optimizer": fit_cfg["optimizer"],
                    "diagnostic_meta": wiggle_fit.meta,
                }
            )

        if alpha_boundary_hits and all(alpha_boundary_hits):
            raise FitFailure(
                "Wiggle-only diagnostic fit collapsed to alpha boundaries in all bins."
            )

        wiggle_path = output_dir / "alpha_by_Ebin_wiggle_only.json"
        with open(wiggle_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "bins": wiggle_records,
                    "bin_edges": env_bin_edges.tolist(),
                    "bin_centers": env_bin_centers.tolist(),
                    "fit_range": {"s_min": fit_range[0], "s_max": fit_range[1]},
                    "template": "wiggle_only_monopole",
                    "broadband": "poly2",
                    "diagnostic_wiggle_only": {
                        "enabled": wiggle_enabled,
                        "alpha_scan_points": wiggle_alpha_points,
                        "sigma_nl_bounds": list(wiggle_sigma_bounds) if wiggle_sigma_bounds else None,
                        "multipoles": ["xi0"],
                    },
                },
                handle,
                indent=2,
            )

    alpha_by_bin_path = output_dir / "alpha_by_Ebin.json"
    with open(alpha_by_bin_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bins": alpha_records,
                "bin_edges": env_bin_edges.tolist(),
                "bin_centers": env_bin_centers.tolist(),
                "fit_range": {"s_min": fit_range[0], "s_max": fit_range[1]},
                "fit_s_min": fit_s_min,
                "fit_s_max": fit_s_max,
                "n_fit_bins": n_fit_bins,
                "template": fit_cfg["template"],
                "nuisance": fit_cfg["nuisance"],
            },
            handle,
            indent=2,
        )

    alpha_values = np.asarray(alpha_values)
    alpha_cov = np.diag(np.square(alpha_sigmas))

    hier_cfg = prereg["hierarchical_inference"]
    method = hier_cfg["primary_method"]
    prior_sigma = None
    if method == "bayesian":
        prior_sigma = hier_cfg["bayesian_option"]["priors"]["beta_sigma"]

    status("STAGE 5: Blinding and hierarchical inference", output_dir)
    blind_state = BlindState(
        unblind=cfg.get("blinding", {}).get("unblind", False),
        key_file=prereg["blinding"]["encryption"]["key_file"],
    )

    if cfg.get("blinding", {}).get("enabled", True) and not blind_state.unblind:
        blind_state = initialize_blinding(blind_state, rng)
        blinded = infer_beta_blinded(
            env_bin_centers,
            alpha_values,
            alpha_cov,
            blind_state,
            prereg_hash,
            method=method,
            prior_sigma=prior_sigma,
        )
        save_blinded_results(blinded, output_dir / "blinded_results.json")
    else:
        raise RuntimeError("Unblinded runs are disabled in this pipeline version.")

    save_metadata(output_dir / "metadata.json", {"seed": seed, "catalog": cfg["catalog"]})
    status("STAGE 6: Writing output package", output_dir)

    if prereg["reporting"]["generate_methods_snapshot"]:
        resolved = {
            "random_seed": seed,
            "environment": {
                "smoothing_radius": smoothing_radius,
                "line_integral_step": line_step,
                "pair_subsample_fraction": pair_subsample_fraction,
                "normalization_method": normalization_method,
                "max_outside_fraction": max_outside_fraction,
            },
            "correlation": {
                "s_edges": s_edges.tolist(),
                "mu_edges": mu_edges.tolist(),
                "tangential_bounds": list(tangential_bounds),
                "radial_bounds": list(radial_bounds),
                "n_bins": n_bins,
            },
            "bao_fitting": {
                "fit_range": {"s_min": fit_range[0], "s_max": fit_range[1]},
                "nuisance_terms": nuisance_terms,
                "optimizer": fit_cfg["optimizer"],
                "template": fit_cfg["template"],
            },
            "density_field": {
                "target_cell_size": target_cell_size,
                "padding": padding,
                "max_n_per_axis": max_n_per_axis,
            },
        }
        write_methods_snapshot({"config": cfg, "resolved_parameters": resolved}, output_dir / "methods_snapshot.yaml")

    paper_package = output_dir / "paper_package"
    paper_package.mkdir(parents=True, exist_ok=True)
    (paper_package / "covariance").mkdir(parents=True, exist_ok=True)

    for src, dest in [
        (output_dir / "metadata.json", paper_package / "metadata.json"),
        (e_by_galaxy_path, paper_package / "E_by_galaxy.npz"),
        (output_dir / "environment_assignment_diagnostics.json", paper_package / "environment_assignment_diagnostics.json"),
        (xi_wedge_path, paper_package / "xi_wedge_by_Ebin.npz"),
        (alpha_by_bin_path, paper_package / "alpha_by_Ebin.json"),
        (cov_path, paper_package / "covariance" / "xi_wedge_covariance.npy"),
        (cov_dir / "metadata.json", paper_package / "covariance" / "metadata.json"),
        (output_dir / "methods_snapshot.yaml", paper_package / "methods_snapshot.yaml"),
        (output_dir / "prereg_hash.txt", paper_package / "prereg_hash.txt"),
        (output_dir / "blinded_results.json", paper_package / "blinded_results.json"),
    ]:
        if src.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(src.read_bytes())

    stage_summary = {
        "environment_assignment": "PASS" if np.any(valid_mask) else "FAIL",
        "per_bin_xi": "PASS" if xi_tangential.shape[0] == n_bins else "FAIL",
        "bao_fits": "PASS" if len(alpha_records) == n_bins else "FAIL",
        "covariance": {"method": cov_meta.get("method"), "n": cov_meta.get("n_mocks") or cov_meta.get("n_jackknife")},
        "inference": "PASS" if (output_dir / "blinded_results.json").exists() else "FAIL",
    }
    with open(output_dir / "stage_summary.json", "w", encoding="utf-8") as handle:
        json.dump(stage_summary, handle, indent=2)
    (paper_package / "stage_summary.json").write_bytes((output_dir / "stage_summary.json").read_bytes())

    try:
        scan_forbidden_keys_in_dir(paper_package)
    except ValueError as exc:
        matches = [line for line in str(exc).splitlines() if ":" in line]
        files = {line.split(":", 1)[0] for line in matches}
        for path in files:
            file_path = Path(path)
            if file_path.exists():
                file_path.unlink()
        raise

    status("PIPELINE COMPLETE - All stages finished successfully", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
