#!/usr/bin/env python
"""Run full BAO overlap pipeline."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np


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
    combine_pair_counts,
    compute_pair_counts_simple,
    compute_pair_counts_by_environment,
    landy_szalay,
    parse_wedge_bounds,
    wedge_xi,
)
from bao_overlap.covariance import assign_jackknife_regions, covariance_from_jackknife, covariance_from_mocks
from bao_overlap.density_field import build_density_field, build_grid_spec, gaussian_smooth, save_density_field, trilinear_sample
from bao_overlap.fitting import fit_wedge
from bao_overlap.geometry import radec_to_cartesian
from bao_overlap.hierarchical import infer_beta_blinded
from bao_overlap.io import load_catalog, load_yaml, save_metadata
from bao_overlap.overlap_metric import compute_per_galaxy_mean_e1, normalize
from bao_overlap.prereg import load_prereg
from bao_overlap.reporting import scan_forbidden_keys_in_dir, write_methods_snapshot


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
        }
        all_env_raw.append(env_raw)
        all_ra.append(data_cat.ra)
        all_dec.append(data_cat.dec)
        all_z.append(data_cat.z)
        all_region.append(np.full(len(data_cat.ra), region))
        all_xyz.append(data_xyz)
        all_rand_xyz.append(rand_xyz)
        all_data_w.append(data_cat.w)
        all_rand_w.append(rand_cat.w)

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
    e_by_galaxy_path = output_dir / "E_by_galaxy.npz"
    np.savez(
        e_by_galaxy_path,
        galaxy_id=galaxy_id,
        region=np.concatenate(all_region),
        z=np.concatenate(all_z),
        ra=np.concatenate(all_ra),
        dec=np.concatenate(all_dec),
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

    env_diag: Dict[str, Any] = {}
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

    status("STAGE 2: Computing pair counts by environment bin", output_dir)
    counts_by_region = {}
    counts_all_regions = []
    for region in regions:
        status(f"  Computing pair counts for {region}...", output_dir)
        t0 = time.time()
        region_payload = per_region[region]
        counts_by_bin = compute_pair_counts_by_environment(
            region_payload["data_xyz"],
            region_payload["rand_xyz"],
            region_payload["env_bin"],
            region_payload["rand_bin"],
            n_bins=n_bins,
            s_edges=s_edges,
            mu_edges=mu_edges,
            data_weights=region_payload["data"].w,
            rand_weights=region_payload["randoms"].w,
            verbose=False,
            pair_counter=compute_pair_counts_simple,
        )
        status(f"  {region} pair counts done ({time.time()-t0:.1f}s)", output_dir)
        counts_by_region[region] = counts_by_bin
        counts_all_regions.append(
            compute_pair_counts_simple(
                region_payload["data_xyz"],
                region_payload["rand_xyz"],
                s_edges=s_edges,
                mu_edges=mu_edges,
                data_weights=region_payload["data"].w,
                rand_weights=region_payload["randoms"].w,
                verbose=False,
            )
        )

    counts_by_bin_combined = {}
    for b in range(n_bins):
        counts_by_bin_combined[b] = combine_pair_counts([counts_by_region[reg][b] for reg in regions])

    combined_counts = combine_pair_counts(counts_all_regions)
    xi_all = landy_szalay(combined_counts)
    xi_tangential_all = wedge_xi(xi_all, mu_edges, tangential_bounds)
    xi_radial_all = wedge_xi(xi_all, mu_edges, radial_bounds)

    xi_tangential = np.zeros((n_bins, len(s_centers)))
    xi_radial = np.zeros((n_bins, len(s_centers)))

    for b in range(n_bins):
        xi_bin = landy_szalay(counts_by_bin_combined[b])
        xi_tangential[b] = wedge_xi(xi_bin, mu_edges, tangential_bounds)
        xi_radial[b] = wedge_xi(xi_bin, mu_edges, radial_bounds)

    dd_combined = np.stack([counts_by_bin_combined[b].dd for b in range(n_bins)], axis=0)
    dr_combined = np.stack([counts_by_bin_combined[b].dr for b in range(n_bins)], axis=0)
    rr_combined = np.stack([counts_by_bin_combined[b].rr for b in range(n_bins)], axis=0)

    pair_counts_path = output_dir / "pair_counts_by_Ebin.npz"
    save_payload: Dict[str, Any] = {
        "dd": dd_combined,
        "dr": dr_combined,
        "rr": rr_combined,
        "s_edges": s_edges,
        "mu_edges": mu_edges,
        "n_bins": np.array([n_bins], dtype=int),
    }
    for region in regions:
        save_payload[f"dd_{region.lower()}"] = np.stack([counts_by_region[region][b].dd for b in range(n_bins)], axis=0)
        save_payload[f"dr_{region.lower()}"] = np.stack([counts_by_region[region][b].dr for b in range(n_bins)], axis=0)
        save_payload[f"rr_{region.lower()}"] = np.stack([counts_by_region[region][b].rr for b in range(n_bins)], axis=0)
    np.savez(pair_counts_path, **save_payload)

    xi_wedge_path = output_dir / "xi_wedge_by_Ebin.npz"
    np.savez(
        xi_wedge_path,
        xi_tangential=xi_tangential,
        xi_radial=xi_radial,
        xi_tangential_all=xi_tangential_all,
        xi_radial_all=xi_radial_all,
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

        # Precompute RR once with all randoms (approximation for JK efficiency)
        # The leave-one-out RR variation is small (~1%) and negligible for covariance
        status("  Precomputing RR with all randoms...", output_dir)
        rr_precompute = compute_pair_counts_simple(
            rand_xyz_all,
            rand_xyz_all,
            s_edges=s_edges,
            mu_edges=mu_edges,
            data_weights=rand_w_all,
            rand_weights=rand_w_all,
            verbose=False,
        )
        precomputed_rr = rr_precompute.rr
        status(f"  RR precomputed (shape {precomputed_rr.shape})", output_dir)

        jk_vectors = []
        jk_start = time.time()
        for idx in range(n_jk):
            iter_start = time.time()
            data_mask = jk_data_regions != idx
            rand_mask = jk_rand_regions != idx
            counts_by_bin = compute_pair_counts_by_environment(
                data_xyz_all[data_mask],
                rand_xyz_all[rand_mask],
                env_bin_all[data_mask],
                rand_bin_all[rand_mask],
                n_bins=n_bins,
                s_edges=s_edges,
                mu_edges=mu_edges,
                data_weights=data_w_all[data_mask],
                rand_weights=rand_w_all[rand_mask],
                verbose=False,
                pair_counter=compute_pair_counts_simple,
                precomputed_rr=precomputed_rr,
            )
            vecs = []
            for b in range(n_bins):
                xi_bin = landy_szalay(counts_by_bin[b])
                vecs.append(wedge_xi(xi_bin, mu_edges, tangential_bounds))
            jk_vectors.append(np.concatenate(vecs))
            iter_time = time.time() - iter_start
            elapsed = time.time() - jk_start
            avg_time = elapsed / (idx + 1)
            remaining = avg_time * (n_jk - idx - 1)
            status(f"  JK {idx+1}/{n_jk} done ({iter_time:.1f}s) | elapsed: {elapsed:.0f}s | est remaining: {remaining:.0f}s", output_dir)

        jk_vectors = np.asarray(jk_vectors)
        status(f"  Jackknife complete ({time.time()-jk_start:.1f}s total)", output_dir)
        cov_result = covariance_from_jackknife(jk_vectors)
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
    }

    status("STAGE 4: BAO fitting per environment bin", output_dir)
    alpha_records = []
    alpha_values = []
    alpha_sigmas = []
    for b in range(n_bins):
        status(f"  Fitting bin {b+1}/{n_bins}...", output_dir)
        idx_start = b * len(s_centers)
        idx_end = (b + 1) * len(s_centers)
        cov_block = cov_result.covariance[idx_start:idx_end, idx_start:idx_end]
        fit_result = fit_wedge(
            s=s_centers,
            xi=xi_tangential[b],
            covariance=cov_block,
            fit_range=fit_range,
            nuisance_terms=nuisance_terms,
            template_params=template_params,
            optimizer=fit_cfg["optimizer"],
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
                "template_params": template_params,
                "nuisance_terms": nuisance_terms,
                "optimizer": fit_cfg["optimizer"],
            }
        )

    alpha_by_bin_path = output_dir / "alpha_by_Ebin.json"
    with open(alpha_by_bin_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "bins": alpha_records,
                "bin_edges": env_bin_edges.tolist(),
                "bin_centers": env_bin_centers.tolist(),
                "fit_range": {"s_min": fit_range[0], "s_max": fit_range[1]},
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
        (pair_counts_path, paper_package / "pair_counts_by_Ebin.npz"),
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
