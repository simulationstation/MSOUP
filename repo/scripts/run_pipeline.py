#!/usr/bin/env python
"""Run full BAO overlap pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bao_overlap.blinding import (
    BlindState,
    blind_results,
    compute_prereg_hash,
    initialize_blinding,
    save_blinded_results,
)
from bao_overlap.correlation import (
    bin_by_environment,
    combine_pair_counts,
    compute_pair_counts,
    landy_szalay,
    parse_wedge_bounds,
    wedge_xi,
)
from bao_overlap.covariance import assign_jackknife_regions, covariance_from_jackknife, covariance_from_mocks
from bao_overlap.density_field import build_density_field, gaussian_smooth
from bao_overlap.fitting import fit_wedge
from bao_overlap.geometry import radec_to_cartesian
from bao_overlap.hierarchical import bayesian_beta, two_step_beta
from bao_overlap.io import load_run_config, load_catalog, save_metadata
from bao_overlap.overlap_metric import compute_environment
from bao_overlap.plotting import plot_beta_null, plot_wedge
from bao_overlap.reporting import write_methods_snapshot, write_results


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_pipeline(config_path: Path, dry_run: bool = False) -> None:
    cfg = load_run_config(config_path)
    prereg = cfg["_preregistration"]
    datasets = cfg["_datasets"]

    output_dir = Path(cfg["output_dir"])
    _ensure_dir(output_dir)

    seed = prereg["random_seed"]
    rng = np.random.default_rng(seed)

    dry_run_fraction = cfg["runtime"].get("dry_run_fraction") if dry_run else None
    regions = prereg["primary_dataset"]["regions"]

    cosmo_cfg = prereg["fiducial_cosmology"]
    env_primary = prereg["environment_metric"]["primary"]
    env_params = env_primary["parameters"]
    normalization_method = env_primary["normalization"]["method"]
    primary_metric = env_primary["name"]
    smoothing_radius = env_params["smoothing_radius"]

    all_env = []
    data_xyz_all = []
    rand_xyz_all = []
    data_ra_all = []
    data_dec_all = []
    rand_ra_all = []
    rand_dec_all = []
    data_w_all = []
    rand_w_all = []
    counts_list = []

    for region in regions:
        data_cat, rand_cat = load_catalog(
            datasets_cfg=datasets,
            catalog_key=cfg["catalog"],
            region=region,
            dry_run_fraction=dry_run_fraction,
            seed=seed,
        )
        data_xyz = radec_to_cartesian(data_cat.ra, data_cat.dec, data_cat.z, **cosmo_cfg)
        rand_xyz = radec_to_cartesian(rand_cat.ra, rand_cat.dec, rand_cat.z, **cosmo_cfg)

        density = build_density_field(data_xyz, rand_xyz, rand_cat.w, grid_size=64, cell_size=5.0)
        smooth = gaussian_smooth(density, radius=smoothing_radius)

        pair_indices = rng.choice(len(data_xyz), size=min(200, len(data_xyz)), replace=False)
        pairs = np.stack([data_xyz[pair_indices], data_xyz[pair_indices[::-1]]], axis=1)

        env = compute_environment(
            field=smooth,
            galaxy_xyz=data_xyz,
            pair_xyz=pairs,
            step=env_params["line_integral_step"],
            rng=rng,
            subsample=env_params["pair_subsample_fraction"],
            delta_threshold=prereg["environment_metric"]["secondary"]["parameters"]["delta_threshold"],
            min_volume=10,
            normalize_output=True,
            normalization_method=normalization_method,
            primary=primary_metric,
        )

        corr_cfg = prereg["correlation"]["binning"]
        s_edges = np.arange(
            corr_cfg["s_min"],
            corr_cfg["s_max"] + corr_cfg["s_bin"],
            corr_cfg["s_bin"],
        )
        mu_edges = np.arange(
            corr_cfg["mu_min"],
            corr_cfg["mu_max"] + corr_cfg["mu_bin"],
            corr_cfg["mu_bin"],
        )

        counts = compute_pair_counts(
            data_xyz,
            rand_xyz,
            s_edges=s_edges,
            mu_edges=mu_edges,
            data_weights=data_cat.w,
            rand_weights=rand_cat.w,
        )
        counts_list.append(counts)

        all_env.append(env.per_galaxy)
        data_xyz_all.append(data_xyz)
        rand_xyz_all.append(rand_xyz)
        data_ra_all.append(data_cat.ra)
        data_dec_all.append(data_cat.dec)
        rand_ra_all.append(rand_cat.ra)
        rand_dec_all.append(rand_cat.dec)
        data_w_all.append(data_cat.w)
        rand_w_all.append(rand_cat.w)

    combined_counts = combine_pair_counts(counts_list)
    xi = landy_szalay(combined_counts)

    wedges = prereg["correlation"]["wedges"]
    tangential_bounds = parse_wedge_bounds(wedges["tangential"])
    tangential = wedge_xi(xi, mu_edges, tangential_bounds)

    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    env_bins = np.asarray(prereg["environment_binning"]["quantile_edges"])
    env_values = np.concatenate(all_env)
    env_tags = bin_by_environment(env_values, env_bins)
    env_means = np.array([env_values[env_tags == i].mean() for i in range(len(env_bins) - 1)])

    template_params = {
        "r_d": prereg["fiducial_cosmology"]["r_d"],
        "sigma_nl": prereg["reconstruction"]["parameters"]["smoothing"],
        "omega_m": prereg["fiducial_cosmology"]["omega_m"],
        "omega_b": prereg["fiducial_cosmology"]["omega_b"],
        "h": prereg["fiducial_cosmology"]["h"],
        "n_s": prereg["fiducial_cosmology"]["n_s"],
    }

    cov_cfg = prereg["covariance"]
    cov_dir = output_dir / "covariance"
    cov_dir.mkdir(parents=True, exist_ok=True)
    mock_cov_path = cov_dir / "mock_xi_wedges.npy"
    jk_cfg = cov_cfg["jackknife"]

    if cov_cfg["primary_method"] == "mocks" and mock_cov_path.exists():
        mock_wedges = np.load(mock_cov_path)
        cov_result = covariance_from_mocks(mock_wedges)
    else:
        data_xyz_all = np.vstack(data_xyz_all)
        rand_xyz_all = np.vstack(rand_xyz_all)
        data_ra_all = np.concatenate(data_ra_all)
        data_dec_all = np.concatenate(data_dec_all)
        rand_ra_all = np.concatenate(rand_ra_all)
        rand_dec_all = np.concatenate(rand_dec_all)
        data_w_all = np.concatenate(data_w_all)
        rand_w_all = np.concatenate(rand_w_all)

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

        jk_wedges = []
        for idx in range(jk_cfg["n_regions"]):
            data_mask = jk_data_regions != idx
            rand_mask = jk_rand_regions != idx
            jk_counts = compute_pair_counts(
                data_xyz_all[data_mask],
                rand_xyz_all[rand_mask],
                s_edges=s_edges,
                mu_edges=mu_edges,
                data_weights=data_w_all[data_mask],
                rand_weights=rand_w_all[rand_mask],
                verbose=False,
            )
            jk_xi = landy_szalay(jk_counts)
            jk_wedge = wedge_xi(jk_xi, mu_edges, tangential_bounds)
            jk_wedges.append(jk_wedge)

        jk_wedges = np.asarray(jk_wedges)
        cov_result = covariance_from_jackknife(jk_wedges)

    np.save(cov_dir / "xi_wedge_covariance.npy", cov_result.covariance)

    fit_cfg = prereg["bao_fitting"]
    fit_range = (fit_cfg["fit_range"]["s_min"], fit_cfg["fit_range"]["s_max"])
    nuisance_terms = fit_cfg["nuisance"]["terms"]
    fit_result = fit_wedge(
        s=s_centers,
        xi=tangential,
        covariance=cov_result.covariance,
        fit_range=fit_range,
        nuisance_terms=nuisance_terms,
        template_params=template_params,
    )

    alpha_bins = np.full_like(env_means, fill_value=fit_result.alpha)
    alpha_cov = np.eye(len(env_means)) * (fit_result.sigma_alpha**2)

    hier_cfg = prereg["hierarchical_inference"]
    bayes_cfg = hier_cfg["bayesian_option"]
    if bayes_cfg["enabled"]:
        beta_result = bayesian_beta(env_means, alpha_bins, alpha_cov, bayes_cfg["priors"]["beta_sigma"])
    else:
        beta_result = two_step_beta(env_means, alpha_bins, alpha_cov)

    results = {
        "alpha_bins": alpha_bins.tolist(),
        "beta": beta_result.beta,
        "sigma_beta": beta_result.sigma_beta,
    }

    blinding_cfg = cfg.get("blinding", {})
    blind_state = BlindState(
        unblind=blinding_cfg.get("unblind", False),
        key_file=prereg["blinding"]["encryption"]["key_file"],
    )

    save_metadata(output_dir / "metadata.json", {"seed": seed, "catalog": cfg["catalog"]})
    np.savez(
        output_dir / "pair_counts.npz",
        dd=combined_counts.dd,
        dr=combined_counts.dr,
        rr=combined_counts.rr,
    )
    np.savez(output_dir / "xi_wedge.npz", s=s_centers, xi=tangential)

    plot_wedge(s_centers, tangential, output_dir / "figures" / "xi_tangential.png", "Tangential wedge")

    prereg_hash = compute_prereg_hash(Path(cfg["preregistration"]))
    (output_dir / "prereg_hash.txt").write_text(prereg_hash, encoding="utf-8")
    if blinding_cfg.get("enabled", True) and not blind_state.unblind:
        blind_state = initialize_blinding(blind_state, rng)
        blinded = blind_results(
            beta=beta_result.beta,
            sigma_beta=beta_result.sigma_beta,
            state=blind_state,
            prereg_hash=prereg_hash,
        )
        save_blinded_results(blinded, output_dir / "blinded_results.json")
    else:
        write_results(output_dir / "results.json", results, blind_state)

    if prereg["reporting"]["generate_methods_snapshot"]:
        write_methods_snapshot(cfg, output_dir / "methods_snapshot.yaml")

    mock_betas = rng.normal(loc=0.0, scale=beta_result.sigma_beta, size=prereg["mocks"]["n_null_distribution"])
    plot_beta_null(mock_betas, output_dir / "figures" / "beta_null.png", beta_obs=None)

    paper_package = output_dir / "paper_package"
    paper_package.mkdir(parents=True, exist_ok=True)
    (paper_package / "figures").mkdir(parents=True, exist_ok=True)
    for src, dest in [
        (output_dir / "metadata.json", paper_package / "metadata.json"),
        (output_dir / "xi_wedge.npz", paper_package / "xi_wedge.npz"),
        (output_dir / "figures" / "xi_tangential.png", paper_package / "figures" / "xi_tangential.png"),
        (cov_dir / "xi_wedge_covariance.npy", paper_package / "xi_wedge_covariance.npy"),
        (output_dir / "methods_snapshot.yaml", paper_package / "methods_snapshot.yaml"),
        (output_dir / "prereg_hash.txt", paper_package / "prereg_hash.txt"),
    ]:
        if src.exists():
            dest.write_bytes(src.read_bytes())
    if (output_dir / "blinded_results.json").exists():
        (paper_package / "blinded_results.json").write_bytes((output_dir / "blinded_results.json").read_bytes())
    if (output_dir / "results.json").exists():
        (paper_package / "results.json").write_bytes((output_dir / "results.json").read_bytes())

    with open(output_dir / "stage_summary.json", "w", encoding="utf-8") as handle:
        json.dump({"stages": cfg["stages"]}, handle, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run_pipeline(args.config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
