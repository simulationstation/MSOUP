#!/usr/bin/env python
"""Run full BAO overlap pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bao_overlap.blinding import BlindState
from bao_overlap.correlation import compute_pair_counts, landy_szalay, wedge_xi, bin_by_environment
from bao_overlap.covariance import covariance_from_mocks
from bao_overlap.density_field import build_density_field, gaussian_smooth
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

    seed = prereg["analysis"]["random_seed"]
    rng = np.random.default_rng(seed)

    dry_run_fraction = cfg["runtime"].get("dry_run_fraction") if dry_run else None
    data_cat, rand_cat = load_catalog(
        datasets_cfg=datasets,
        catalog_key=cfg["catalog"],
        dry_run_fraction=dry_run_fraction,
        seed=seed,
    )

    cosmo_cfg = prereg["analysis"]["fiducial_cosmology"]
    data_xyz = radec_to_cartesian(data_cat.ra, data_cat.dec, data_cat.z, **cosmo_cfg)
    rand_xyz = radec_to_cartesian(rand_cat.ra, rand_cat.dec, rand_cat.z, **cosmo_cfg)

    density = build_density_field(data_xyz, rand_xyz, rand_cat.w, grid_size=64, cell_size=5.0)
    smooth = gaussian_smooth(density, radius=prereg["analysis"]["overlap_metric"]["smoothing_radii"][0])

    pair_indices = rng.choice(len(data_xyz), size=min(200, len(data_xyz)), replace=False)
    pairs = np.stack([data_xyz[pair_indices], data_xyz[pair_indices[::-1]]], axis=1)

    overlap_cfg = prereg["analysis"]["overlap_metric"]
    env = compute_environment(
        field=smooth,
        galaxy_xyz=data_xyz,
        pair_xyz=pairs,
        step=overlap_cfg["line_integral_step"],
        rng=rng,
        subsample=overlap_cfg["pair_subsample"],
        delta_threshold=1.0,
        min_volume=10,
        normalize_output=overlap_cfg["normalize"],
        primary=overlap_cfg["primary"],
    )

    s_edges = np.arange(
        prereg["analysis"]["correlation"]["s_min"],
        prereg["analysis"]["correlation"]["s_max"] + prereg["analysis"]["correlation"]["s_bin"],
        prereg["analysis"]["correlation"]["s_bin"],
    )
    mu_edges = np.arange(
        prereg["analysis"]["correlation"]["mu_min"],
        prereg["analysis"]["correlation"]["mu_max"] + prereg["analysis"]["correlation"]["mu_bin"],
        prereg["analysis"]["correlation"]["mu_bin"],
    )

    counts = compute_pair_counts(data_xyz, rand_xyz, s_edges=s_edges, mu_edges=mu_edges)
    xi = landy_szalay(counts)

    wedges = prereg["analysis"]["correlation"]["wedges"]
    tangential = wedge_xi(xi, mu_edges, tuple(wedges["tangential"]))

    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    env_bins = prereg["analysis"]["environment_binning"]["quantiles"]
    env_tags = bin_by_environment(env.per_galaxy, np.asarray(env_bins))
    env_means = np.array([env.per_galaxy[env_tags == i].mean() for i in range(len(env_bins) - 1)])

    alpha_bins = np.full_like(env_means, fill_value=1.0)
    alpha_cov = np.eye(len(env_means)) * 0.01

    hier_cfg = prereg["analysis"]["hierarchical_inference"]
    if hier_cfg["bayesian"]:
        beta_result = bayesian_beta(env_means, alpha_bins, alpha_cov, hier_cfg["priors"]["beta_sigma"])
    else:
        beta_result = two_step_beta(env_means, alpha_bins, alpha_cov)

    results = {
        "alpha_bins": alpha_bins.tolist(),
        "beta": beta_result.beta,
        "beta_sigma": beta_result.sigma_beta,
    }

    blind_state = BlindState(unblind=cfg.get("blinding", {}).get("unblind", False))

    save_metadata(output_dir / "metadata.json", {"seed": seed, "catalog": cfg["catalog"]})
    np.savez(output_dir / "pair_counts.npz", dd=counts.dd, dr=counts.dr, rr=counts.rr)
    np.savez(output_dir / "xi_wedge.npz", s=s_centers, xi=tangential)

    plot_wedge(s_centers, tangential, output_dir / "figures" / "xi_tangential.png", "Tangential wedge")

    write_results(output_dir / "results.json", results, blind_state)
    if prereg["analysis"]["reporting"]["generate_methods_snapshot"]:
        write_methods_snapshot(cfg, output_dir / "methods_snapshot.yaml")

    mock_betas = rng.normal(loc=0.0, scale=beta_result.sigma_beta, size=prereg["analysis"]["mocks"]["n_mocks"])
    plot_beta_null(mock_betas, output_dir / "figures" / "beta_null.png", beta_obs=None)

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
