#!/usr/bin/env python
"""Run a single pipeline stage for debugging."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bao_overlap.blinding import BlindState
from bao_overlap.correlation import compute_pair_counts, landy_szalay
from bao_overlap.density_field import build_density_field, gaussian_smooth
from bao_overlap.geometry import radec_to_cartesian
from bao_overlap.io import load_run_config, load_catalog
from bao_overlap.overlap_metric import compute_environment
from bao_overlap.reporting import write_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    prereg = cfg["_preregistration"]
    datasets = cfg["_datasets"]
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    seed = prereg["analysis"]["random_seed"]
    dry_run_fraction = cfg["runtime"].get("dry_run_fraction") if args.dry_run else None

    data_cat, rand_cat = load_catalog(
        datasets_cfg=datasets,
        catalog_key=cfg["catalog"],
        dry_run_fraction=dry_run_fraction,
        seed=seed,
    )

    cosmo_cfg = prereg["analysis"]["fiducial_cosmology"]
    data_xyz = radec_to_cartesian(data_cat.ra, data_cat.dec, data_cat.z, **cosmo_cfg)
    rand_xyz = radec_to_cartesian(rand_cat.ra, rand_cat.dec, rand_cat.z, **cosmo_cfg)

    if args.stage == "io":
        np.savez(output_dir / "io_catalogs.npz", data=data_xyz, randoms=rand_xyz)
        return

    if args.stage == "overlap_metric":
        density = build_density_field(data_xyz, rand_xyz, rand_cat.w, grid_size=64, cell_size=5.0)
        smooth = gaussian_smooth(density, radius=prereg["analysis"]["overlap_metric"]["smoothing_radii"][0])
        rng = np.random.default_rng(seed)
        pair_indices = rng.choice(len(data_xyz), size=min(200, len(data_xyz)), replace=False)
        pairs = np.stack([data_xyz[pair_indices], data_xyz[pair_indices[::-1]]], axis=1)
        env = compute_environment(
            field=smooth,
            galaxy_xyz=data_xyz,
            pair_xyz=pairs,
            step=prereg["analysis"]["overlap_metric"]["line_integral_step"],
            rng=rng,
            subsample=prereg["analysis"]["overlap_metric"]["pair_subsample"],
            delta_threshold=1.0,
            min_volume=10,
            normalize_output=prereg["analysis"]["overlap_metric"]["normalize"],
            primary=prereg["analysis"]["overlap_metric"]["primary"],
        )
        np.savez(output_dir / "environment.npz", per_galaxy=env.per_galaxy, per_pair=env.per_pair)
        return

    if args.stage == "correlation":
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
        np.savez(output_dir / "correlation.npz", xi=xi)
        return

    if args.stage == "reporting":
        results = {"status": "placeholder"}
        blind_state = BlindState(unblind=cfg.get("blinding", {}).get("unblind", False))
        write_results(output_dir / "results.json", results, blind_state)
        return

    raise ValueError(f"Unknown stage: {args.stage}")


if __name__ == "__main__":
    main()
