#!/usr/bin/env python
"""Run density grid diagnostics and append results to AUDIT.md."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np

from bao_overlap.density_field import (
    build_density_field,
    build_grid_spec,
    gaussian_smooth,
    sample_with_mask,
)
from bao_overlap.geometry import radec_to_cartesian
from bao_overlap.io import load_catalog, load_run_config
from bao_overlap.overlap_metric import compute_per_galaxy_mean_e1


def _grid_params(prereg: Dict[str, Any]) -> Dict[str, float | int]:
    env_primary = prereg["environment_metric"]["primary"]
    env_params = env_primary["parameters"]
    grid_cfg = env_primary.get("grid", {})
    smoothing_radius = float(env_params["smoothing_radius"])
    target_cell_size = float(env_params.get("target_cell_size", grid_cfg.get("target_cell_size", 10.0)))
    padding = float(env_params.get("padding", grid_cfg.get("padding", max(3.0 * smoothing_radius, 50.0))))
    max_n_per_axis = int(env_params.get("max_n_per_axis", grid_cfg.get("max_n_per_axis", 512)))
    return {
        "target_cell_size": target_cell_size,
        "padding": padding,
        "max_n_per_axis": max_n_per_axis,
    }


def _append_audit(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(text)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--region", default=None)
    parser.add_argument("--dry-run-fraction", type=float, default=None)
    parser.add_argument("--sample-galaxies", type=int, default=200)
    parser.add_argument("--e1-galaxies", type=int, default=100)
    parser.add_argument("--max-pairs", type=int, default=20)
    args = parser.parse_args()

    cfg = load_run_config(args.config)
    prereg = cfg["_preregistration"]
    datasets = cfg["_datasets"]

    env_primary = prereg["environment_metric"]["primary"]
    env_params = env_primary["parameters"]
    smoothing_radius = float(env_params["smoothing_radius"])
    line_step = float(env_params["line_integral_step"])
    pair_subsample_fraction = float(env_params["pair_subsample_fraction"])

    grid_params = _grid_params(prereg)
    regions = prereg["primary_dataset"]["regions"]
    if args.region is not None:
        regions = [args.region]

    dry_run_fraction = args.dry_run_fraction
    if dry_run_fraction is None:
        dry_run_fraction = cfg.get("runtime", {}).get("dry_run_fraction")

    rng = np.random.default_rng(prereg["random_seed"])
    cosmo_cfg = prereg["fiducial_cosmology"]

    audit_lines = []
    audit_lines.append("\n## Grid Diagnostic\n")
    audit_lines.append(f"Timestamp: {datetime.utcnow().isoformat()}Z\n")
    audit_lines.append(f"Config: {args.config}\n")
    audit_lines.append(
        "Grid params: target_cell_size={target_cell_size}, padding={padding}, max_n_per_axis={max_n_per_axis}\n".format(
            **grid_params
        )
    )

    try:
        for region in regions:
            data_cat, rand_cat = load_catalog(
                datasets_cfg=datasets,
                catalog_key=cfg["catalog"],
                region=region,
                dry_run_fraction=dry_run_fraction,
                seed=prereg["random_seed"],
            )
            data_xyz = radec_to_cartesian(data_cat.ra, data_cat.dec, data_cat.z, **cosmo_cfg)
            rand_xyz = radec_to_cartesian(rand_cat.ra, rand_cat.dec, rand_cat.z, **cosmo_cfg)

            span = np.max(np.vstack([data_xyz, rand_xyz]), axis=0) - np.min(
                np.vstack([data_xyz, rand_xyz]), axis=0
            )
            grid_spec = build_grid_spec(
                data_xyz=data_xyz,
                random_xyz=rand_xyz,
                target_cell_size=grid_params["target_cell_size"],
                padding=grid_params["padding"],
                max_n_per_axis=grid_params["max_n_per_axis"],
            )
            coverage = grid_spec.cell_sizes * np.array(grid_spec.grid_shape)
            n_cells = int(np.prod(grid_spec.grid_shape))
            mem_gb = n_cells * 4 / (1024**3)

            density = build_density_field(
                data_xyz=data_xyz,
                random_xyz=rand_xyz,
                data_weight=data_cat.w,
                random_weight=rand_cat.w,
                grid_spec=grid_spec,
            )
            smooth = gaussian_smooth(density, radius=smoothing_radius)

            n_sample = min(args.sample_galaxies, len(data_xyz))
            sample_idx = rng.choice(len(data_xyz), size=n_sample, replace=False)
            _, inside = sample_with_mask(smooth, data_xyz[sample_idx])
            invalid_frac = float(1.0 - np.mean(inside))

            n_e1 = min(args.e1_galaxies, len(data_xyz))
            e1_idx = rng.choice(len(data_xyz), size=n_e1, replace=False)
            e1_vals, meta = compute_per_galaxy_mean_e1(
                field=smooth,
                galaxy_xyz=data_xyz[e1_idx],
                s_min=prereg["bao_fitting"]["fit_range"]["s_min"],
                s_max=prereg["bao_fitting"]["fit_range"]["s_max"],
                step=line_step,
                rng=rng,
                pair_subsample_fraction=pair_subsample_fraction,
                max_outside_fraction=0.2,
                max_pairs_per_galaxy=args.max_pairs,
            )
            attempted = meta["attempted_pairs"]
            valid = meta["valid_pairs"]
            invalid = meta["invalid_pairs"]
            mean_attempted = float(np.mean(attempted)) if len(attempted) else 0.0
            mean_valid = float(np.mean(valid)) if len(valid) else 0.0
            invalid_pairs_fraction = float(np.sum(invalid) / max(np.sum(attempted), 1))
            finite_fraction = float(np.mean(np.isfinite(e1_vals))) if len(e1_vals) else 0.0

            audit_lines.append(f"\nRegion: {region}\n")
            audit_lines.append(f"Span (h^-1 Mpc): {span.tolist()}\n")
            audit_lines.append(f"Grid shape: {grid_spec.grid_shape}\n")
            audit_lines.append(f"Cell sizes (h^-1 Mpc): {grid_spec.cell_sizes.tolist()}\n")
            audit_lines.append(f"Coverage (h^-1 Mpc): {coverage.tolist()}\n")
            audit_lines.append(f"Estimated grid memory (GB): {mem_gb:.3f}\n")
            audit_lines.append(f"Random sample invalid fraction: {invalid_frac:.3f}\n")
            audit_lines.append(
                "E1 diagnostic: mean attempted pairs per galaxy={:.2f}, mean valid pairs={:.2f}, invalid fraction={:.3f}, finite fraction={:.3f}\n".format(
                    mean_attempted, mean_valid, invalid_pairs_fraction, finite_fraction
                )
            )
    except Exception as exc:  # pylint: disable=broad-except
        audit_lines.append("\nGrid diagnostic failed with exception:\n")
        audit_lines.append(f"{type(exc).__name__}: {exc}\n")

    _append_audit(Path(__file__).resolve().parents[1] / "AUDIT.md", "".join(audit_lines))


if __name__ == "__main__":
    main()
