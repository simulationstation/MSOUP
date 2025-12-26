"""Run MSoup experiments or the m3alpha universality harness."""
from __future__ import annotations

import argparse
from pathlib import Path

from msoup.experiments import ExperimentConfig, run_experiments
from msoup.report import write_report

from m3alpha.rg_bkt import RGConfig, run_rg_scan
from m3alpha.report import write_report as write_m3alpha_report
from m3alpha.xy_mc import MCConfig, run_mc


def build_config(heavy: bool) -> ExperimentConfig:
    if heavy:
        return ExperimentConfig(
            sizes=(128, 256, 512),
            j_over_t_grid=tuple(__import__("numpy").logspace(-2, 2, 25)),
            n_samples=12000,
            coarse_pairs=10,
        )
    return ExperimentConfig()


def build_m3alpha_configs(heavy: bool) -> tuple[RGConfig, MCConfig]:
    if heavy:
        mc_config = MCConfig(
            lattice_sizes=(16, 32, 64),
            temperatures=tuple(__import__("numpy").linspace(0.7, 1.1, 30)),
            thermalization_sweeps=1200,
            measurement_sweeps=2000,
            sweep_stride=5,
        )
    else:
        mc_config = MCConfig(
            lattice_sizes=(16, 32, 64),
            temperatures=tuple(__import__("numpy").linspace(0.7, 1.1, 20)),
            thermalization_sweeps=500,
            measurement_sweeps=800,
            sweep_stride=10,
        )
    rg_config = RGConfig()
    return rg_config, mc_config


def run_m3alpha(heavy: bool, output_dir: Path) -> None:
    rg_config, mc_config = build_m3alpha_configs(heavy)
    rg_result = run_rg_scan(rg_config)
    mc_result = run_mc(mc_config)
    write_m3alpha_report(rg_result, mc_result, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MSoup or m3alpha harness")
    parser.add_argument("--heavy", action="store_true", help="Run heavier sweeps")
    parser.add_argument(
        "--m3alpha",
        action="store_true",
        help="Run m3alpha universality harness (RG + MC)",
    )
    args = parser.parse_args()

    if args.m3alpha:
        run_m3alpha(args.heavy, Path("outputs"))
        return

    config = build_config(args.heavy)
    outputs = run_experiments(config)
    write_report(outputs, Path("outputs"))


if __name__ == "__main__":
    main()
