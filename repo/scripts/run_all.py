"""Run full MSoup neutrality experiment suite."""
from __future__ import annotations

import argparse
from pathlib import Path

from msoup.experiments import ExperimentConfig, run_experiments
from msoup.report import write_report


def build_config(heavy: bool) -> ExperimentConfig:
    if heavy:
        return ExperimentConfig(
            sizes=(128, 256, 512),
            j_over_t_grid=tuple(__import__("numpy").logspace(-2, 2, 25)),
            n_samples=12000,
            coarse_pairs=10,
        )
    return ExperimentConfig()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MSoup neutrality experiments")
    parser.add_argument("--heavy", action="store_true", help="Run heavier sweeps")
    args = parser.parse_args()

    config = build_config(args.heavy)
    outputs = run_experiments(config)
    write_report(outputs, Path("outputs"))


if __name__ == "__main__":
    main()
