"""Run a minimal power scan."""

from pathlib import Path

from mverse_channel.config import SweepConfig
from mverse_channel.sim.sweeps import power_sweep


def main() -> None:
    sweep_config = SweepConfig()
    power_sweep(sweep_config, Path("outputs/mverse_sweep"))


if __name__ == "__main__":
    main()
