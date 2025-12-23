#!/usr/bin/env python
"""Create a preregistration PDF snapshot from YAML."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as handle:
        content = yaml.safe_dump(yaml.safe_load(handle), sort_keys=True)

    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    ax.text(0.01, 0.99, content, va="top", family="monospace", fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
