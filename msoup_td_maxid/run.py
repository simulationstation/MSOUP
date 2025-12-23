from __future__ import annotations

import argparse
from pathlib import Path

from .config import MaxIDConfig, load_config
from .inference import SUPPORTED_MODES, run_inference
from .report import write_report


def parse_args():
    parser = argparse.ArgumentParser(description="TD-only maximally identifying inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, choices=list(SUPPORTED_MODES) + ["all"], default="base", help="Inference mode")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    modes = list(SUPPORTED_MODES) if args.mode == "all" else [args.mode]
    for mode in modes:
        result = run_inference(cfg, mode=mode)
        out_dir = Path(cfg.paths.output_dir) / mode
        write_report(cfg, result, out_dir)


if __name__ == "__main__":
    main()
