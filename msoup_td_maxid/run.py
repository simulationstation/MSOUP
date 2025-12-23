from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path

from .config import MaxIDConfig, load_config
from .inference import SUPPORTED_MODES, run_inference
from .report import write_report


def parse_args():
    parser = argparse.ArgumentParser(description="TD-only maximally identifying inference")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", type=str, choices=list(SUPPORTED_MODES) + ["all"], default="base", help="Inference mode")
    parser.add_argument("--write-loo", type=str, choices=["true", "false"], default="true", help="Compute exact leave-one-out refits")
    parser.add_argument("--max-workers", type=int, default=1, help="Maximum workers (set >1 at your own risk on memory-limited systems)")
    return parser.parse_args()


def _resolve_output_dir(cfg: MaxIDConfig) -> Path:
    base = Path(cfg.paths.output_dir)
    if base.name == "td_maxid":
        base = base.parent / "msoup_td_maxid"
    default_base = Path("results") / "msoup_td_maxid"
    if base == Path("results") / "td_maxid":
        base = default_base
    timestamp_pattern = re.compile(r"^[0-9]{8}_[0-9]{6}$")
    if timestamp_pattern.match(base.name):
        return base
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    return base / timestamp


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg.compute.max_workers = args.max_workers
    write_loo = args.write_loo.lower() != "false"
    output_dir = _resolve_output_dir(cfg)
    cfg.paths.output_dir = output_dir
    modes = list(SUPPORTED_MODES) if args.mode == "all" else [args.mode]
    for mode in modes:
        result = run_inference(cfg, mode=mode, write_loo=write_loo)
        out_dir = Path(cfg.paths.output_dir) / mode
        write_report(cfg, result, out_dir)


if __name__ == "__main__":
    main()
