"""
CLI orchestration for the JWST COWLS pocket-domain study.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from . import candidates, io, preprocess, qc, report, sensitivity, stats_real


def run_pipeline(subset: List[str], band: str, out: Path, data_root: Path) -> None:
    lenses = io.discover_lenses(data_root=data_root, subset=subset if subset else None)
    preprocess_map: Dict[str, preprocess.PreprocessResult] = {}
    candidate_map: Dict[str, candidates.CandidateResult] = {}
    windows: Dict[str, sensitivity.SensitivityWindowReal] = {}
    kept: List[str] = []

    for lens in tqdm(lenses, desc="Processing lenses"):
        try:
            band_used, image, noise, _ = io.load_lens_data(lens, band=band)
        except Exception:
            continue

        prep = preprocess.build_arc_mask(image, noise, lens)
        preprocess_map[lens.lens_id] = prep

        window = sensitivity.load_or_compute_sensitivity(
            snr_map=prep.snr_map,
            mask=prep.arc_mask,
            center=prep.center,
            cache_root=out,
            lens_id=lens.lens_id,
        )
        windows[lens.lens_id] = window

        cand = candidates.detect_candidates(
            image=image,
            noise=noise,
            preprocess=prep,
            cache_root=out,
            lens_id=lens.lens_id,
            model_products=lens.model_products,
        )
        candidate_map[lens.lens_id] = cand

        decision = qc.evaluate_qc(prep, cand)
        if decision.keep:
            kept.append(lens.lens_id)

    theta_by_lens = {lid: candidate_map[lid].theta for lid in kept}
    aggregate = stats_real.aggregate_clustering(theta_by_lens, windows)

    report.build_report(out, kept, aggregate, preprocess_map, candidate_map, windows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JWST COWLS pocket-domain study")
    parser.add_argument("--subset", type=str, default="", help="Comma-separated lens IDs (default: all)")
    parser.add_argument("--band", type=str, default="auto", help="Band to analyze or 'auto'")
    parser.add_argument("--out", type=str, default="results/cowls_pocket_study/", help="Output directory")
    parser.add_argument("--data-root", type=str, default=str(io.DEFAULT_DATA_ROOT), help="Data root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = [s.strip() for s in args.subset.split(",") if s.strip()] if args.subset else []
    run_pipeline(subset=subset, band=args.band, out=Path(args.out), data_root=Path(args.data_root))


if __name__ == "__main__":
    main()
