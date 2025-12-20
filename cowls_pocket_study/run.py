"""
CLI orchestration for the JWST COWLS pocket-domain study.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import candidates, io, preprocess, qc, report, sensitivity, stats_real


def run_pipeline(
    subset: List[str],
    band: str,
    out: Path,
    data_root: Path,
    score_bins: Optional[List[str]] = None,
    n_resamples: int = 200,
    theta0: float = 0.30,
) -> Dict:
    """Run the pocket-domain clustering analysis pipeline."""
    out.mkdir(parents=True, exist_ok=True)

    print(f"Discovering lenses in {data_root}...")
    lenses = io.discover_lenses(
        data_root=data_root,
        subset=subset if subset else None,
        score_bins=score_bins,
    )
    print(f"Found {len(lenses)} lenses")

    if not lenses:
        raise RuntimeError("No lenses found! Check data_root path.")

    preprocess_map: Dict[str, preprocess.PreprocessResult] = {}
    candidate_map: Dict[str, candidates.CandidateResult] = {}
    windows: Dict[str, sensitivity.SensitivityWindowReal] = {}
    kept: List[str] = []
    errors: List[str] = []

    for i, lens in enumerate(lenses):
        print(f"[{i+1}/{len(lenses)}] Processing {lens.lens_id} ({lens.score_bin})...")
        try:
            band_used, image, noise, _ = io.load_lens_data(lens, band=band)
            print(f"  Using band {band_used}, image shape {image.shape}")
        except Exception as e:
            errors.append(f"{lens.lens_id}: load failed - {e}")
            continue

        try:
            prep = preprocess.build_arc_mask(image, noise, lens)
            preprocess_map[lens.lens_id] = prep
            print(f"  Arc mask: {np.sum(prep.arc_mask)} pixels")
        except Exception as e:
            errors.append(f"{lens.lens_id}: preprocess failed - {e}")
            continue

        try:
            window = sensitivity.load_or_compute_sensitivity(
                snr_map=prep.snr_map,
                mask=prep.arc_mask,
                center=prep.center,
                cache_root=out,
                lens_id=lens.lens_id,
            )
            windows[lens.lens_id] = window
        except Exception as e:
            errors.append(f"{lens.lens_id}: sensitivity failed - {e}")
            continue

        try:
            cand = candidates.detect_candidates(
                image=image,
                noise=noise,
                preprocess=prep,
                cache_root=out,
                lens_id=lens.lens_id,
                model_products=lens.model_products,
            )
            candidate_map[lens.lens_id] = cand
            print(f"  Found {cand.n_candidates} candidates")
        except Exception as e:
            errors.append(f"{lens.lens_id}: candidate detection failed - {e}")
            continue

        decision = qc.evaluate_qc(prep, cand)
        if decision.keep:
            kept.append(lens.lens_id)

    print(f"\nKept {len(kept)}/{len(lenses)} lenses after QC")

    if not kept:
        print("WARNING: No lenses passed QC!")
        # Still generate a report showing the failure
        aggregate = None
    else:
        theta_by_lens = {lid: candidate_map[lid].theta for lid in kept}
        aggregate = stats_real.aggregate_clustering(
            theta_by_lens, windows, theta0=theta0, n_resamples=n_resamples
        )

    report.build_report(out, kept, aggregate, preprocess_map, candidate_map, windows, errors=errors)

    return {
        "n_lenses": len(lenses),
        "n_kept": len(kept),
        "aggregate": aggregate,
        "errors": errors,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="JWST COWLS pocket-domain study")
    parser.add_argument("--subset", type=str, default="", help="Comma-separated lens IDs (default: all)")
    parser.add_argument("--score-bins", type=str, default="", help="Comma-separated score bins (e.g., M25,S12,S11)")
    parser.add_argument("--band", type=str, default="auto", help="Band to analyze or 'auto'")
    parser.add_argument("--out", type=str, default="results/cowls_pocket_study/", help="Output directory")
    parser.add_argument("--data-root", type=str, default=str(io.DEFAULT_DATA_ROOT), help="Data root")
    parser.add_argument("--n-resamples", type=int, default=200, help="Number of null resamples")
    parser.add_argument("--theta0", type=float, default=0.30, help="Clustering threshold (radians)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    subset = [s.strip() for s in args.subset.split(",") if s.strip()] if args.subset else []
    score_bins = [s.strip() for s in args.score_bins.split(",") if s.strip()] if args.score_bins else None

    try:
        run_pipeline(
            subset=subset,
            band=args.band,
            out=Path(args.out),
            data_root=Path(args.data_root),
            score_bins=score_bins,
            n_resamples=args.n_resamples,
            theta0=args.theta0,
        )
    except Exception as e:
        # Write failure log
        failure_log = Path(args.out) / "FAILURE_LOG.md"
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        with open(failure_log, "w") as f:
            f.write(f"# Pipeline Failure\n\n")
            f.write(f"**Error:** {e}\n\n")
            f.write(f"**Traceback:**\n```\n{traceback.format_exc()}\n```\n")
        raise


if __name__ == "__main__":
    main()
