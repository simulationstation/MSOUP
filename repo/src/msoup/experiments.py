"""Experiment suite for neutrality tests."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .constraints import build_constraint_pair
from .coarsegrain import coarsegrain_blocks, coarsegrain_random
from .entropy import delta_entropy
from .neutrality import estimate_mu_over_t


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 123
    sizes: tuple[int, ...] = (128, 256)
    densities: tuple[float, ...] = (0.5, 0.8, 1.0, 1.2)
    delta_m: int = 5
    density_matrix: float = 0.2
    j_over_t_grid: tuple[float, ...] = tuple(np.logspace(-2, 2, 13))
    n_samples: int = 4096
    coarse_block_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    coarse_pairs: int = 6
    coarse_density: float = 0.3


@dataclass(frozen=True)
class ExperimentOutputs:
    config: dict
    entropy_checks: list[dict]
    dual_sweep: list[dict]
    coarsegrain: dict
    combined: dict


def run_experiments(config: ExperimentConfig) -> ExperimentOutputs:
    rng = np.random.default_rng(config.seed)
    entropy_checks = []
    for n in config.sizes:
        pair = build_constraint_pair(
            rng,
            n=n,
            m_base=int(0.8 * n),
            delta_m=config.delta_m,
            density=config.density_matrix,
        )
        delta_s = delta_entropy(pair.peeled, pair.pinned)
        entropy_checks.append(
            {
                "n": n,
                "delta_m": pair.delta_m,
                "delta_s": float(delta_s),
            }
        )

    dual_sweep = []
    for n in config.sizes:
        for density in config.densities:
            pair = build_constraint_pair(
                rng,
                n=n,
                m_base=int(density * n),
                delta_m=config.delta_m,
                density=config.density_matrix,
            )
            for j_over_t in config.j_over_t_grid:
                estimate = estimate_mu_over_t(
                    pair,
                    j_over_t=j_over_t,
                    rng=rng,
                    n_samples=config.n_samples,
                )
                dual_sweep.append(
                    {
                        "n": n,
                        "density": density,
                        "j_over_t": float(j_over_t),
                        "mu_over_t": float(estimate.mu_over_t),
                        "delta_m": pair.delta_m,
                    }
                )

    coarsegrain_records = {"cg1": [], "cg2": []}
    for _ in range(config.coarse_pairs):
        pair = build_constraint_pair(
            rng,
            n=max(config.sizes),
            m_base=int(0.8 * max(config.sizes)),
            delta_m=config.delta_m,
            density=config.coarse_density,
        )
        results_cg1 = coarsegrain_blocks(pair, config.coarse_block_sizes)
        results_cg2 = coarsegrain_random(pair, config.coarse_block_sizes, rng=rng)
        coarsegrain_records["cg1"].append([asdict(r) for r in results_cg1])
        coarsegrain_records["cg2"].append([asdict(r) for r in results_cg2])

    combined = _combine_metrics(dual_sweep, coarsegrain_records)

    return ExperimentOutputs(
        config=asdict(config),
        entropy_checks=entropy_checks,
        dual_sweep=dual_sweep,
        coarsegrain=coarsegrain_records,
        combined=combined,
    )


def _combine_metrics(dual_sweep: list[dict], coarsegrain: dict) -> dict:
    ln2 = float(np.log(2))
    tolerance = 0.1
    mu_hits = [
        entry
        for entry in dual_sweep
        if abs(entry["mu_over_t"] - ln2) < tolerance
    ]
    mu_hit_rate = len(mu_hits) / max(len(dual_sweep), 1)

    def cg_hits(records: list[list[dict]]) -> float:
        hits = 0
        total = 0
        for record in records:
            deltas = [entry["delta_m"] for entry in record]
            if deltas and max(deltas) <= 5:
                hits += 1
            total += 1
        return hits / max(total, 1)

    cg1_rate = cg_hits(coarsegrain["cg1"])
    cg2_rate = cg_hits(coarsegrain["cg2"])

    return {
        "mu_hit_rate": mu_hit_rate,
        "cg1_rate": cg1_rate,
        "cg2_rate": cg2_rate,
        "ln2": ln2,
        "tolerance": tolerance,
    }
