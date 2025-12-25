"""Decision harness for MSoup/Mverse neutrality program."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .coarsegrain import coarsegrain_delta_m
from .gf2 import gf2_matmul_batch, gf2_rank
from .xor_model import generate_constraint_pair


@dataclass(frozen=True)
class HarnessConfig:
    n_values: tuple[int, ...] = (128, 256)
    densities: tuple[float, ...] = (0.8, 1.0)
    delta_m: int = 5
    j_over_t_grid: tuple[float, ...] = tuple(np.logspace(-2, 2, 21))
    instances_per_setting: int = 32
    coarsegrain_scales: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    n_samples: int = 256
    seed: int = 1234
    random_projection_density: float = 0.5


@dataclass(frozen=True)
class HarnessResult:
    config: HarnessConfig
    q1: dict
    q2: dict
    verdict: str

    def to_dict(self) -> dict:
        return {
            "config": asdict(self.config),
            "q1": self.q1,
            "q2": self.q2,
            "verdict": self.verdict,
        }


def _log_mean_exp(values: np.ndarray) -> float:
    max_val = np.max(values)
    return float(max_val + np.log(np.mean(np.exp(values - max_val))))


def _mu_over_t_grid(
    violations_minus: np.ndarray,
    violations_plus: np.ndarray,
    j_grid: np.ndarray,
    delta_m: int,
    n_vars: int,
) -> np.ndarray:
    mu_values = []
    for j_over_t in j_grid:
        log_weights_minus = -j_over_t * violations_minus
        log_weights_plus = -j_over_t * violations_plus
        log_z_minus = n_vars * np.log(2.0) + _log_mean_exp(log_weights_minus)
        log_z_plus = n_vars * np.log(2.0) + _log_mean_exp(log_weights_plus)
        mu_values.append((log_z_plus - log_z_minus) / delta_m)
    return np.array(mu_values, dtype=np.float64)


def _has_contiguous_region(
    j_grid: np.ndarray,
    mu_over_t: np.ndarray,
    tolerance: float = 0.1,
    min_decades: float = 0.5,
) -> bool:
    mask = np.abs(mu_over_t - np.log(2.0)) <= tolerance
    if not np.any(mask):
        return False
    log_j = np.log10(j_grid)
    start = None
    for idx, ok in enumerate(mask):
        if ok and start is None:
            start = idx
        if not ok and start is not None:
            width = log_j[idx - 1] - log_j[start]
            if width >= min_decades:
                return True
            start = None
    if start is not None:
        width = log_j[len(mask) - 1] - log_j[start]
        if width >= min_decades:
            return True
    return False


def _median_by_scale(values: dict[int, list[int]]) -> dict[int, float]:
    return {scale: float(np.median(vals)) for scale, vals in values.items()}


def run_harness(config: HarnessConfig) -> HarnessResult:
    rng = np.random.default_rng(config.seed)
    j_grid = np.array(config.j_over_t_grid, dtype=np.float64)
    q1_settings = []
    q2_block: dict[int, list[int]] = {scale: [] for scale in config.coarsegrain_scales}
    q2_random: dict[int, list[int]] = {scale: [] for scale in config.coarsegrain_scales}

    for n_vars in config.n_values:
        for density in config.densities:
            m_base = int(round(density * n_vars))
            instance_records = []
            successes = []
            mu_over_t_values = []
            for _ in range(config.instances_per_setting):
                inst_seed = int(rng.integers(0, np.iinfo(np.uint32).max))
                inst_rng = np.random.default_rng(inst_seed)
                pair = generate_constraint_pair(inst_rng, n_vars, m_base, config.delta_m)
                delta_m = gf2_rank(pair.a_plus) - gf2_rank(pair.a_minus)
                samples = inst_rng.integers(
                    0, 2, size=(config.n_samples, n_vars), dtype=np.uint8
                )
                parity_minus = gf2_matmul_batch(pair.a_minus, samples)
                parity_plus = gf2_matmul_batch(pair.a_plus, samples)
                violations_minus = np.sum(parity_minus != pair.b_minus, axis=1).astype(np.float64)
                violations_plus = np.sum(parity_plus != pair.b_plus, axis=1).astype(np.float64)

                mu_values = _mu_over_t_grid(
                    violations_minus,
                    violations_plus,
                    j_grid,
                    delta_m,
                    n_vars,
                )
                mu_over_t_values.append(mu_values)
                success = _has_contiguous_region(j_grid, mu_values)
                successes.append(success)
                instance_records.append(
                    {
                        "seed": inst_seed,
                        "mu_over_t": mu_values.tolist(),
                        "success": bool(success),
                    }
                )

                cg_seed = int(inst_rng.integers(0, np.iinfo(np.uint32).max))
                cg_rng = np.random.default_rng(cg_seed)
                cg_results = coarsegrain_delta_m(
                    pair.a_minus,
                    pair.a_plus,
                    config.coarsegrain_scales,
                    cg_rng,
                    density=config.random_projection_density,
                )
                for result in cg_results:
                    if result.map_name == "block":
                        q2_block[result.block_size].append(result.delta_m)
                    else:
                        q2_random[result.block_size].append(result.delta_m)

            mu_over_t_values = np.vstack(mu_over_t_values)
            q1_settings.append(
                {
                    "n": n_vars,
                    "density": density,
                    "m_base": m_base,
                    "instances": config.instances_per_setting,
                    "success_rate": float(np.mean(successes)),
                    "j_over_t": j_grid.tolist(),
                    "mu_over_t_mean": np.mean(mu_over_t_values, axis=0).tolist(),
                    "mu_over_t_std": np.std(mu_over_t_values, axis=0).tolist(),
                    "instances_detail": instance_records,
                }
            )

    q1_success_rates = [setting["success_rate"] for setting in q1_settings]
    q1_robust = all(rate >= 0.7 for rate in q1_success_rates)
    q1_fail = all(rate < 0.7 for rate in q1_success_rates)

    q2_block_median = _median_by_scale(q2_block)
    q2_random_median = _median_by_scale(q2_random)
    q2_marginal = True
    q2_fail = False
    for scale in config.coarsegrain_scales:
        if scale < 2:
            continue
        for med in (q2_block_median[scale], q2_random_median[scale]):
            if not (1 <= med <= 10):
                q2_marginal = False
                q2_fail = True

    q2 = {
        "block": {
            "values": q2_block,
            "median": q2_block_median,
        },
        "random": {
            "values": q2_random,
            "median": q2_random_median,
        },
        "marginal": q2_marginal,
    }

    if q1_robust and q2_marginal:
        verdict = "DERIVATION FOUND"
    elif q1_fail or q2_fail:
        verdict = "NO-GO"
    else:
        verdict = "INCONCLUSIVE"

    q1 = {
        "settings": q1_settings,
        "robust": q1_robust,
        "failed": q1_fail,
        "tolerance": 0.1,
        "min_decades": 0.5,
    }

    return HarnessResult(config=config, q1=q1, q2=q2, verdict=verdict)
