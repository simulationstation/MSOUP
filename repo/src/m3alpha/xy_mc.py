"""Metropolis Monte Carlo for the 2D XY model."""
from __future__ import annotations

import dataclasses
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np

from .mapping import alpha_eff_from_k, k_from_helicity
from .observables import helicity_modulus


@dataclasses.dataclass(frozen=True)
class MCConfig:
    lattice_sizes: tuple[int, ...] = (16, 32, 64)
    temperatures: tuple[float, ...] = tuple(np.linspace(0.7, 1.1, 20))
    thermalization_sweeps: int = 500
    measurement_sweeps: int = 800
    sweep_stride: int = 10
    proposal_width: float = math.pi / 3.0
    seed: int = 1234


@dataclasses.dataclass(frozen=True)
class MCResult:
    lattice_sizes: tuple[int, ...]
    temperatures: tuple[float, ...]
    helicity_modulus: Dict[int, np.ndarray]
    alpha_eff: Dict[int, np.ndarray]


def _metropolis_sweep(theta: np.ndarray, temperature: float, width: float, rng: np.random.Generator) -> None:
    l_size = theta.shape[0]
    for _ in range(l_size * l_size):
        i = rng.integers(0, l_size)
        j = rng.integers(0, l_size)
        old_theta = theta[i, j]
        proposal = old_theta + rng.uniform(-width, width)

        up = theta[(i - 1) % l_size, j]
        down = theta[(i + 1) % l_size, j]
        left = theta[i, (j - 1) % l_size]
        right = theta[i, (j + 1) % l_size]

        def energy(angle: float) -> float:
            return -(
                math.cos(angle - up)
                + math.cos(angle - down)
                + math.cos(angle - left)
                + math.cos(angle - right)
            )

        delta_e = energy(proposal) - energy(old_theta)
        if delta_e <= 0.0 or rng.random() < math.exp(-delta_e / temperature):
            theta[i, j] = proposal


def _run_single_temperature(l_size: int, temperature: float, config: MCConfig, rng: np.random.Generator) -> float:
    theta = rng.uniform(0.0, 2.0 * math.pi, size=(l_size, l_size))
    for _ in range(config.thermalization_sweeps):
        _metropolis_sweep(theta, temperature, config.proposal_width, rng)

    measurements: List[float] = []
    total_sweeps = config.measurement_sweeps
    for sweep in range(total_sweeps):
        _metropolis_sweep(theta, temperature, config.proposal_width, rng)
        if sweep % config.sweep_stride == 0:
            measurements.append(helicity_modulus(theta, temperature))

    return float(np.mean(measurements))


def _worker(args: Tuple[int, int, float, int, int, int, float, int]) -> Tuple[int, int, float, float]:
    """Worker function for parallel execution."""
    l_size, temp_idx, temperature, therm_sweeps, meas_sweeps, stride, width, seed = args
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0.0, 2.0 * math.pi, size=(l_size, l_size))

    for _ in range(therm_sweeps):
        _metropolis_sweep(theta, temperature, width, rng)

    measurements: List[float] = []
    for sweep in range(meas_sweeps):
        _metropolis_sweep(theta, temperature, width, rng)
        if sweep % stride == 0:
            measurements.append(helicity_modulus(theta, temperature))

    hel = float(np.mean(measurements))
    return l_size, temp_idx, temperature, hel


def run_mc(config: MCConfig) -> MCResult:
    rng = np.random.default_rng(config.seed)
    helicity: Dict[int, np.ndarray] = {}
    alpha_eff: Dict[int, np.ndarray] = {}

    temps = np.array(config.temperatures, dtype=float)

    # Build work items for parallel execution
    work_items = []
    for l_size in config.lattice_sizes:
        for idx, temp in enumerate(temps):
            # Generate unique seed for each (L, T) pair
            seed = rng.integers(0, 2**31)
            work_items.append((
                l_size, idx, float(temp),
                config.thermalization_sweeps,
                config.measurement_sweeps,
                config.sweep_stride,
                config.proposal_width,
                seed
            ))

    # Initialize storage
    for l_size in config.lattice_sizes:
        helicity[l_size] = np.zeros(len(temps))
        alpha_eff[l_size] = np.zeros(len(temps))

    # Run in parallel
    n_workers = min(os.cpu_count() or 4, len(work_items))
    print(f"Running {len(work_items)} MC simulations on {n_workers} workers...")

    completed = 0
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_worker, item): item for item in work_items}
        for future in as_completed(futures):
            l_size, temp_idx, temperature, hel = future.result()
            k_r = k_from_helicity(hel, temperature)
            helicity[l_size][temp_idx] = hel
            alpha_eff[l_size][temp_idx] = alpha_eff_from_k(k_r)
            completed += 1
            if completed % 10 == 0 or completed == len(work_items):
                print(f"  Progress: {completed}/{len(work_items)}")

    return MCResult(
        lattice_sizes=config.lattice_sizes,
        temperatures=config.temperatures,
        helicity_modulus=helicity,
        alpha_eff=alpha_eff,
    )
