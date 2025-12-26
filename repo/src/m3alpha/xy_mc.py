"""Metropolis Monte Carlo for the 2D XY model."""
from __future__ import annotations

import dataclasses
import math
from typing import Dict, List

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


def run_mc(config: MCConfig) -> MCResult:
    rng = np.random.default_rng(config.seed)
    helicity: Dict[int, np.ndarray] = {}
    alpha_eff: Dict[int, np.ndarray] = {}

    temps = np.array(config.temperatures, dtype=float)
    for l_size in config.lattice_sizes:
        hel_values = np.zeros_like(temps)
        alpha_values = np.zeros_like(temps)
        for idx, temp in enumerate(temps):
            hel = _run_single_temperature(l_size, float(temp), config, rng)
            k_r = k_from_helicity(hel, float(temp))
            hel_values[idx] = hel
            alpha_values[idx] = alpha_eff_from_k(k_r)
        helicity[l_size] = hel_values
        alpha_eff[l_size] = alpha_values

    return MCResult(
        lattice_sizes=config.lattice_sizes,
        temperatures=config.temperatures,
        helicity_modulus=helicity,
        alpha_eff=alpha_eff,
    )
