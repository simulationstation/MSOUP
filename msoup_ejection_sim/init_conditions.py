"""Initial condition builders for the ejection simulation."""
from __future__ import annotations

import numpy as np

from .density_field import make_fourier_field, lognormal_density, evolve_sigma, gaussian_smooth
from .config import SimulationConfig


def initialize_fields(cfg: SimulationConfig):
    rng = np.random.default_rng(cfg.seed)
    base_field = make_fourier_field(rng, cfg.grid, cfg.spectral_index, cfg.kd)
    sigma0 = evolve_sigma(0.0, cfg.sigma0, cfg.sigmaF)
    rho_b0 = lognormal_density(base_field, sigma0)

    rho_dm3 = initialize_dm3(rng, rho_b0, cfg)
    A = np.full_like(rho_b0, cfg.A0)
    wlt2 = np.zeros_like(rho_b0)
    rho_mem = rho_b0 + cfg.g_dm * rho_dm3
    return base_field, rho_b0, rho_dm3, A, wlt2, rho_mem


def initialize_dm3(rng: np.random.Generator, rho_b0: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    flat = rho_b0.flatten()
    threshold = np.quantile(flat, cfg.q_pocket)
    pocket_mask = rho_b0 >= threshold

    if cfg.pocket_extra > 0:
        rand_mask = rng.random(size=rho_b0.shape)
        pocket_mask |= rand_mask > (1.0 - cfg.q_pocket)  # sprinkle extras

    dm3_raw = np.zeros_like(rho_b0)
    dm3_raw[pocket_mask] = rho_b0[pocket_mask] ** cfg.pocket_alpha
    dm3_raw = gaussian_smooth(dm3_raw, cfg.pocket_smooth)
    dm3_raw = np.clip(dm3_raw, 0, None)
    total_mass = dm3_raw.sum()
    if total_mass <= 0:
        return dm3_raw

    target_mass = cfg.f_dm3_0 * rho_b0.sum()
    dm3_scaled = dm3_raw * (target_mass / (total_mass + 1e-8))
    return dm3_scaled
