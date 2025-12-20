"""
Msoup Closure Model: Minimal observability-horizon effective model.

This package implements the constrained Msoup closure:
  (C1) visibility V(M; kappa) = exp[-kappa * (M - 2)]
  (C2) a single scalar M_vis(z) controls low-order observables
  (C3) a single mapping M_vis(z) -> effective response channel (c_eff^2)

Global parameters (the ONLY free parameters):
  - c_star_sq: effective sound speed squared (km/s)^2 - suppression strength
  - Delta_M: amplitude of M_vis rise above threshold (M=2)
  - z_t: transition redshift
  - w: transition width

The model maps these 4 parameters to:
  - Power spectrum suppression P(k, z) / P_CDM(k, z)
  - Halo mass function suppression
  - Core radius in dwarf galaxy halos
  - Half-mode mass for lensing constraints
"""

from .visibility import visibility, m_vis, c_eff_squared, MsoupParams
from .growth import MsoupGrowthSolver, GrowthSolution, validate_lcdm_limit
from .cosmology import CosmologyParams
from .halo import HaloMassFunction, HaloMFResult
from .rotation_curves import (
    RotationCurveFitter,
    nfw_velocity,
    cored_nfw_velocity,
    total_velocity,
    fit_galaxy_sample,
    log_likelihood_sample,
)

__version__ = "0.1.0"
__all__ = [
    # Visibility and core model
    "visibility",
    "m_vis",
    "c_eff_squared",
    "MsoupParams",
    # Growth and cosmology
    "MsoupGrowthSolver",
    "GrowthSolution",
    "validate_lcdm_limit",
    "CosmologyParams",
    # Halo mass function
    "HaloMassFunction",
    "HaloMFResult",
    # Rotation curves
    "RotationCurveFitter",
    "nfw_velocity",
    "cored_nfw_velocity",
    "total_velocity",
    "fit_galaxy_sample",
    "log_likelihood_sample",
]
