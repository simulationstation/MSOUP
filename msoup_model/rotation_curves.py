"""
Rotation curve modeling and fitting for Msoup pipeline.

This module:
- Models V_total^2(r) = V_baryon^2(r) + V_DM^2(r)
- Implements NFW and cored-NFW profiles
- Ties core radius to Msoup suppression scale
- Provides hierarchical fitting framework

Key feature: ONE global Msoup parameter set controls cores across all galaxies.
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.interpolate import interp1d
from typing import Optional, Tuple, Dict, List, Callable
from dataclasses import dataclass, field
import warnings

from .visibility import MsoupParams
from .cosmology import CosmologyParams
from .growth import GrowthSolution, MsoupGrowthSolver, compute_half_mode_scale


# Physical constants
G_kpc = 4.302e-6  # G in (km/s)^2 * kpc / M_sun


@dataclass
class HaloParams:
    """
    Per-galaxy halo parameters.

    The Msoup model adds r_c (core radius) which is DERIVED from
    global Msoup params, not fitted per galaxy.
    """
    M200: float        # Virial mass in M_sun
    c: float           # Concentration
    ml_disk: float     # Stellar disk M/L ratio
    ml_bulge: float    # Bulge M/L ratio (optional)

    def log_prior(self,
                  M200_range: Tuple[float, float] = (1e8, 1e13),
                  c_range: Tuple[float, float] = (1, 50),
                  ml_disk_range: Tuple[float, float] = (0.1, 2.0),
                  ml_bulge_range: Tuple[float, float] = (0.3, 1.5)) -> float:
        """
        Log prior probability.

        Uses flat priors in log M200, flat in c, flat in M/L.
        """
        if not (M200_range[0] < self.M200 < M200_range[1]):
            return -np.inf
        if not (c_range[0] < self.c < c_range[1]):
            return -np.inf
        if not (ml_disk_range[0] < self.ml_disk < ml_disk_range[1]):
            return -np.inf
        if not (ml_bulge_range[0] < self.ml_bulge < ml_bulge_range[1]):
            return -np.inf

        # Concentration-mass prior (lognormal around median relation)
        # c_median ~ 10 * (M200 / 10^12)^(-0.1)
        c_median = 10.0 * (self.M200 / 1e12)**(-0.1)
        sigma_c = 0.3  # dex scatter
        log_prior_c = -0.5 * ((np.log10(self.c) - np.log10(c_median)) / sigma_c)**2

        return log_prior_c


def nfw_velocity(r: np.ndarray,
                 M200: float,
                 c: float,
                 cosmo: CosmologyParams) -> np.ndarray:
    """
    NFW rotation velocity in km/s.

    V_NFW^2(r) = (G M200 / r) * g(c) * [ln(1 + x) - x/(1+x)] / [ln(1+c) - c/(1+c)]

    where x = r / r_s, r_s = r_200 / c, and g(c) accounts for the
    enclosed mass fraction.

    Parameters:
        r: radius in kpc
        M200: virial mass in M_sun
        c: concentration
        cosmo: cosmology parameters

    Returns:
        V_NFW in km/s
    """
    # Virial radius
    # r_200 = (3 M200 / (4 pi * 200 * rho_crit))^(1/3)
    rho_crit = cosmo.rho_crit_0 * cosmo.h**2  # M_sun / Mpc^3
    r_200_mpc = (3 * M200 / (4 * np.pi * 200 * rho_crit))**(1.0/3.0)
    r_200 = r_200_mpc * 1000  # kpc

    r_s = r_200 / c  # Scale radius in kpc
    x = r / r_s

    # NFW enclosed mass function
    def f_nfw(x):
        return np.log(1 + x) - x / (1 + x)

    # Velocity squared
    V_c = np.sqrt(G_kpc * M200 / r_200)  # Characteristic velocity
    f_c = f_nfw(c)
    f_x = f_nfw(x)

    V_sq = V_c**2 * (r_200 / r) * f_x / f_c

    return np.sqrt(np.maximum(V_sq, 0))


def cored_nfw_velocity(r: np.ndarray,
                       M200: float,
                       c: float,
                       r_c: float,
                       cosmo: CosmologyParams) -> np.ndarray:
    """
    Cored NFW rotation velocity.

    Uses the cNFW profile (Read et al. 2016):
        rho(r) = rho_NFW(r) * tanh(r / r_c)

    Or equivalently, Burkert-like core inside r_c.

    For simplicity, we use an interpolation:
        V^2(r) = V_NFW^2(r) * min(1, (r/r_c)^n)

    where n controls the core shape. This gives:
        - r >> r_c: standard NFW
        - r << r_c: solid body (V ∝ r)

    Parameters:
        r: radius in kpc
        M200: virial mass in M_sun
        c: concentration
        r_c: core radius in kpc
        cosmo: cosmology parameters

    Returns:
        V_cored in km/s
    """
    if r_c <= 0:
        return nfw_velocity(r, M200, c, cosmo)

    V_nfw = nfw_velocity(r, M200, c, cosmo)

    # Core modification: smooth transition
    # f(r) = (r/r_c)^n / (1 + (r/r_c)^n) with n ~ 2 gives nice core
    n = 2.0
    x = r / r_c
    core_factor = x**n / (1 + x**n)

    # The cored profile has less mass at center
    # V_cored^2 = V_NFW^2 * core_factor (approximately)
    V_sq = V_nfw**2 * core_factor

    return np.sqrt(np.maximum(V_sq, 0))


def total_velocity(r: np.ndarray,
                   v_gas: np.ndarray,
                   v_disk: np.ndarray,
                   v_bulge: np.ndarray,
                   M200: float,
                   c: float,
                   ml_disk: float,
                   ml_bulge: float,
                   r_c: float,
                   cosmo: CosmologyParams) -> np.ndarray:
    """
    Total rotation velocity.

    V_tot^2 = V_gas^2 + ml_disk * V_disk^2 + ml_bulge * V_bulge^2 + V_DM^2

    Parameters:
        r: radius array in kpc
        v_gas, v_disk, v_bulge: baryonic components at M/L=1
        M200, c: halo parameters
        ml_disk, ml_bulge: mass-to-light ratios
        r_c: core radius (0 for NFW, >0 for cored)
        cosmo: cosmology

    Returns:
        V_total in km/s
    """
    # Baryonic contribution
    V_bar_sq = v_gas**2 + ml_disk * v_disk**2 + ml_bulge * v_bulge**2

    # DM contribution
    if r_c > 0:
        V_dm = cored_nfw_velocity(r, M200, c, r_c, cosmo)
    else:
        V_dm = nfw_velocity(r, M200, c, cosmo)

    V_tot_sq = V_bar_sq + V_dm**2

    return np.sqrt(np.maximum(V_tot_sq, 0))


def chi_squared(v_obs: np.ndarray,
                v_err: np.ndarray,
                v_model: np.ndarray) -> float:
    """Compute chi-squared."""
    return np.sum(((v_obs - v_model) / v_err)**2)


def reduced_chi_squared(v_obs: np.ndarray,
                        v_err: np.ndarray,
                        v_model: np.ndarray,
                        n_params: int) -> float:
    """Compute reduced chi-squared."""
    n_dof = len(v_obs) - n_params
    if n_dof <= 0:
        return np.inf
    return chi_squared(v_obs, v_err, v_model) / n_dof


class RotationCurveFitter:
    """
    Rotation curve fitter with Msoup-derived cores.

    Fitting strategy:
    1. Global Msoup params (c_star_sq, Delta_M, z_t, w) shared across galaxies
    2. Per-galaxy params (M200, c, ml_disk) fitted with priors
    3. Core radius derived from Msoup suppression at formation epoch
    """

    def __init__(self,
                 cosmo: Optional[CosmologyParams] = None,
                 msoup_params: Optional[MsoupParams] = None):
        """
        Initialize fitter.

        Parameters:
            cosmo: Cosmology parameters
            msoup_params: Msoup model parameters (None = CDM baseline)
        """
        self.cosmo = cosmo if cosmo is not None else CosmologyParams()
        self.msoup_params = msoup_params
        self.growth_solution = None

        # Pre-compute growth solution if Msoup params given
        if msoup_params is not None and msoup_params.c_star_sq > 0:
            solver = MsoupGrowthSolver(msoup_params, self.cosmo)
            self.growth_solution = solver.solve()

    def set_msoup_params(self, params: MsoupParams):
        """Update Msoup parameters and recompute growth."""
        self.msoup_params = params
        if params.c_star_sq > 0:
            solver = MsoupGrowthSolver(params, self.cosmo)
            self.growth_solution = solver.solve()
        else:
            self.growth_solution = None

    def formation_redshift(self, M200: float) -> float:
        """
        Estimate formation redshift for a halo of mass M200.

        Simple approximation: more massive halos form later.
        z_form ~ 2 * (M200 / 10^12)^(-0.1)
        """
        z_form = 2.0 * (M200 / 1e12)**(-0.1)
        return max(0.5, min(z_form, 10.0))

    def core_radius(self, M200: float, c: float) -> float:
        """
        Compute core radius from Msoup suppression.

        The core radius is tied to the suppression scale at the
        halo's formation epoch.

        Parameters:
            M200: halo mass
            c: concentration

        Returns:
            r_c in kpc
        """
        if self.growth_solution is None or self.msoup_params is None:
            return 0.0  # No coring in CDM

        if self.msoup_params.c_star_sq <= 0:
            return 0.0

        # Formation redshift
        z_form = self.formation_redshift(M200)

        # NFW scale radius
        rho_crit = self.cosmo.rho_crit_0 * self.cosmo.h**2
        r_200_mpc = (3 * M200 / (4 * np.pi * 200 * rho_crit))**(1.0/3.0)
        r_200_kpc = r_200_mpc * 1000
        r_s = r_200_kpc / c

        # Suppression scale at formation from half-mode definition
        hm_scale = compute_half_mode_scale(
            self.growth_solution,
            z=z_form,
            power_threshold=0.25
        )

        if hm_scale.k_hm is None or hm_scale.k_hm <= 0:
            # If suppression never reaches 25%, fall back to the most suppressed mode
            ratios = self.growth_solution.power_ratio(self.growth_solution.k_grid, z_form)
            if np.allclose(ratios, 1.0, atol=1e-4):
                return 0.0
            k_effective = self.growth_solution.k_grid[np.argmin(ratios)]
        else:
            k_effective = hm_scale.k_hm

        # Convert suppression wavelength to physical length
        r_halfmode_kpc = (np.pi / k_effective) * 1000 / self.cosmo.h  # kpc

        # Core radius scales with the geometric mean of r_s and the suppression length
        r_c = np.sqrt(r_halfmode_kpc * r_s)

        # Cap at a reasonable fraction of scale radius to avoid unphysical cores
        r_c = np.clip(r_c, 0.0, 1.5 * r_s)

        return float(r_c)

    def fit_single_galaxy(self,
                          r: np.ndarray,
                          v_obs: np.ndarray,
                          v_err: np.ndarray,
                          v_gas: np.ndarray,
                          v_disk: np.ndarray,
                          v_bulge: np.ndarray,
                          fix_ml_disk: Optional[float] = None) -> Dict:
        """
        Fit rotation curve for a single galaxy.

        Parameters:
            r, v_obs, v_err: rotation curve data
            v_gas, v_disk, v_bulge: baryonic components
            fix_ml_disk: fix disk M/L to this value (if given)

        Returns:
            Dictionary with best-fit parameters and chi-squared
        """
        # Bounds
        bounds = [
            (1e8, 1e13),   # M200
            (1, 50),       # c
        ]

        if fix_ml_disk is None:
            bounds.append((0.2, 1.5))  # ml_disk
            n_free = 3
        else:
            n_free = 2

        def objective(x):
            M200, c = x[0], x[1]
            ml_disk = fix_ml_disk if fix_ml_disk else x[2]
            ml_bulge = 0.7  # Fixed

            r_c = self.core_radius(M200, c)

            v_model = total_velocity(
                r, v_gas, v_disk, v_bulge,
                M200, c, ml_disk, ml_bulge, r_c,
                self.cosmo
            )

            return chi_squared(v_obs, v_err, v_model)

        # Initial guess
        v_max = np.max(v_obs)
        M200_init = 1e10 * (v_max / 50)**3
        x0 = [M200_init, 10.0]
        if fix_ml_disk is None:
            x0.append(0.5)

        # Optimize
        try:
            result = differential_evolution(objective, bounds, seed=42, maxiter=200)
            x_best = result.x
            chi2_best = result.fun
        except Exception as e:
            warnings.warn(f"Optimization failed: {e}")
            x_best = x0
            chi2_best = objective(x0)

        M200, c = x_best[0], x_best[1]
        ml_disk = fix_ml_disk if fix_ml_disk else x_best[2]
        r_c = self.core_radius(M200, c)

        v_model = total_velocity(
            r, v_gas, v_disk, v_bulge,
            M200, c, ml_disk, 0.7, r_c,
            self.cosmo
        )

        return {
            'M200': M200,
            'c': c,
            'ml_disk': ml_disk,
            'r_c': r_c,
            'chi2': chi2_best,
            'chi2_red': chi2_best / (len(r) - n_free),
            'v_model': v_model,
            'n_dof': len(r) - n_free,
        }


def fit_galaxy_sample(galaxies: List,
                      msoup_params: Optional[MsoupParams] = None,
                      cosmo: Optional[CosmologyParams] = None,
                      verbose: bool = True) -> Dict:
    """
    Fit a sample of galaxies with shared Msoup parameters.

    Parameters:
        galaxies: List of GalaxyData objects
        msoup_params: Global Msoup parameters
        cosmo: Cosmology

    Returns:
        Dictionary with results for each galaxy and summary statistics
    """
    fitter = RotationCurveFitter(cosmo=cosmo, msoup_params=msoup_params)

    results = []
    total_chi2 = 0
    total_dof = 0

    for i, gal in enumerate(galaxies):
        if verbose and i % 5 == 0:
            print(f"  Fitting galaxy {i+1}/{len(galaxies)}: {gal.name}")

        try:
            fit = fitter.fit_single_galaxy(
                gal.radius, gal.v_obs, gal.v_err,
                gal.v_gas, gal.v_disk, gal.v_bulge
            )
            fit['name'] = gal.name
            fit['success'] = True
            results.append(fit)

            total_chi2 += fit['chi2']
            total_dof += fit['n_dof']

        except Exception as e:
            if verbose:
                print(f"    Warning: Failed to fit {gal.name}: {e}")
            results.append({
                'name': gal.name,
                'success': False,
                'error': str(e)
            })

    return {
        'galaxies': results,
        'total_chi2': total_chi2,
        'total_dof': total_dof,
        'chi2_per_dof': total_chi2 / total_dof if total_dof > 0 else np.inf,
        'n_success': sum(1 for r in results if r.get('success', False)),
        'msoup_params': msoup_params,
    }


def log_likelihood_sample(theta: np.ndarray,
                          galaxies: List,
                          cosmo: CosmologyParams) -> float:
    """
    Log-likelihood for global Msoup parameters given galaxy sample.

    Parameters:
        theta: [c_star_sq, Delta_M, z_t, w]
        galaxies: List of GalaxyData
        cosmo: Cosmology

    Returns:
        Log-likelihood (higher is better)
    """
    c_star_sq, Delta_M, z_t, w = theta

    # Prior bounds
    if c_star_sq < 0 or c_star_sq > 1000:
        return -np.inf
    if Delta_M < 0 or Delta_M > 2:
        return -np.inf
    if z_t < 0.5 or z_t > 10:
        return -np.inf
    if w < 0.05 or w > 2:
        return -np.inf

    try:
        params = MsoupParams(
            c_star_sq=c_star_sq,
            Delta_M=Delta_M,
            z_t=z_t,
            w=w
        )
    except ValueError:
        return -np.inf

    results = fit_galaxy_sample(galaxies, params, cosmo, verbose=False)

    # Log-likelihood ~ -chi^2 / 2
    log_lik = -0.5 * results['total_chi2']

    return log_lik
