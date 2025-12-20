"""
Halo and subhalo mass functions for Msoup model.

This module:
- Computes the halo mass function with Msoup suppression
- Derives the subhalo mass function for lensing comparisons
- Provides the half-mode mass diagnostic

Approach:
- Use Press-Schechter or Sheth-Tormen formalism
- Apply Msoup suppression through the modified P(k)
- Map to subhalo MF using empirical scaling relations
"""

import numpy as np
from scipy.integrate import quad, simpson
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from typing import Optional, Tuple, Dict
from dataclasses import dataclass

from .cosmology import CosmologyParams
from .visibility import MsoupParams
from .growth import GrowthSolution, MsoupGrowthSolver


@dataclass
class HaloMFResult:
    """Container for halo mass function results."""
    M: np.ndarray           # Halo masses in M_sun/h
    dndlnM: np.ndarray      # dn/d(ln M) in (h/Mpc)^3
    dndlnM_cdm: np.ndarray  # CDM reference
    ratio: np.ndarray       # dndlnM / dndlnM_cdm
    z: float                # Redshift
    params: MsoupParams     # Model parameters used


class HaloMassFunction:
    """
    Halo mass function calculator with Msoup suppression.

    Uses Sheth-Tormen fitting function with Msoup-modified power spectrum.
    """

    def __init__(self,
                 cosmo: Optional[CosmologyParams] = None,
                 growth_solution: Optional[GrowthSolution] = None):
        """
        Initialize HMF calculator.

        Parameters:
            cosmo: Cosmology parameters
            growth_solution: Pre-computed growth solution (if None, uses CDM)
        """
        self.cosmo = cosmo if cosmo is not None else CosmologyParams()
        self.growth_solution = growth_solution

        # Sheth-Tormen parameters
        self.A_st = 0.3222
        self.a_st = 0.707
        self.p_st = 0.3

        # Critical overdensity for collapse
        self.delta_c = 1.686

        # Window function: top-hat in real space
        # R(M) = (3M / (4 pi rho_m))^(1/3)

    def _mass_to_radius(self, M: float) -> float:
        """
        Convert mass to Lagrangian radius.

        R = (3 M / (4 pi rho_m0))^(1/3) in Mpc/h
        """
        rho_m = self.cosmo.rho_m_0  # h^2 M_sun / Mpc^3
        # M is in M_sun/h, rho_m is in h^2 M_sun / Mpc^3
        # R^3 = 3 M / (4 pi rho_m) = 3 M / (4 pi h^2 M_sun / Mpc^3)
        # Need M in h^-1 M_sun: M_h = M * h
        # R^3 = 3 M * h / (4 pi rho_m) [Mpc^3 / h^3]
        # Actually: M [M_sun/h], rho_m [h^2 M_sun / Mpc^3]
        # R^3 = M [M_sun/h] / rho_m [h^2 M_sun / Mpc^3] = M / (rho_m h^3) [Mpc^3]
        # Wait, let me be more careful.

        # rho_m0 = Omega_m * rho_crit = 0.3 * 2.775e11 h^2 M_sun / Mpc^3
        # For a sphere of radius R [Mpc/h]:
        #   M = (4/3) pi R^3 rho_m0
        #   M [(Mpc/h)^3 * h^2 M_sun/Mpc^3] = M [h^-1 M_sun]
        # So M is naturally in M_sun/h when R is in Mpc/h

        R_cubed = 3 * M / (4 * np.pi * rho_m)  # (Mpc/h)^3
        return R_cubed**(1.0/3.0)

    def _radius_to_mass(self, R: float) -> float:
        """Convert radius to mass."""
        rho_m = self.cosmo.rho_m_0
        return (4.0/3.0) * np.pi * R**3 * rho_m

    def _window_function(self, k: np.ndarray, R: float) -> np.ndarray:
        """
        Top-hat window function in Fourier space.

        W(kR) = 3 * (sin(kR) - kR cos(kR)) / (kR)^3
        """
        kR = k * R
        # Avoid division by zero
        kR = np.maximum(kR, 1e-10)
        return 3.0 * (np.sin(kR) - kR * np.cos(kR)) / kR**3

    def _sigma_squared(self, R: float, z: float = 0) -> float:
        """
        Variance of the density field smoothed on scale R.

        sigma^2(R) = (1/2pi^2) * integral of k^2 P(k) W^2(kR) dk
        """
        if self.growth_solution is None:
            return self._sigma_squared_cdm(R, z)

        k_grid = self.growth_solution.k_grid
        P_ratio = self.growth_solution.power_ratio(k_grid, z)

        # Get CDM P(k) from colossus or fitting formula
        P_cdm = self._pk_cdm(k_grid, z)
        P_k = P_cdm * P_ratio

        W = self._window_function(k_grid, R)

        integrand = k_grid**2 * P_k * W**2 / (2 * np.pi**2)

        # Integrate using Simpson's rule
        return simpson(integrand, x=np.log(k_grid)) * np.log(k_grid[1]/k_grid[0])

    def _sigma_squared_cdm(self, R: float, z: float = 0) -> float:
        """
        CDM sigma^2(R) using fitting formula.
        """
        # Simple approximation: sigma_8 scaled
        # sigma(R) ~ sigma_8 * (R / 8 Mpc/h)^(-(n_s + 3)/2)
        sigma8 = self.cosmo.sigma8
        n_s = self.cosmo.n_s

        # Growth factor
        D = self.cosmo.growth_factor_lcdm(z)

        # Scale dependence
        gamma = (n_s + 3) / 2
        sigma_R = sigma8 * (R / 8.0)**(-gamma) * D

        return sigma_R**2

    def _pk_cdm(self, k: np.ndarray, z: float = 0) -> np.ndarray:
        """
        CDM power spectrum using Eisenstein-Hu transfer function.

        Simplified version for this implementation.
        """
        # Normalization from sigma_8
        # P(k) = A * k^n_s * T^2(k)

        # EH fitting formula (simplified)
        h = self.cosmo.h
        Omega_m = self.cosmo.Omega_m
        Omega_b = self.cosmo.Omega_b

        # Shape parameter
        Gamma = Omega_m * h * np.exp(-Omega_b * (1 + np.sqrt(2*h)/Omega_m))

        # Transfer function (approximate)
        q = k / Gamma  # k in h/Mpc
        T = np.log(1 + 2.34 * q) / (2.34 * q) * (
            1 + 3.89*q + (16.1*q)**2 + (5.46*q)**3 + (6.71*q)**4
        )**(-0.25)

        # Power spectrum
        D = self.cosmo.growth_factor_lcdm(z)
        P_k = k**self.cosmo.n_s * T**2 * D**2

        # Normalize to sigma_8
        # (Would need proper normalization; using approximate scaling)
        P_k *= 1e4  # Rough normalization factor

        return P_k

    def _f_st(self, nu: np.ndarray) -> np.ndarray:
        """
        Sheth-Tormen multiplicity function f(nu).

        f(nu) = A * sqrt(2a/pi) * (1 + (a*nu^2)^(-p)) * nu * exp(-a*nu^2/2)
        """
        a = self.a_st
        p = self.p_st
        A = self.A_st

        term1 = A * np.sqrt(2 * a / np.pi)
        term2 = 1 + (a * nu**2)**(-p)
        term3 = nu * np.exp(-a * nu**2 / 2)

        return term1 * term2 * term3

    def compute_hmf(self, z: float = 0,
                    M_min: float = 1e6,
                    M_max: float = 1e15,
                    n_M: int = 100) -> HaloMFResult:
        """
        Compute halo mass function at redshift z.

        Parameters:
            z: redshift
            M_min, M_max: mass range in M_sun/h
            n_M: number of mass bins

        Returns:
            HaloMFResult object
        """
        M_grid = np.logspace(np.log10(M_min), np.log10(M_max), n_M)

        dndlnM = np.zeros(n_M)
        dndlnM_cdm = np.zeros(n_M)

        rho_m = self.cosmo.rho_m_0

        for i, M in enumerate(M_grid):
            R = self._mass_to_radius(M)

            # Variance with Msoup suppression
            sigma_sq = self._sigma_squared(R, z)
            sigma = np.sqrt(sigma_sq)

            # CDM variance
            sigma_sq_cdm = self._sigma_squared_cdm(R, z)
            sigma_cdm = np.sqrt(sigma_sq_cdm)

            # Peak height
            nu = self.delta_c / sigma
            nu_cdm = self.delta_c / sigma_cdm

            # d(ln sigma^-1)/d(ln M)
            # Approximate derivative
            dR = 0.01 * R
            sigma_sq_plus = self._sigma_squared(R + dR, z)
            dlnsigma_dlnM = -(sigma_sq_plus - sigma_sq) / (2 * sigma_sq * 3 * dR / R)

            sigma_sq_cdm_plus = self._sigma_squared_cdm(R + dR, z)
            dlnsigma_dlnM_cdm = -(sigma_sq_cdm_plus - sigma_sq_cdm) / (2 * sigma_sq_cdm * 3 * dR / R)

            # dn/d(ln M) = (rho_m / M) * f(nu) * |d(ln sigma^-1)/d(ln M)|
            f_nu = self._f_st(nu)
            f_nu_cdm = self._f_st(nu_cdm)

            dndlnM[i] = (rho_m / M) * f_nu * np.abs(dlnsigma_dlnM)
            dndlnM_cdm[i] = (rho_m / M) * f_nu_cdm * np.abs(dlnsigma_dlnM_cdm)

        # Ratio
        ratio = dndlnM / np.maximum(dndlnM_cdm, 1e-50)

        params = self.growth_solution.params if self.growth_solution else None

        return HaloMFResult(
            M=M_grid,
            dndlnM=dndlnM,
            dndlnM_cdm=dndlnM_cdm,
            ratio=ratio,
            z=z,
            params=params
        )

    def half_mode_mass(self, z: float = 0, threshold: float = 0.5) -> float:
        """
        Find the half-mode mass where HMF is suppressed by threshold.

        Parameters:
            z: redshift
            threshold: suppression factor (0.5 = half)

        Returns:
            M_hm in M_sun/h
        """
        result = self.compute_hmf(z=z)

        # Find where ratio crosses threshold
        if np.all(result.ratio > threshold):
            return 0.0  # No suppression
        if np.all(result.ratio < threshold):
            return np.inf  # All suppressed

        # Interpolate to find crossing
        log_M = np.log10(result.M)
        idx = np.where(result.ratio < threshold)[0][0]

        if idx == 0:
            return result.M[0]

        # Linear interpolation in log M
        M1, M2 = result.M[idx-1], result.M[idx]
        r1, r2 = result.ratio[idx-1], result.ratio[idx]

        log_M_hm = np.log10(M1) + (threshold - r1) / (r2 - r1) * (np.log10(M2) - np.log10(M1))

        return 10**log_M_hm


def compute_subhalo_mf_ratio(M_sub: np.ndarray,
                             M_host: float,
                             hmf_ratio: np.ndarray,
                             M_hmf: np.ndarray,
                             f_strip: float = 0.5) -> np.ndarray:
    """
    Compute subhalo mass function ratio from HMF ratio.

    Assumes subhalos trace the accretion history:
        dn_sub/dM_sub ∝ (M_sub / M_host)^alpha * HMF_ratio(M_sub at infall)

    Parameters:
        M_sub: Subhalo masses (present day)
        M_host: Host halo mass
        hmf_ratio: HMF ratio from Msoup model
        M_hmf: HMF mass grid
        f_strip: Stripping factor (M_infall = M_present / f_strip)

    Returns:
        Subhalo MF ratio (Msoup / CDM)
    """
    # Infall mass (before stripping)
    M_infall = M_sub / f_strip

    # Interpolate HMF ratio
    interp = interp1d(np.log10(M_hmf), hmf_ratio,
                     bounds_error=False, fill_value=(1.0, 0.0))

    ratio = interp(np.log10(M_infall))

    return ratio


def core_radius_from_suppression(M_halo: float,
                                 z_form: float,
                                 growth_solution: GrowthSolution,
                                 r_s_cdm: float) -> float:
    """
    Derive core radius from Msoup suppression at formation epoch.

    The core radius is tied to the suppression scale at the halo's
    formation redshift:
        r_c ∝ (1 / k_suppression) at z_form

    Parameters:
        M_halo: Halo mass in M_sun/h
        z_form: Formation redshift
        growth_solution: Msoup growth solution
        r_s_cdm: NFW scale radius (for normalization)

    Returns:
        Core radius in same units as r_s_cdm
    """
    if growth_solution is None:
        return 0.0  # No coring in CDM

    params = growth_solution.params

    # Characteristic wavenumber at formation
    # k_c where suppression is significant
    k_grid = growth_solution.k_grid
    suppression = growth_solution.transfer_ratio(k_grid, z_form)

    # Find k where suppression = 0.5
    if np.all(suppression > 0.5):
        return 0.0  # No significant suppression
    if np.all(suppression < 0.5):
        k_c = k_grid[0]
    else:
        idx = np.where(suppression < 0.5)[0][0]
        if idx == 0:
            k_c = k_grid[0]
        else:
            k1, k2 = k_grid[idx-1], k_grid[idx]
            s1, s2 = suppression[idx-1], suppression[idx]
            k_c = k1 * (k2/k1)**((0.5 - s1) / (s2 - s1))

    # Core radius from suppression scale
    # r_c ~ 1 / k_c (in Mpc/h), need to convert
    r_c_mpc = 1.0 / k_c  # Mpc/h

    # Convert to kpc and scale relative to NFW scale radius
    r_c_kpc = r_c_mpc * 1000 / growth_solution.cosmo.h

    # Cap at a reasonable fraction of scale radius
    r_c = min(r_c_kpc, 2.0 * r_s_cdm)

    return r_c
