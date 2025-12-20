"""
Cosmology utilities for Msoup closure model.

Uses standard ΛCDM background cosmology as baseline.
The Msoup model only modifies perturbation growth, not the background.
"""

import numpy as np
from dataclasses import dataclass
from scipy.integrate import quad
from typing import Union

ArrayLike = Union[float, np.ndarray]


@dataclass
class CosmologyParams:
    """
    Standard ΛCDM cosmological parameters.

    Default values from Planck 2018.
    """
    h: float = 0.6736          # Hubble parameter h = H0/(100 km/s/Mpc)
    Omega_m: float = 0.3153    # Total matter density
    Omega_b: float = 0.0493    # Baryon density
    Omega_Lambda: float = 0.6847  # Dark energy density
    sigma8: float = 0.8111     # Power spectrum normalization
    n_s: float = 0.9649        # Spectral index
    T_cmb: float = 2.7255      # CMB temperature in K

    @property
    def H0(self) -> float:
        """Hubble constant in km/s/Mpc."""
        return 100.0 * self.h

    @property
    def Omega_cdm(self) -> float:
        """Cold dark matter density."""
        return self.Omega_m - self.Omega_b

    @property
    def rho_crit_0(self) -> float:
        """Critical density at z=0 in h^2 M_sun / Mpc^3."""
        # rho_crit = 3 H0^2 / (8 pi G)
        # In units of h^2 M_sun / Mpc^3
        return 2.775e11

    @property
    def rho_m_0(self) -> float:
        """Mean matter density at z=0 in h^2 M_sun / Mpc^3."""
        return self.Omega_m * self.rho_crit_0

    def H(self, z: ArrayLike) -> ArrayLike:
        """
        Hubble parameter H(z) in km/s/Mpc.

        H(z) = H0 * sqrt(Omega_m * (1+z)^3 + Omega_Lambda)
        """
        z = np.asarray(z)
        return self.H0 * np.sqrt(
            self.Omega_m * (1 + z)**3 + self.Omega_Lambda
        )

    def E(self, z: ArrayLike) -> ArrayLike:
        """
        Dimensionless Hubble parameter E(z) = H(z)/H0.
        """
        z = np.asarray(z)
        return np.sqrt(self.Omega_m * (1 + z)**3 + self.Omega_Lambda)

    def Omega_m_z(self, z: ArrayLike) -> ArrayLike:
        """Matter density parameter at redshift z."""
        z = np.asarray(z)
        Ez = self.E(z)
        return self.Omega_m * (1 + z)**3 / Ez**2

    def comoving_distance(self, z: float) -> float:
        """
        Comoving distance to redshift z in Mpc/h.
        """
        c_over_H0 = 2997.92458 / self.h  # c / H0 in Mpc/h

        def integrand(zp):
            return 1.0 / self.E(zp)

        result, _ = quad(integrand, 0, z)
        return c_over_H0 * result

    def angular_diameter_distance(self, z: float) -> float:
        """Angular diameter distance in Mpc/h."""
        return self.comoving_distance(z) / (1 + z)

    def growth_factor_lcdm(self, z: ArrayLike) -> ArrayLike:
        """
        Standard ΛCDM growth factor D(z), normalized to D(0)=1.

        Uses the fitting formula from Carroll, Press & Turner (1992).
        """
        z = np.asarray(z)
        a = 1.0 / (1 + z)

        Om_z = self.Omega_m_z(z)
        Ol_z = 1 - Om_z  # Approximate for flat universe

        # CPT fitting formula
        g = 2.5 * Om_z / (
            Om_z**(4.0/7.0) - Ol_z +
            (1 + Om_z/2) * (1 + Ol_z/70)
        )

        # Normalize
        g0 = 2.5 * self.Omega_m / (
            self.Omega_m**(4.0/7.0) - self.Omega_Lambda +
            (1 + self.Omega_m/2) * (1 + self.Omega_Lambda/70)
        )

        return (g / g0) * a

    def rho_m(self, z: ArrayLike) -> ArrayLike:
        """Matter density at z in h^2 M_sun / Mpc^3."""
        z = np.asarray(z)
        return self.rho_m_0 * (1 + z)**3

    def delta_c(self, z: float = 0) -> float:
        """
        Critical overdensity for spherical collapse.

        Weakly z-dependent; use approximation from Nakamura & Suto (1997).
        """
        Om_z = self.Omega_m_z(z)
        return 0.15 * (12 * np.pi)**(2.0/3.0) * Om_z**0.0055

    def time_from_z(self, z: float) -> float:
        """
        Cosmic time at redshift z in Gyr.
        """
        H0_per_Gyr = self.H0 * 1.0227e-3  # Convert km/s/Mpc to 1/Gyr

        def integrand(zp):
            return 1.0 / ((1 + zp) * self.E(zp))

        result, _ = quad(integrand, z, np.inf)
        return result / H0_per_Gyr

    def z_from_time(self, t: float, z_max: float = 1000) -> float:
        """
        Redshift at cosmic time t (in Gyr).
        Uses bisection search.
        """
        from scipy.optimize import brentq

        def func(z):
            return self.time_from_z(z) - t

        return brentq(func, 0, z_max)
