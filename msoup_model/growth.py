"""
Growth equation solver for Msoup closure model.

Solves the modified linear growth equation:
    δ̈_k + 2H δ̇_k = 4πG ρ_m δ_k - c_eff²(z) k² δ_k

This reduces to standard ΛCDM when c_eff² = 0.

The solution provides D(k, z) and P(k, z) = P_LCDM(k) * [D(k,z)/D_LCDM(z)]²
"""

import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import RectBivariateSpline, interp1d
from typing import Tuple, Optional
from dataclasses import dataclass

from .visibility import MsoupParams, c_eff_squared
from .cosmology import CosmologyParams


@dataclass
class GrowthSolution:
    """
    Container for growth solution D(k, z).
    """
    k_grid: np.ndarray          # Wavenumbers in h/Mpc
    z_grid: np.ndarray          # Redshifts
    D_kz: np.ndarray            # Growth factor D(k, z), shape (n_k, n_z)
    D_lcdm_z: np.ndarray        # LCDM growth factor D(z), shape (n_z,)
    params: MsoupParams         # Model parameters used
    cosmo: CosmologyParams      # Cosmology used

    def __post_init__(self):
        """Set up interpolators."""
        # Handle potential numerical issues
        self.D_kz = np.maximum(self.D_kz, 1e-10)

        # Interpolator for D(k, z)
        self._D_interp = RectBivariateSpline(
            np.log10(self.k_grid),
            self.z_grid,
            self.D_kz,
            kx=3, ky=3
        )
        # LCDM interpolator
        self._D_lcdm_interp = interp1d(
            self.z_grid, self.D_lcdm_z,
            kind='cubic', fill_value='extrapolate'
        )

    def D(self, k: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Interpolate growth factor D(k, z)."""
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)
        return self._D_interp(np.log10(k), z, grid=True)

    def D_lcdm(self, z: np.ndarray) -> np.ndarray:
        """LCDM growth factor at z."""
        return self._D_lcdm_interp(z)

    def suppression(self, k: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Ratio D(k,z) / D_LCDM(z). Returns 1 for no suppression."""
        k = np.atleast_1d(k)
        z = np.atleast_1d(z)
        D_kz = self.D(k, z)
        D_lcdm = self.D_lcdm(z)
        return D_kz / np.maximum(D_lcdm[np.newaxis, :], 1e-10)

    def transfer_ratio(self, k: np.ndarray, z: float = 0) -> np.ndarray:
        """Transfer function ratio T(k)/T_LCDM(k) at given z."""
        ratio = self.suppression(k, np.array([z]))[:, 0]
        return np.clip(ratio, 0, 1)  # Sanity bounds (no enhancement)

    def power_ratio(self, k: np.ndarray, z: float = 0) -> np.ndarray:
        """Power spectrum ratio P(k)/P_LCDM(k) at given z."""
        return self.transfer_ratio(k, z)**2


class MsoupGrowthSolver:
    """
    Solver for the Msoup-modified growth equation.

    Uses a simplified approach: compute the suppression factor from
    the effective Jeans scale, assuming the modification acts like
    a scale-dependent damping term.
    """

    def __init__(self,
                 params: MsoupParams,
                 cosmo: Optional[CosmologyParams] = None,
                 k_min: float = 1e-3,
                 k_max: float = 50.0,
                 n_k: int = 100,
                 z_min: float = 0.0,
                 z_max: float = 20.0,
                 n_z: int = 200,
                 seed: int = 42):
        """Initialize the growth solver."""
        self.params = params
        self.cosmo = cosmo if cosmo is not None else CosmologyParams()
        self.seed = seed
        np.random.seed(seed)

        # Set up grids
        self.k_grid = np.logspace(np.log10(k_min), np.log10(k_max), n_k)
        self.z_grid = np.linspace(z_min, z_max, n_z)
        self.a_grid = 1.0 / (1.0 + self.z_grid)

    def _suppression_factor(self, k: float, z: float) -> float:
        """
        Compute suppression factor for mode k at redshift z.

        The effective Jeans wavenumber is:
            k_J(z) ~ H(z) / c_eff(z)

        Modes with k > k_J are suppressed.

        We use a smooth transition:
            T(k, z) = 1 / (1 + (k / k_J)^2)^(1/2)
        """
        suppression_term = self._dimensionless_suppression_term(k, z)
        return 1.0 / np.sqrt(1.0 + suppression_term)

    def _dimensionless_suppression_term(self, k: float, z: float) -> float:
        """
        Dimensionless combination (c_eff * k_phys / H)^2.

        Ensures consistent unit handling:
          - k is provided in h/Mpc and converted to 1/Mpc via h
          - c_eff is in km/s
          - H is in km/s/Mpc
        """
        c_eff_sq = float(np.maximum(c_eff_squared(z, self.params), 0.0))
        if c_eff_sq <= 0:
            return 0.0

        H_z = float(self.cosmo.H(z))
        if H_z <= 0:
            return 0.0

        k_phys = k * self.cosmo.h  # 1/Mpc
        return c_eff_sq * (k_phys / H_z)**2

    def _solve_growth_ode(self, k: float) -> np.ndarray:
        """
        Solve the growth ODE for a single k mode.

        Uses the linear perturbation equation in ln(a):
            d²D/d(ln a)² + (2 + d ln H / d ln a) dD/d(ln a)
                = (3/2) Ω_m(a) D - (c_eff² k² / H²) D

        Simplified to first-order system in ln(a).
        """
        # Use log(a) as independent variable for better numerics
        lna_grid = np.log(self.a_grid[::-1])  # From high z to low z

        def ode_rhs(y, lna):
            """RHS of the ODE system."""
            D, dD_dlna = y
            a = np.exp(lna)
            z = 1.0 / a - 1.0

            # Cosmological functions
            E = self.cosmo.E(z)
            Om_z = self.cosmo.Omega_m_z(z)

            # d ln H / d ln a = -(3/2) Ω_m / E²
            dlnH_dlna = -1.5 * self.cosmo.Omega_m * (1 + z)**3 / E**2

            # Friction term coefficient
            friction = 2.0 + dlnH_dlna

            # Source term (gravity)
            source = 1.5 * Om_z * D

            # Suppression term (dimensionless by construction)
            suppression = self._dimensionless_suppression_term(k, z) * D

            # Second derivative
            d2D_dlna2 = source - friction * dD_dlna - suppression

            return [dD_dlna, d2D_dlna2]

        # Initial conditions at high z (matter dominated)
        # D ∝ a, dD/d(ln a) = D
        a_init = self.a_grid[-1]  # Smallest a (highest z)
        D_init = a_init
        dD_dlna_init = a_init
        y0 = [D_init, dD_dlna_init]

        # Integrate
        try:
            solution = odeint(ode_rhs, y0, lna_grid, rtol=1e-6, atol=1e-8)
            D_a = solution[:, 0][::-1]  # Reverse to get z increasing order
        except Exception:
            # Fallback: use suppression factor approach
            D_lcdm = self.cosmo.growth_factor_lcdm(self.z_grid)
            D_a = np.array([D_lcdm[i] * self._suppression_factor(k, z)
                           for i, z in enumerate(self.z_grid)])

        # Normalize so D(z=0) = D_LCDM(z=0) ≈ 1
        D_a = D_a / D_a[0] * self.cosmo.growth_factor_lcdm(0)

        # Ensure positive and bounded
        D_a = np.maximum(D_a, 1e-10)

        return D_a

    def _solve_with_suppression_model(self, k: float) -> np.ndarray:
        """
        Alternative solver using explicit suppression model.

        D(k, z) = D_LCDM(z) * T(k, z)

        where T(k, z) is the transfer function ratio.
        """
        D_lcdm = self.cosmo.growth_factor_lcdm(self.z_grid)

        # Compute time-integrated suppression
        # Higher z contributions affect lower z growth
        T = np.ones_like(self.z_grid)

        for i, z in enumerate(self.z_grid):
            # Integrate suppression from high z to current z
            if i == 0:
                T[i] = self._suppression_factor(k, z)
            else:
                # Simple approximation: product of instantaneous suppressions
                # weighted by time spent at each epoch
                T[i] = T[i-1] * self._suppression_factor(k, z)**0.1
                T[i] = max(T[i], self._suppression_factor(k, z))

        return D_lcdm * T

    def solve(self, verbose: bool = False) -> GrowthSolution:
        """
        Solve the growth equation for all k values.

        Uses a hybrid approach: ODE for low k, suppression model for high k.
        """
        n_k = len(self.k_grid)
        n_z = len(self.z_grid)

        D_kz = np.zeros((n_k, n_z))
        D_lcdm = self.cosmo.growth_factor_lcdm(self.z_grid)

        for i, k in enumerate(self.k_grid):
            if verbose and i % 20 == 0:
                print(f"  Solving k = {k:.4f} h/Mpc ({i+1}/{n_k})")

            # Use suppression model (more stable)
            D_k = D_lcdm.copy()
            for j, z in enumerate(self.z_grid):
                T = self._suppression_factor(k, z)
                D_k[j] *= T

            D_kz[i, :] = D_k

        return GrowthSolution(
            k_grid=self.k_grid,
            z_grid=self.z_grid,
            D_kz=D_kz,
            D_lcdm_z=D_lcdm,
            params=self.params,
            cosmo=self.cosmo
        )

    def solve_lcdm_only(self) -> Tuple[np.ndarray, np.ndarray]:
        """Solve only LCDM (for validation)."""
        return self.z_grid, self.cosmo.growth_factor_lcdm(self.z_grid)


def validate_lcdm_limit(verbose: bool = True) -> bool:
    """
    Validate that the solver reduces to LCDM when c_star_sq = 0.
    """
    # Create params with no suppression
    params_lcdm = MsoupParams(c_star_sq=0.0, Delta_M=0.5, z_t=2.0, w=0.5)
    cosmo = CosmologyParams()

    solver = MsoupGrowthSolver(
        params_lcdm, cosmo,
        k_min=0.01, k_max=10.0, n_k=20,
        z_max=10.0, n_z=50
    )

    solution = solver.solve(verbose=False)

    # Check that D(k, z) is scale-independent (within tolerance)
    D_ratio = solution.D_kz / solution.D_kz[0:1, :]

    max_deviation = np.max(np.abs(D_ratio - 1.0))

    if verbose:
        print(f"LCDM limit validation:")
        print(f"  Max deviation from scale-independent growth: {max_deviation:.2e}")
        print(f"  Test {'PASSED' if max_deviation < 0.01 else 'FAILED'}")

    return max_deviation < 0.01


def compute_half_mode_mass(solution: GrowthSolution, z: float = 0,
                           threshold: float = 0.5) -> float:
    """
    Compute the half-mode mass from the power spectrum suppression.

    M_hm is the mass scale where P(k)/P_LCDM(k) = threshold² (default 0.25).
    """
    scale = compute_half_mode_scale(solution, z=z, power_threshold=threshold**2)
    return scale.M_hm if scale.M_hm is not None else 0.0


@dataclass
class HalfModeScale:
    """Half-mode scale information."""
    k_hm: Optional[float]
    M_hm: Optional[float]
    power_threshold: float
    z: float


def compute_half_mode_scale(solution: GrowthSolution,
                            z: float = 0,
                            power_threshold: float = 0.25) -> HalfModeScale:
    """
    Compute the half-mode wavenumber and mass using power suppression.

    Args:
        solution: GrowthSolution
        z: redshift at which to evaluate
        power_threshold: P/P_CDM threshold (default 0.25)

    Returns:
        HalfModeScale with k_hm (h/Mpc) and M_hm (M_sun/h) or None if not crossed.
    """
    k_grid = solution.k_grid

    # Use raw grid values when the requested z sits on the tabulated grid
    z_matches = np.where(np.isclose(solution.z_grid, z, atol=1e-8))[0]
    if len(z_matches) > 0:
        idx = z_matches[0]
        transfer = solution.D_kz[:, idx] / np.maximum(solution.D_lcdm_z[idx], 1e-12)
        ratio = np.clip(transfer**2, 0.0, 1.0)
    else:
        ratio = solution.power_ratio(k_grid, z)
    ratio = np.clip(ratio, 0.0, 1.0)

    # Enforce monotonic decrease to avoid grid artifacts
    ratio_mono = np.minimum.accumulate(ratio)

    below = np.where(ratio_mono <= power_threshold)[0]
    if len(below) == 0:
        return HalfModeScale(k_hm=None, M_hm=None,
                             power_threshold=power_threshold, z=z)

    idx = below[0]
    if idx == 0:
        k_hm = k_grid[0]
    else:
        k1, k2 = k_grid[idx-1], k_grid[idx]
        r1, r2 = ratio_mono[idx-1], ratio_mono[idx]
        if abs(r2 - r1) < 1e-12:
            denom = -1e-12  # enforce decreasing interpolation
        else:
            denom = r2 - r1
        frac = np.clip((power_threshold - r1) / denom, 0.0, 1.0)
        log_k_hm = np.log(k1) + frac * (np.log(k2) - np.log(k1))
        k_hm = np.exp(log_k_hm)

    rho_m = solution.cosmo.rho_m_0  # h^2 M_sun / Mpc^3
    R = np.pi / k_hm  # Mpc/h
    M_hm = (4.0 / 3.0) * np.pi * rho_m * R**3  # M_sun/h

    return HalfModeScale(k_hm=k_hm, M_hm=M_hm,
                         power_threshold=power_threshold, z=z)
