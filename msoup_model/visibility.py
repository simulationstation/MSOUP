"""
Visibility and effective response functions for Msoup closure.

The minimal closure model has:
  - V(M; kappa) = exp[-kappa * (M - 2)]: visibility function
  - M_vis(z) = 2 + Delta_M / (1 + exp[(z - z_t) / w]): visible order statistic
  - c_eff^2(z): effective "viscosity" tied to M_vis via fixed monotone map

Parameters:
  kappa: visibility decay rate (kept in code but absorbed into effective params)
  Delta_M: amplitude above threshold M=2
  z_t: transition redshift (when M_vis rises most rapidly)
  w: transition width
  c_star_sq: maximum effective sound speed squared
  Delta_threshold: threshold for sigmoid mapping (default: small value)
  sigma_map: width of sigmoid mapping (default: 0.1)
"""

import numpy as np
from dataclasses import dataclass
from typing import Union, Optional

ArrayLike = Union[float, np.ndarray]


@dataclass
class MsoupParams:
    """
    Global Msoup closure parameters.

    These are the ONLY free parameters in the minimal model:
      - Delta_M: amplitude of M_vis rise above 2
      - z_t: transition redshift
      - w: transition width (>0)
      - c_star_sq: effective sound speed squared (km/s)^2

    Fixed/absorbed parameters:
      - kappa: visibility decay (kept for documentation, absorbed)
      - Delta_threshold: sigmoid threshold offset (fixed)
      - sigma_map: sigmoid width (fixed)
    """
    Delta_M: float = 0.5        # Amplitude of M_vis rise
    z_t: float = 2.0            # Transition redshift
    w: float = 0.5              # Transition width
    c_star_sq: float = 100.0    # (km/s)^2, suppression strength

    # Fixed parameters (not varied in fits)
    kappa: float = 1.0          # Absorbed into Delta_M effectively
    Delta_threshold: float = 0.1  # Small offset in sigmoid
    sigma_map: float = 0.1      # Sigmoid width

    def __post_init__(self):
        """Validate parameters."""
        if self.Delta_M < 0:
            raise ValueError("Delta_M must be >= 0")
        if self.w <= 0:
            raise ValueError("w must be > 0")
        if self.c_star_sq < 0:
            raise ValueError("c_star_sq must be >= 0")

    def to_array(self) -> np.ndarray:
        """Return fitted parameters as array [c_star_sq, Delta_M, z_t, w]."""
        return np.array([self.c_star_sq, self.Delta_M, self.z_t, self.w])

    @classmethod
    def from_array(cls, arr: np.ndarray, **fixed_kwargs) -> "MsoupParams":
        """Create from array [c_star_sq, Delta_M, z_t, w]."""
        return cls(
            c_star_sq=arr[0],
            Delta_M=arr[1],
            z_t=arr[2],
            w=arr[3],
            **fixed_kwargs
        )

    @staticmethod
    def param_names() -> list:
        """Names of fitted parameters."""
        return ["c_star_sq", "Delta_M", "z_t", "w"]

    @staticmethod
    def param_labels() -> list:
        """LaTeX labels for plotting."""
        return [r"$c_*^2$", r"$\Delta M$", r"$z_t$", r"$w$"]

    @staticmethod
    def get_prior_bounds() -> dict:
        """Prior bounds for MCMC sampling."""
        return {
            "c_star_sq": (0.0, 1000.0),   # (km/s)^2
            "Delta_M": (0.0, 2.0),         # Amplitude
            "z_t": (0.5, 10.0),            # Transition z
            "w": (0.05, 2.0),              # Width
        }


def visibility(M: ArrayLike, kappa: float = 1.0) -> ArrayLike:
    """
    Visibility function V(M; kappa) = exp[-kappa * (M - 2)].

    This quantifies how "visible" or measurable phenomena at order M are.
    M=2 is the threshold where visibility = 1.
    Higher M -> exponentially suppressed visibility.

    Parameters:
        M: order statistic (scalar or array)
        kappa: decay rate (default 1.0, absorbed into effective params)

    Returns:
        V(M): visibility in [0, 1]
    """
    return np.exp(-kappa * np.maximum(M - 2, 0))


def m_vis(z: ArrayLike, params: MsoupParams) -> ArrayLike:
    """
    Visible order statistic M_vis(z).

    M_vis(z) = 2 + Delta_M / (1 + exp[(z - z_t) / w])

    Properties:
      - At high z >> z_t: M_vis -> 2 (threshold, full visibility)
      - At low z << z_t: M_vis -> 2 + Delta_M (above threshold)
      - Monotonically decreasing with z (increasing toward present)
      - Smooth transition around z = z_t with width w

    Parameters:
        z: redshift (scalar or array)
        params: MsoupParams instance

    Returns:
        M_vis(z): visible order statistic
    """
    z = np.asarray(z)
    # Logistic turn-on: approaches 2 at high z, 2+Delta_M at low z
    sigmoid_arg = (z - params.z_t) / params.w
    # Clip to avoid overflow
    sigmoid_arg = np.clip(sigmoid_arg, -50, 50)
    return 2.0 + params.Delta_M / (1.0 + np.exp(sigmoid_arg))


def c_eff_squared(z: ArrayLike, params: MsoupParams) -> ArrayLike:
    """
    Effective sound speed squared c_eff^2(z) in (km/s)^2.

    This is the single response channel: a viscosity-like term that
    suppresses small-scale growth.

    c_eff^2(z) = c_*^2 * sigmoid((M_vis(z) - 2 - Delta_threshold) / sigma_map)

    Properties:
      - When M_vis ~ 2: c_eff^2 -> 0 (no suppression, standard LCDM)
      - When M_vis > 2: c_eff^2 -> c_*^2 (maximum suppression)
      - Tied to M_vis(z) with a fixed sigmoid map

    Parameters:
        z: redshift (scalar or array)
        params: MsoupParams instance

    Returns:
        c_eff^2(z) in (km/s)^2
    """
    z = np.asarray(z)
    M = m_vis(z, params)

    # Sigmoid mapping from M_vis to c_eff^2
    sigmoid_arg = (M - 2.0 - params.Delta_threshold) / params.sigma_map
    sigmoid_arg = np.clip(sigmoid_arg, -50, 50)
    sigmoid_val = 1.0 / (1.0 + np.exp(-sigmoid_arg))

    return params.c_star_sq * sigmoid_val


def suppression_scale_k(z: ArrayLike, params: MsoupParams,
                        H_z: Optional[ArrayLike] = None) -> ArrayLike:
    """
    Characteristic suppression wavenumber k_s(z) in h/Mpc.

    k_s ~ H(z) / c_eff(z)

    Above this scale (k > k_s), growth is suppressed.

    Parameters:
        z: redshift
        params: MsoupParams
        H_z: Hubble parameter at z in km/s/Mpc (if None, uses simple approximation)

    Returns:
        k_s(z) in h/Mpc
    """
    z = np.asarray(z)
    c_eff_sq = c_eff_squared(z, params)

    if H_z is None:
        # Simple LCDM approximation: H(z) ~ H0 * sqrt(Omega_m*(1+z)^3 + Omega_Lambda)
        H0 = 70.0  # km/s/Mpc
        Om = 0.3
        H_z = H0 * np.sqrt(Om * (1 + z)**3 + (1 - Om))

    c_eff = np.sqrt(np.maximum(c_eff_sq, 1e-10))  # km/s

    # k_s = H / c_eff, convert to h/Mpc
    # H is in km/s/Mpc, c is in km/s
    # k = H/c has units 1/Mpc, multiply by h^-1 to get h/Mpc
    h = 0.7
    k_s = H_z / c_eff / h

    return k_s


def half_mode_mass(z: float, params: MsoupParams,
                   rho_m0: float = 2.775e11) -> float:
    """
    Half-mode mass M_hm(z): mass scale where power is suppressed by 50%.

    This is a key observable for comparison with lensing constraints.

    M_hm ~ (4/3) * pi * rho_m * (pi / k_s)^3

    Parameters:
        z: redshift
        params: MsoupParams
        rho_m0: present matter density in h^2 M_sun / Mpc^3

    Returns:
        M_hm in M_sun/h
    """
    k_s = suppression_scale_k(z, params)

    # Characteristic radius: lambda_s / 2 = pi / k_s (in Mpc/h)
    R_s = np.pi / k_s

    # Mass enclosed
    M_hm = (4.0 / 3.0) * np.pi * rho_m0 * R_s**3

    return M_hm
