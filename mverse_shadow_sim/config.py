"""
Configuration for Mverse Shadow Simulation

All physical and observational parameters are defined here for reproducibility.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import numpy as np


@dataclass
class CosmologyConfig:
    """Cosmological parameters (Planck 2018)."""
    H0: float = 70.0  # km/s/Mpc
    Om0: float = 0.3
    Ob0: float = 0.05
    sigma8: float = 0.81

    @property
    def h(self) -> float:
        return self.H0 / 100.0

    def E(self, z: float) -> float:
        """Hubble parameter E(z) = H(z)/H0."""
        return np.sqrt(self.Om0 * (1 + z)**3 + (1 - self.Om0))

    def rho_crit(self, z: float) -> float:
        """Critical density in M_sun/Mpc^3."""
        # rho_crit,0 = 3H0^2/(8*pi*G) ~ 2.775e11 h^2 M_sun/Mpc^3
        rho_crit_0 = 2.775e11 * self.h**2
        return rho_crit_0 * self.E(z)**2


@dataclass
class TruthConfig:
    """Configuration for the truth generator."""
    # Redshift distribution
    z_min: float = 0.1
    z_max: float = 0.8

    # Mass distribution (log10(M200/Msun))
    log_mass_mean: float = 14.5
    log_mass_std: float = 0.4
    log_mass_min: float = 13.5
    log_mass_max: float = 15.5

    # Concentration-mass relation (Duffy et al. 2008 like)
    # c = A * (M/Mpiv)^B * (1+z)^C
    c_A: float = 5.71
    c_B: float = -0.084
    c_C: float = -0.47
    c_Mpiv: float = 2e12  # M_sun
    c_scatter: float = 0.15  # dex (lognormal)

    # Triaxiality projection boost for lensing
    # Lensing mass is boosted by (1 + delta_proj) with lognormal scatter
    proj_boost_scatter: float = 0.12  # dex

    # Baryon fractions
    f_gas_mean: float = 0.10
    f_gas_scatter: float = 0.02
    f_gas_mass_slope: float = 0.02  # per dex in M
    f_gas_z_slope: float = -0.01  # per unit z

    f_star_mean: float = 0.015
    f_star_scatter: float = 0.005

    # Universal gravity coupling (the truth)
    alpha_true: float = 1.0
    # Optional drift toggle (for testing K2 detection)
    alpha_z_drift: float = 0.0  # d(alpha)/dz, default 0 = universal

    # Evaluation radii (as fractions of r500)
    eval_radii_r500_frac: List[float] = field(default_factory=lambda: [0.5, 1.0])


@dataclass
class LensingConfig:
    """Configuration for weak lensing observables."""
    # Radial bins for shear profile
    r_min_mpc: float = 0.2
    r_max_mpc: float = 2.0
    n_bins: int = 10

    # Shape noise
    sigma_shape: float = 0.25
    n_source_per_bin_base: float = 50.0  # scales with mass/z

    # Multiplicative calibration bias
    m_cal_mean: float = 0.0
    m_cal_scatter: float = 0.02

    # Strong lensing subset
    strong_lensing_fraction: float = 0.2
    einstein_mass_scatter: float = 0.10  # fractional


@dataclass
class DynamicsConfig:
    """Configuration for dynamical mass observables."""
    # Hydrostatic bias: M_X = (1 - b_hse) * M_tot
    b_hse_mean: float = 0.20
    b_hse_mass_slope: float = 0.05  # per dex in M
    b_hse_z_slope: float = 0.1  # per unit z
    b_hse_scatter: float = 0.08
    b_hse_min: float = 0.0
    b_hse_max: float = 0.5

    # X-ray measurement noise (fractional)
    xray_noise: float = 0.08

    # Velocity dispersion scaling: sigma_v = A * (M200)^(1/3) * E(z)^(1/3)
    sigma_v_A: float = 1082.9  # km/s for M in M_sun (tuned)
    sigma_v_exp: float = 1.0 / 3.0

    # Anisotropy bias: measured sigma = true * (1 + b_aniso)
    b_aniso_mean: float = 0.0
    b_aniso_scatter: float = 0.07

    # Velocity measurement noise (fractional)
    vel_noise: float = 0.07


@dataclass
class InferenceConfig:
    """Configuration for M2 analyst inference pipelines."""
    # Lensing inference
    use_concentration_prior: bool = True
    c_prior_scatter: float = 0.20  # dex

    # X-ray inference (analyst assumes zero bias)
    assumed_b_hse: float = 0.0

    # Velocity inference (analyst assumes isotropy)
    assumed_b_aniso: float = 0.0

    # Visible mass inference biases
    f_gas_inferred_bias: float = 0.05  # fractional
    f_gas_inferred_scatter: float = 0.10
    f_star_inferred_scatter: float = 0.20


@dataclass
class SimConfig:
    """Master simulation configuration."""
    n_clusters: int = 10000
    seed: int = 42
    fast_mode: bool = True

    cosmology: CosmologyConfig = field(default_factory=CosmologyConfig)
    truth: TruthConfig = field(default_factory=TruthConfig)
    lensing: LensingConfig = field(default_factory=LensingConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)

    def __post_init__(self):
        # Validate
        assert self.n_clusters > 0
        assert self.truth.alpha_true > 0


def get_default_config(n_clusters: int = 10000,
                       seed: int = 42,
                       fast_mode: bool = True) -> SimConfig:
    """Get default simulation configuration."""
    return SimConfig(n_clusters=n_clusters, seed=seed, fast_mode=fast_mode)
