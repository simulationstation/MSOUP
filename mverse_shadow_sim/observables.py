"""
Observables Generator for Mverse Shadow Simulation

Generates simulated measurements in two windows:
1. Lensing window: weak-lensing shear profiles (and optional strong lensing)
2. Dynamics window: X-ray hydrostatic mass and galaxy velocity dispersion
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
from .config import SimConfig
from .truth import ClusterTruth, nfw_mass_enclosed


@dataclass
class LensingObservables:
    """Lensing observables for a cluster."""
    cluster_id: int

    # Weak lensing shear profile
    r_bins: np.ndarray  # Mpc, bin centers
    g_t: np.ndarray  # tangential shear (signal + noise)
    g_t_err: np.ndarray  # shear errors
    g_t_true: np.ndarray  # true shear without noise

    # Calibration factor applied
    m_cal: float

    # Strong lensing (optional)
    has_strong_lensing: bool
    M_einstein: Optional[float] = None  # Mass within Einstein radius
    M_einstein_err: Optional[float] = None
    r_einstein: Optional[float] = None


@dataclass
class DynamicsObservables:
    """Dynamical mass observables for a cluster."""
    cluster_id: int

    # X-ray hydrostatic mass
    M_X_r500: float  # Observed X-ray mass at r500
    M_X_half_r500: float  # at 0.5*r500
    M_X_err_r500: float  # fractional error
    M_X_err_half_r500: float
    b_hse_true: float  # true hydrostatic bias (unknown to analyst)

    # Velocity dispersion
    sigma_v: float  # km/s, observed
    sigma_v_err: float
    sigma_v_true: float  # true value without noise/bias
    b_aniso_true: float  # true anisotropy bias


def nfw_shear_profile(r: np.ndarray, M200: float, c: float, r200: float,
                      z: float, Sigma_crit: float) -> np.ndarray:
    """
    NFW tangential shear profile (approximate).

    For weak lensing: gamma_t = (Sigma_bar - Sigma) / Sigma_crit

    Using the NFW surface density and mean surface density formulas.
    This is an approximation for speed.
    """
    rs = r200 / c
    x = r / rs

    # NFW characteristic surface density
    # Sigma_s = rho_s * rs where rho_s is the scale density
    delta_c = (200.0 / 3.0) * c**3 / (np.log(1 + c) - c / (1 + c))

    # For the shear, we use the approximation:
    # Delta_Sigma = Sigma_bar(<R) - Sigma(R)
    # For NFW this can be computed analytically but we use a simpler form

    # Approximate tangential shear using mass enclosed
    M_enc = nfw_mass_enclosed(r, M200, c, r200)
    Sigma_bar = M_enc / (np.pi * r**2)

    # Surface density at r (approximate)
    # Sigma(r) ~ M(<r) / (2*pi*r^2) for outer regions, use a correction
    # For simplicity, use Delta_Sigma ~ 0.5 * Sigma_bar for typical NFW
    f_x = np.where(x < 1,
                   (1 - 2 / np.sqrt(1 - x**2) * np.arctanh(np.sqrt((1 - x) / (1 + x)))) / (x**2 - 1),
                   np.where(x > 1,
                            (1 - 2 / np.sqrt(x**2 - 1) * np.arctan(np.sqrt((x - 1) / (1 + x)))) / (x**2 - 1),
                            1.0 / 3.0))  # x = 1 limit

    # Simplified: gamma_t ~ M_enc / (pi * r^2 * Sigma_crit) * correction
    # The correction accounts for the difference between Sigma_bar and Sigma
    gamma_t = Sigma_bar / Sigma_crit * (1.0 - f_x * 0.5)

    # Clip to reasonable values
    gamma_t = np.clip(gamma_t, 0, 0.5)

    return gamma_t


def compute_sigma_crit(z_lens: float, z_source: float = 1.0,
                       H0: float = 70.0, Om0: float = 0.3) -> float:
    """
    Compute critical surface density in M_sun/Mpc^2.

    Sigma_crit = c^2 / (4 * pi * G) * D_s / (D_l * D_ls)
    """
    # Simplified angular diameter distances (flat universe)
    c_km_s = 299792.458
    dH = c_km_s / H0  # Mpc

    def comoving_distance(z):
        # Simple integration (trapezoidal)
        z_arr = np.linspace(0, z, 100)
        E_arr = np.sqrt(Om0 * (1 + z_arr)**3 + (1 - Om0))
        return np.trapezoid(dH / E_arr, z_arr)

    D_l = comoving_distance(z_lens) / (1 + z_lens)
    D_s = comoving_distance(z_source) / (1 + z_source)
    D_ls = (comoving_distance(z_source) - comoving_distance(z_lens)) / (1 + z_source)

    # G in Mpc^3 / M_sun / s^2: G = 4.518e-48
    # c^2 / (4 * pi * G) in M_sun / Mpc
    c_mpc_s = c_km_s / 3.086e19  # c in Mpc/s
    G_mpc = 4.518e-48

    Sigma_crit = c_mpc_s**2 / (4 * np.pi * G_mpc) * D_s / (D_l * D_ls)

    return Sigma_crit


def generate_lensing_observables(cluster: ClusterTruth,
                                  config: SimConfig,
                                  rng: np.random.Generator) -> LensingObservables:
    """Generate lensing observables for a single cluster."""
    lc = config.lensing

    # Radial bins
    r_bins = np.logspace(np.log10(lc.r_min_mpc), np.log10(lc.r_max_mpc), lc.n_bins)

    # Critical surface density
    Sigma_crit = compute_sigma_crit(cluster.z)

    # True shear profile
    g_t_true = nfw_shear_profile(r_bins, cluster.M200, cluster.c200,
                                  cluster.r200, cluster.z, Sigma_crit)

    # Apply projection boost (triaxiality)
    g_t_true = g_t_true * cluster.proj_boost

    # Shape noise
    n_source = lc.n_source_per_bin_base * (cluster.M200 / 1e14)**0.3
    g_err = lc.sigma_shape / np.sqrt(n_source)

    # Multiplicative calibration bias
    m_cal = rng.normal(lc.m_cal_mean, lc.m_cal_scatter)

    # Observed shear with noise
    g_t_obs = g_t_true * (1 + m_cal) + rng.normal(0, g_err, lc.n_bins)
    g_t_obs = np.maximum(g_t_obs, 0)  # shear should be positive

    # Strong lensing (subset)
    has_sl = rng.random() < lc.strong_lensing_fraction
    M_einstein = None
    M_einstein_err = None
    r_einstein = None

    if has_sl:
        # Einstein radius roughly proportional to sqrt(M200)
        r_einstein = 0.1 * np.sqrt(cluster.M200 / 1e15)  # Mpc
        M_einstein_true = nfw_mass_enclosed(np.array([r_einstein]),
                                             cluster.M200, cluster.c200,
                                             cluster.r200)[0]
        M_einstein = M_einstein_true * (1 + rng.normal(0, lc.einstein_mass_scatter))
        M_einstein_err = M_einstein_true * lc.einstein_mass_scatter

    return LensingObservables(
        cluster_id=cluster.cluster_id,
        r_bins=r_bins,
        g_t=g_t_obs,
        g_t_err=np.full(lc.n_bins, g_err),
        g_t_true=g_t_true,
        m_cal=m_cal,
        has_strong_lensing=has_sl,
        M_einstein=M_einstein,
        M_einstein_err=M_einstein_err,
        r_einstein=r_einstein
    )


def generate_dynamics_observables(cluster: ClusterTruth,
                                   config: SimConfig,
                                   rng: np.random.Generator) -> DynamicsObservables:
    """Generate dynamical mass observables for a single cluster."""
    dc = config.dynamics
    cosmo = config.cosmology

    # Hydrostatic bias
    delta_log_mass = np.log10(cluster.M200) - 14.5
    delta_z = cluster.z - 0.3

    b_hse = (dc.b_hse_mean +
             dc.b_hse_mass_slope * delta_log_mass +
             dc.b_hse_z_slope * delta_z +
             rng.normal(0, dc.b_hse_scatter))
    b_hse = np.clip(b_hse, dc.b_hse_min, dc.b_hse_max)

    # X-ray hydrostatic mass (biased low)
    M_X_r500_true = (1 - b_hse) * cluster.M_tot_r500
    M_X_half_true = (1 - b_hse) * cluster.M_tot_half_r500

    # Add measurement noise
    M_X_r500 = M_X_r500_true * (1 + rng.normal(0, dc.xray_noise))
    M_X_half = M_X_half_true * (1 + rng.normal(0, dc.xray_noise))

    M_X_r500 = max(M_X_r500, 1e12)  # floor
    M_X_half = max(M_X_half, 1e11)

    # Velocity dispersion
    # sigma_v = A * (M200)^(1/3) * E(z)^(1/3)
    E_z = cosmo.E(cluster.z)
    sigma_v_true = dc.sigma_v_A * (cluster.M200 / 1e15)**(dc.sigma_v_exp) * E_z**(dc.sigma_v_exp)

    # Anisotropy bias
    b_aniso = rng.normal(dc.b_aniso_mean, dc.b_aniso_scatter)

    # Observed velocity dispersion
    sigma_v_obs = sigma_v_true * (1 + b_aniso) * (1 + rng.normal(0, dc.vel_noise))
    sigma_v_obs = max(sigma_v_obs, 100)  # floor

    return DynamicsObservables(
        cluster_id=cluster.cluster_id,
        M_X_r500=M_X_r500,
        M_X_half_r500=M_X_half,
        M_X_err_r500=dc.xray_noise,
        M_X_err_half_r500=dc.xray_noise,
        b_hse_true=b_hse,
        sigma_v=sigma_v_obs,
        sigma_v_err=dc.vel_noise * sigma_v_obs,
        sigma_v_true=sigma_v_true,
        b_aniso_true=b_aniso
    )


def generate_all_observables(clusters: list, config: SimConfig,
                              rng: Optional[np.random.Generator] = None
                              ) -> Tuple[list, list]:
    """
    Generate all observables for a population of clusters.

    Returns
    -------
    lensing_obs : list of LensingObservables
    dynamics_obs : list of DynamicsObservables
    """
    if rng is None:
        rng = np.random.default_rng(config.seed + 1000)

    lensing_obs = []
    dynamics_obs = []

    for cluster in clusters:
        lensing_obs.append(generate_lensing_observables(cluster, config, rng))
        dynamics_obs.append(generate_dynamics_observables(cluster, config, rng))

    return lensing_obs, dynamics_obs


def generate_all_observables_vectorized(clusters: list, config: SimConfig,
                                         rng: Optional[np.random.Generator] = None
                                         ) -> Tuple[Dict, Dict]:
    """
    Vectorized observable generation for fast mode.

    Returns dictionaries of arrays instead of lists of dataclasses.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed + 1000)

    N = len(clusters)
    lc = config.lensing
    dc = config.dynamics
    cosmo = config.cosmology

    # Extract arrays from clusters
    z_arr = np.array([c.z for c in clusters])
    M200_arr = np.array([c.M200 for c in clusters])
    c200_arr = np.array([c.c200 for c in clusters])
    r200_arr = np.array([c.r200 for c in clusters])
    r500_arr = np.array([c.r500 for c in clusters])
    proj_boost_arr = np.array([c.proj_boost for c in clusters])
    M_tot_r500_arr = np.array([c.M_tot_r500 for c in clusters])
    M_tot_half_arr = np.array([c.M_tot_half_r500 for c in clusters])

    # Lensing: simplified mass proxy (aperture mass at fixed radius)
    # Instead of fitting shear profiles, use direct mass measurement with scatter
    # M_lens = M_true * proj_boost * (1 + m_cal) * (1 + noise)
    m_cal_arr = rng.normal(lc.m_cal_mean, lc.m_cal_scatter, N)

    # Shape noise contribution (effective noise on mass)
    mass_noise_frac = 0.15 / np.sqrt(N / 1000)  # typical weak lensing mass error

    M_lens_r500 = (M_tot_r500_arr * proj_boost_arr *
                   (1 + m_cal_arr) *
                   (1 + rng.normal(0, mass_noise_frac, N)))

    M_lens_half = (M_tot_half_arr * proj_boost_arr *
                   (1 + m_cal_arr) *
                   (1 + rng.normal(0, mass_noise_frac * 1.5, N)))  # more noise at smaller r

    # Dynamics: X-ray hydrostatic
    delta_log_mass = np.log10(M200_arr) - 14.5
    delta_z = z_arr - 0.3

    b_hse_arr = (dc.b_hse_mean +
                 dc.b_hse_mass_slope * delta_log_mass +
                 dc.b_hse_z_slope * delta_z +
                 rng.normal(0, dc.b_hse_scatter, N))
    b_hse_arr = np.clip(b_hse_arr, dc.b_hse_min, dc.b_hse_max)

    M_X_r500 = ((1 - b_hse_arr) * M_tot_r500_arr *
                (1 + rng.normal(0, dc.xray_noise, N)))
    M_X_half = ((1 - b_hse_arr) * M_tot_half_arr *
                (1 + rng.normal(0, dc.xray_noise, N)))

    M_X_r500 = np.maximum(M_X_r500, 1e12)
    M_X_half = np.maximum(M_X_half, 1e11)

    # Dynamics: velocity dispersion
    E_z = np.array([cosmo.E(z) for z in z_arr])
    sigma_v_true = dc.sigma_v_A * (M200_arr / 1e15)**dc.sigma_v_exp * E_z**dc.sigma_v_exp

    b_aniso_arr = rng.normal(dc.b_aniso_mean, dc.b_aniso_scatter, N)
    sigma_v_obs = sigma_v_true * (1 + b_aniso_arr) * (1 + rng.normal(0, dc.vel_noise, N))
    sigma_v_obs = np.maximum(sigma_v_obs, 100)

    lensing = {
        'M_lens_r500': M_lens_r500,
        'M_lens_half_r500': M_lens_half,
        'm_cal': m_cal_arr,
        'proj_boost': proj_boost_arr,
    }

    dynamics = {
        'M_X_r500': M_X_r500,
        'M_X_half_r500': M_X_half,
        'b_hse_true': b_hse_arr,
        'sigma_v': sigma_v_obs,
        'sigma_v_true': sigma_v_true,
        'b_aniso_true': b_aniso_arr,
    }

    return lensing, dynamics
