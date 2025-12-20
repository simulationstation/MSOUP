"""
Truth Generator for Mverse Shadow Simulation

Generates a population of clusters with "true" physical parameters.
The key invariant is alpha_true = 1.0 (universal gravity channel).
"""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from .config import SimConfig, CosmologyConfig


@dataclass
class ClusterTruth:
    """True physical properties of a single cluster."""
    # Identifiers
    cluster_id: int

    # Fundamental properties
    z: float  # redshift
    M200: float  # M_sun, total mass within r200
    c200: float  # concentration

    # Derived radii (Mpc)
    r200: float
    r500: float
    rs: float  # scale radius

    # Baryon fractions
    f_gas: float
    f_star: float

    # Projection effects (for lensing)
    proj_boost: float  # multiplicative factor (1 + delta)

    # True masses at evaluation radii
    # Keys are r/r500 fractions, values are masses in M_sun
    M_tot_r500: float  # M_tot(<r500)
    M_tot_half_r500: float  # M_tot(<0.5*r500)
    M_vis_r500: float
    M_vis_half_r500: float
    M_exc_r500: float  # M_tot - M_vis at r500
    M_exc_half_r500: float

    # The universal coupling (always 1.0 in base truth)
    alpha_true: float = 1.0


def nfw_mass_enclosed(r: np.ndarray, M200: float, c: float, r200: float) -> np.ndarray:
    """
    NFW enclosed mass profile.

    M(<r) = M200 * f(x) / f(c)
    where x = r/rs, f(x) = ln(1+x) - x/(1+x)
    """
    rs = r200 / c
    x = r / rs

    def f(y):
        return np.log(1 + y) - y / (1 + y)

    return M200 * f(x) / f(c)


def nfw_density(r: np.ndarray, M200: float, c: float, r200: float) -> np.ndarray:
    """NFW density profile in M_sun/Mpc^3."""
    rs = r200 / c
    x = r / rs

    # Characteristic density
    delta_c = (200.0 / 3.0) * c**3 / (np.log(1 + c) - c / (1 + c))
    # Need rho_crit at z, but for simplicity use a fixed value
    # This is for relative calculations, absolute normalization comes from M200
    rho_s = M200 / (4 * np.pi * rs**3 * (np.log(1 + c) - c / (1 + c)))

    return rho_s / (x * (1 + x)**2)


def compute_r500_from_r200(M200: float, c: float, r200: float, rho_crit: float) -> Tuple[float, float]:
    """
    Compute r500 and M500 from M200 and concentration.

    r500 is where mean enclosed density = 500 * rho_crit.
    Uses iterative bisection.
    """
    def mean_density(r):
        M_enc = nfw_mass_enclosed(np.array([r]), M200, c, r200)[0]
        V = (4 / 3) * np.pi * r**3
        return M_enc / V

    # Bisection to find r500
    r_lo, r_hi = 0.1 * r200, r200
    target = 500 * rho_crit

    for _ in range(50):
        r_mid = (r_lo + r_hi) / 2
        if mean_density(r_mid) > target:
            r_lo = r_mid
        else:
            r_hi = r_mid

    r500 = (r_lo + r_hi) / 2
    M500 = nfw_mass_enclosed(np.array([r500]), M200, c, r200)[0]
    return r500, M500


def generate_cluster_population(config: SimConfig,
                                 rng: Optional[np.random.Generator] = None
                                 ) -> list:
    """
    Generate a population of clusters with true physical properties.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration
    rng : np.random.Generator, optional
        Random number generator

    Returns
    -------
    list of ClusterTruth
        Population of clusters
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    N = config.n_clusters
    tc = config.truth
    cosmo = config.cosmology

    # Sample redshifts uniformly
    z_arr = rng.uniform(tc.z_min, tc.z_max, N)

    # Sample masses from truncated normal in log space
    log_mass = rng.normal(tc.log_mass_mean, tc.log_mass_std, N)
    log_mass = np.clip(log_mass, tc.log_mass_min, tc.log_mass_max)
    M200_arr = 10**log_mass

    # Sample concentrations from c(M,z) relation with scatter
    c_mean = tc.c_A * (M200_arr / tc.c_Mpiv)**tc.c_B * (1 + z_arr)**tc.c_C
    log_c_scatter = rng.normal(0, tc.c_scatter, N)
    c200_arr = c_mean * 10**log_c_scatter
    c200_arr = np.clip(c200_arr, 2.0, 15.0)  # physical bounds

    # Compute r200 from M200 and rho_crit(z)
    # M200 = (4/3) * pi * r200^3 * 200 * rho_crit
    rho_crit_arr = np.array([cosmo.rho_crit(z) for z in z_arr])
    r200_arr = (3 * M200_arr / (4 * np.pi * 200 * rho_crit_arr))**(1/3)

    # Projection boost for lensing (lognormal scatter)
    proj_boost_arr = 10**(rng.normal(0, tc.proj_boost_scatter, N))

    # Baryon fractions
    delta_log_mass = log_mass - 14.5
    delta_z = z_arr - 0.3
    f_gas_mean = tc.f_gas_mean + tc.f_gas_mass_slope * delta_log_mass + tc.f_gas_z_slope * delta_z
    f_gas_arr = rng.normal(f_gas_mean, tc.f_gas_scatter)
    f_gas_arr = np.clip(f_gas_arr, 0.02, 0.20)

    f_star_arr = rng.normal(tc.f_star_mean, tc.f_star_scatter, N)
    f_star_arr = np.clip(f_star_arr, 0.005, 0.05)

    # Alpha (universal by default)
    if tc.alpha_z_drift != 0:
        alpha_arr = tc.alpha_true + tc.alpha_z_drift * (z_arr - 0.4)
    else:
        alpha_arr = np.full(N, tc.alpha_true)

    # Build cluster objects
    clusters = []
    for i in range(N):
        z = z_arr[i]
        M200 = M200_arr[i]
        c = c200_arr[i]
        r200 = r200_arr[i]
        rs = r200 / c

        # Compute r500
        rho_crit = rho_crit_arr[i]
        r500, M500 = compute_r500_from_r200(M200, c, r200, rho_crit)

        # Evaluation radii
        r_eval_500 = r500
        r_eval_half = 0.5 * r500

        # Total mass at evaluation radii (NFW)
        M_tot_r500 = nfw_mass_enclosed(np.array([r_eval_500]), M200, c, r200)[0]
        M_tot_half = nfw_mass_enclosed(np.array([r_eval_half]), M200, c, r200)[0]

        # Visible mass (simplified: gas + stars distributed like NFW)
        # M_vis(<r) = (f_gas + f_star) * M_nfw(<r)
        f_vis = f_gas_arr[i] + f_star_arr[i]
        M_vis_r500 = f_vis * M_tot_r500
        M_vis_half = f_vis * M_tot_half

        # Excess mass (dark matter dominated)
        M_exc_r500 = M_tot_r500 - M_vis_r500
        M_exc_half = M_tot_half - M_vis_half

        cluster = ClusterTruth(
            cluster_id=i,
            z=z,
            M200=M200,
            c200=c,
            r200=r200,
            r500=r500,
            rs=rs,
            f_gas=f_gas_arr[i],
            f_star=f_star_arr[i],
            proj_boost=proj_boost_arr[i],
            M_tot_r500=M_tot_r500,
            M_tot_half_r500=M_tot_half,
            M_vis_r500=M_vis_r500,
            M_vis_half_r500=M_vis_half,
            M_exc_r500=M_exc_r500,
            M_exc_half_r500=M_exc_half,
            alpha_true=alpha_arr[i]
        )
        clusters.append(cluster)

    return clusters


def clusters_to_arrays(clusters: list) -> dict:
    """Convert list of ClusterTruth to arrays for vectorized operations."""
    N = len(clusters)
    arrays = {
        'cluster_id': np.array([c.cluster_id for c in clusters]),
        'z': np.array([c.z for c in clusters]),
        'M200': np.array([c.M200 for c in clusters]),
        'c200': np.array([c.c200 for c in clusters]),
        'r200': np.array([c.r200 for c in clusters]),
        'r500': np.array([c.r500 for c in clusters]),
        'rs': np.array([c.rs for c in clusters]),
        'f_gas': np.array([c.f_gas for c in clusters]),
        'f_star': np.array([c.f_star for c in clusters]),
        'proj_boost': np.array([c.proj_boost for c in clusters]),
        'M_tot_r500': np.array([c.M_tot_r500 for c in clusters]),
        'M_tot_half_r500': np.array([c.M_tot_half_r500 for c in clusters]),
        'M_vis_r500': np.array([c.M_vis_r500 for c in clusters]),
        'M_vis_half_r500': np.array([c.M_vis_half_r500 for c in clusters]),
        'M_exc_r500': np.array([c.M_exc_r500 for c in clusters]),
        'M_exc_half_r500': np.array([c.M_exc_half_r500 for c in clusters]),
        'alpha_true': np.array([c.alpha_true for c in clusters]),
    }
    return arrays
