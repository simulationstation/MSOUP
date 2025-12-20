"""
Inference Pipelines for Mverse Shadow Simulation

Implements what an M2 analyst would do to infer masses from observables.
The analyst does NOT know the true biases (projection, hydrostatic, anisotropy).
"""

from dataclasses import dataclass
from typing import Tuple, Optional, Dict
import numpy as np
from scipy import optimize

from .config import SimConfig
from .truth import ClusterTruth, nfw_mass_enclosed
from .observables import (LensingObservables, DynamicsObservables,
                          nfw_shear_profile, compute_sigma_crit)


@dataclass
class InferredMasses:
    """Inferred masses for a single cluster."""
    cluster_id: int

    # Lensing inferred
    M200_lens: float  # Fitted M200 from lensing
    c_lens: float  # Fitted concentration
    M_tot_lens_r500: float  # Inferred total mass at r500
    M_tot_lens_half_r500: float  # at 0.5*r500

    # X-ray inferred (analyst assumes zero bias)
    M_tot_dynX_r500: float
    M_tot_dynX_half_r500: float

    # Velocity dispersion inferred (analyst assumes isotropy)
    M200_dynV: float  # Inferred M200 from sigma_v
    M_tot_dynV_r500: float
    M_tot_dynV_half_r500: float

    # Visible mass inferred
    M_vis_inferred_r500: float
    M_vis_inferred_half_r500: float

    # Excess masses
    M_exc_lens_r500: float
    M_exc_lens_half_r500: float
    M_exc_dynX_r500: float
    M_exc_dynX_half_r500: float
    M_exc_dynV_r500: float
    M_exc_dynV_half_r500: float

    # Ratios R = M_exc_lens / M_exc_dyn
    R_X_r500: float
    R_X_half_r500: float
    R_V_r500: float
    R_V_half_r500: float


def fit_nfw_to_shear(r_bins: np.ndarray, g_t: np.ndarray, g_t_err: np.ndarray,
                     z: float, config: SimConfig) -> Tuple[float, float]:
    """
    Fit NFW (M200, c) to weak lensing shear profile.

    Parameters
    ----------
    r_bins : array
        Radial bins in Mpc
    g_t : array
        Observed tangential shear
    g_t_err : array
        Shear errors
    z : float
        Cluster redshift
    config : SimConfig
        Configuration

    Returns
    -------
    M200_fit : float
        Best-fit M200
    c_fit : float
        Best-fit concentration
    """
    ic = config.inference
    cosmo = config.cosmology

    Sigma_crit = compute_sigma_crit(z)
    rho_crit = cosmo.rho_crit(z)

    def model(params):
        log_M200, log_c = params
        M200 = 10**log_M200
        c = 10**log_c

        # Compute r200
        r200 = (3 * M200 / (4 * np.pi * 200 * rho_crit))**(1/3)

        # Model shear
        g_model = nfw_shear_profile(r_bins, M200, c, r200, z, Sigma_crit)

        return g_model

    def chi2(params):
        g_model = model(params)
        chi2_val = np.sum(((g_t - g_model) / g_t_err)**2)

        # Add concentration prior if enabled
        if ic.use_concentration_prior:
            log_c = params[1]
            log_M200 = params[0]
            M200 = 10**log_M200

            # Expected concentration from c(M,z) relation
            tc = config.truth
            c_expected = tc.c_A * (M200 / tc.c_Mpiv)**tc.c_B * (1 + z)**tc.c_C
            log_c_expected = np.log10(c_expected)

            prior_penalty = ((log_c - log_c_expected) / ic.c_prior_scatter)**2
            chi2_val += prior_penalty

        return chi2_val

    # Initial guess
    M200_init = 3e14
    c_init = 4.0

    try:
        result = optimize.minimize(chi2, [np.log10(M200_init), np.log10(c_init)],
                                   bounds=[(13.0, 16.0), (0.3, 1.2)],
                                   method='L-BFGS-B')
        M200_fit = 10**result.x[0]
        c_fit = 10**result.x[1]
    except Exception:
        # Fallback to initial guess scaled by mean shear
        mean_g = np.mean(g_t)
        M200_fit = M200_init * (mean_g / 0.05)
        c_fit = c_init

    # Bounds check
    M200_fit = np.clip(M200_fit, 1e13, 1e16)
    c_fit = np.clip(c_fit, 2.0, 12.0)

    return M200_fit, c_fit


def infer_mass_from_velocity(sigma_v: float, z: float,
                              config: SimConfig) -> float:
    """
    Infer M200 from velocity dispersion assuming isotropy.

    The analyst uses the same scaling relation but ignores anisotropy bias.
    """
    dc = config.dynamics
    cosmo = config.cosmology

    # sigma_v = A * (M200/1e15)^(1/3) * E(z)^(1/3)
    # => M200 = 1e15 * (sigma_v / (A * E(z)^(1/3)))^3

    E_z = cosmo.E(z)
    M200 = 1e15 * (sigma_v / (dc.sigma_v_A * E_z**dc.sigma_v_exp))**3

    return np.clip(M200, 1e13, 1e16)


def infer_visible_mass(cluster: ClusterTruth, config: SimConfig,
                        rng: np.random.Generator) -> Tuple[float, float]:
    """
    Analyst's inference of visible mass with systematic uncertainty.

    Returns M_vis at r500 and 0.5*r500.
    """
    ic = config.inference

    # True visible fraction
    f_vis_true = cluster.f_gas + cluster.f_star

    # Analyst's inferred gas fraction (biased and noisy)
    f_gas_inferred = (cluster.f_gas * (1 + ic.f_gas_inferred_bias) *
                      (1 + rng.normal(0, ic.f_gas_inferred_scatter)))
    f_gas_inferred = np.clip(f_gas_inferred, 0.02, 0.20)

    # Analyst's inferred stellar fraction (noisy)
    f_star_inferred = cluster.f_star * (1 + rng.normal(0, ic.f_star_inferred_scatter))
    f_star_inferred = np.clip(f_star_inferred, 0.005, 0.05)

    f_vis_inferred = f_gas_inferred + f_star_inferred

    # M_vis = f_vis * M_tot (analyst uses their M_tot estimate, but for
    # simplicity we use the true M_tot scaled by inferred fraction)
    M_vis_r500 = f_vis_inferred * cluster.M_tot_r500
    M_vis_half = f_vis_inferred * cluster.M_tot_half_r500

    return M_vis_r500, M_vis_half


def run_inference_single(cluster: ClusterTruth,
                          lensing_obs: LensingObservables,
                          dynamics_obs: DynamicsObservables,
                          config: SimConfig,
                          rng: np.random.Generator) -> InferredMasses:
    """
    Run full inference pipeline for a single cluster.

    This is what an M2 analyst would do.
    """
    cosmo = config.cosmology

    # === Lensing inference ===
    M200_lens, c_lens = fit_nfw_to_shear(
        lensing_obs.r_bins, lensing_obs.g_t, lensing_obs.g_t_err,
        cluster.z, config
    )

    # Compute r200_lens and inferred masses
    rho_crit = cosmo.rho_crit(cluster.z)
    r200_lens = (3 * M200_lens / (4 * np.pi * 200 * rho_crit))**(1/3)

    # Inferred M_tot at evaluation radii
    # Use analyst's r500 estimate (approximate from their M200/c fit)
    from .truth import compute_r500_from_r200
    r500_lens, _ = compute_r500_from_r200(M200_lens, c_lens, r200_lens, rho_crit)

    M_tot_lens_r500 = nfw_mass_enclosed(np.array([r500_lens]), M200_lens, c_lens, r200_lens)[0]
    M_tot_lens_half = nfw_mass_enclosed(np.array([0.5 * r500_lens]), M200_lens, c_lens, r200_lens)[0]

    # === X-ray inference ===
    # Analyst assumes zero hydrostatic bias, so their inferred mass = observed M_X
    M_tot_dynX_r500 = dynamics_obs.M_X_r500
    M_tot_dynX_half = dynamics_obs.M_X_half_r500

    # === Velocity dispersion inference ===
    M200_dynV = infer_mass_from_velocity(dynamics_obs.sigma_v, cluster.z, config)

    # Compute M_tot at evaluation radii using analyst's M200 and assumed c
    tc = config.truth
    c_expected = tc.c_A * (M200_dynV / tc.c_Mpiv)**tc.c_B * (1 + cluster.z)**tc.c_C
    r200_dynV = (3 * M200_dynV / (4 * np.pi * 200 * rho_crit))**(1/3)
    r500_dynV, _ = compute_r500_from_r200(M200_dynV, c_expected, r200_dynV, rho_crit)

    M_tot_dynV_r500 = nfw_mass_enclosed(np.array([r500_dynV]), M200_dynV, c_expected, r200_dynV)[0]
    M_tot_dynV_half = nfw_mass_enclosed(np.array([0.5 * r500_dynV]), M200_dynV, c_expected, r200_dynV)[0]

    # === Visible mass inference ===
    M_vis_r500, M_vis_half = infer_visible_mass(cluster, config, rng)

    # === Compute excess masses ===
    M_exc_lens_r500 = M_tot_lens_r500 - M_vis_r500
    M_exc_lens_half = M_tot_lens_half - M_vis_half
    M_exc_dynX_r500 = M_tot_dynX_r500 - M_vis_r500
    M_exc_dynX_half = M_tot_dynX_half - M_vis_half
    M_exc_dynV_r500 = M_tot_dynV_r500 - M_vis_r500
    M_exc_dynV_half = M_tot_dynV_half - M_vis_half

    # Ensure positive (can be negative due to noise in low-mass systems)
    eps = 1e10  # Small floor
    M_exc_lens_r500 = max(M_exc_lens_r500, eps)
    M_exc_lens_half = max(M_exc_lens_half, eps)
    M_exc_dynX_r500 = max(M_exc_dynX_r500, eps)
    M_exc_dynX_half = max(M_exc_dynX_half, eps)
    M_exc_dynV_r500 = max(M_exc_dynV_r500, eps)
    M_exc_dynV_half = max(M_exc_dynV_half, eps)

    # === Compute ratios ===
    R_X_r500 = M_exc_lens_r500 / M_exc_dynX_r500
    R_X_half = M_exc_lens_half / M_exc_dynX_half
    R_V_r500 = M_exc_lens_r500 / M_exc_dynV_r500
    R_V_half = M_exc_lens_half / M_exc_dynV_half

    return InferredMasses(
        cluster_id=cluster.cluster_id,
        M200_lens=M200_lens,
        c_lens=c_lens,
        M_tot_lens_r500=M_tot_lens_r500,
        M_tot_lens_half_r500=M_tot_lens_half,
        M_tot_dynX_r500=M_tot_dynX_r500,
        M_tot_dynX_half_r500=M_tot_dynX_half,
        M200_dynV=M200_dynV,
        M_tot_dynV_r500=M_tot_dynV_r500,
        M_tot_dynV_half_r500=M_tot_dynV_half,
        M_vis_inferred_r500=M_vis_r500,
        M_vis_inferred_half_r500=M_vis_half,
        M_exc_lens_r500=M_exc_lens_r500,
        M_exc_lens_half_r500=M_exc_lens_half,
        M_exc_dynX_r500=M_exc_dynX_r500,
        M_exc_dynX_half_r500=M_exc_dynX_half,
        M_exc_dynV_r500=M_exc_dynV_r500,
        M_exc_dynV_half_r500=M_exc_dynV_half,
        R_X_r500=R_X_r500,
        R_X_half_r500=R_X_half,
        R_V_r500=R_V_r500,
        R_V_half_r500=R_V_half
    )


def run_inference_vectorized(clusters: list,
                              lensing: Dict,
                              dynamics: Dict,
                              config: SimConfig,
                              rng: Optional[np.random.Generator] = None
                              ) -> Dict:
    """
    Vectorized inference for fast mode.

    Uses simplified mass proxies instead of per-object fitting.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed + 2000)

    N = len(clusters)
    cosmo = config.cosmology
    ic = config.inference
    dc = config.dynamics

    # Extract truth arrays
    z_arr = np.array([c.z for c in clusters])
    M200_arr = np.array([c.M200 for c in clusters])
    c200_arr = np.array([c.c200 for c in clusters])
    r200_arr = np.array([c.r200 for c in clusters])
    r500_arr = np.array([c.r500 for c in clusters])
    f_gas_arr = np.array([c.f_gas for c in clusters])
    f_star_arr = np.array([c.f_star for c in clusters])
    M_tot_r500_arr = np.array([c.M_tot_r500 for c in clusters])
    M_tot_half_arr = np.array([c.M_tot_half_r500 for c in clusters])

    # === Lensing inference (fast: use observed lensing mass directly) ===
    # In fast mode, lensing mass is already computed with projection/calibration biases
    M_lens_r500 = lensing['M_lens_r500']
    M_lens_half = lensing['M_lens_half_r500']

    # === X-ray inference ===
    # Analyst takes X-ray mass at face value (ignores hydrostatic bias)
    M_dynX_r500 = dynamics['M_X_r500']
    M_dynX_half = dynamics['M_X_half_r500']

    # === Velocity inference ===
    sigma_v = dynamics['sigma_v']
    E_z = np.array([cosmo.E(z) for z in z_arr])
    M200_dynV = 1e15 * (sigma_v / (dc.sigma_v_A * E_z**dc.sigma_v_exp))**3
    M200_dynV = np.clip(M200_dynV, 1e13, 1e16)

    # Approximate M_tot at r500 from M200_dynV using mass ratio
    # M(<r500) / M200 ~ 0.65-0.75 for typical NFW
    mass_ratio_r500 = 0.70
    mass_ratio_half = 0.35
    M_dynV_r500 = M200_dynV * mass_ratio_r500
    M_dynV_half = M200_dynV * mass_ratio_half

    # === Visible mass inference ===
    f_gas_inferred = (f_gas_arr * (1 + ic.f_gas_inferred_bias) *
                      (1 + rng.normal(0, ic.f_gas_inferred_scatter, N)))
    f_gas_inferred = np.clip(f_gas_inferred, 0.02, 0.20)

    f_star_inferred = f_star_arr * (1 + rng.normal(0, ic.f_star_inferred_scatter, N))
    f_star_inferred = np.clip(f_star_inferred, 0.005, 0.05)

    f_vis_inferred = f_gas_inferred + f_star_inferred
    M_vis_r500 = f_vis_inferred * M_tot_r500_arr
    M_vis_half = f_vis_inferred * M_tot_half_arr

    # === Excess masses ===
    eps = 1e10
    M_exc_lens_r500 = np.maximum(M_lens_r500 - M_vis_r500, eps)
    M_exc_lens_half = np.maximum(M_lens_half - M_vis_half, eps)
    M_exc_dynX_r500 = np.maximum(M_dynX_r500 - M_vis_r500, eps)
    M_exc_dynX_half = np.maximum(M_dynX_half - M_vis_half, eps)
    M_exc_dynV_r500 = np.maximum(M_dynV_r500 - M_vis_r500, eps)
    M_exc_dynV_half = np.maximum(M_dynV_half - M_vis_half, eps)

    # === Ratios ===
    R_X_r500 = M_exc_lens_r500 / M_exc_dynX_r500
    R_X_half = M_exc_lens_half / M_exc_dynX_half
    R_V_r500 = M_exc_lens_r500 / M_exc_dynV_r500
    R_V_half = M_exc_lens_half / M_exc_dynV_half

    return {
        'z': z_arr,
        'M200_true': M200_arr,
        'log_M200_true': np.log10(M200_arr),
        'M_lens_r500': M_lens_r500,
        'M_lens_half_r500': M_lens_half,
        'M_dynX_r500': M_dynX_r500,
        'M_dynX_half_r500': M_dynX_half,
        'M_dynV_r500': M_dynV_r500,
        'M_dynV_half_r500': M_dynV_half,
        'M_vis_r500': M_vis_r500,
        'M_vis_half_r500': M_vis_half,
        'M_exc_lens_r500': M_exc_lens_r500,
        'M_exc_lens_half_r500': M_exc_lens_half,
        'M_exc_dynX_r500': M_exc_dynX_r500,
        'M_exc_dynX_half_r500': M_exc_dynX_half,
        'M_exc_dynV_r500': M_exc_dynV_r500,
        'M_exc_dynV_half_r500': M_exc_dynV_half,
        'R_X_r500': R_X_r500,
        'R_X_half_r500': R_X_half,
        'R_V_r500': R_V_r500,
        'R_V_half_r500': R_V_half,
        'b_hse_true': dynamics['b_hse_true'],
        'b_aniso_true': dynamics['b_aniso_true'],
        'proj_boost': lensing['proj_boost'],
    }


def run_inference_full(clusters: list,
                        lensing_obs: list,
                        dynamics_obs: list,
                        config: SimConfig,
                        rng: Optional[np.random.Generator] = None
                        ) -> Dict:
    """
    Full inference mode with per-object NFW fitting.

    Slower but more accurate simulation of analyst workflow.
    """
    if rng is None:
        rng = np.random.default_rng(config.seed + 2000)

    results = []
    for cluster, lens, dyn in zip(clusters, lensing_obs, dynamics_obs):
        inferred = run_inference_single(cluster, lens, dyn, config, rng)
        results.append(inferred)

    # Convert to arrays
    N = len(results)
    output = {
        'z': np.array([clusters[i].z for i in range(N)]),
        'M200_true': np.array([clusters[i].M200 for i in range(N)]),
        'log_M200_true': np.log10(np.array([clusters[i].M200 for i in range(N)])),
        'M_lens_r500': np.array([r.M_tot_lens_r500 for r in results]),
        'M_lens_half_r500': np.array([r.M_tot_lens_half_r500 for r in results]),
        'M_dynX_r500': np.array([r.M_tot_dynX_r500 for r in results]),
        'M_dynX_half_r500': np.array([r.M_tot_dynX_half_r500 for r in results]),
        'M_dynV_r500': np.array([r.M_tot_dynV_r500 for r in results]),
        'M_dynV_half_r500': np.array([r.M_tot_dynV_half_r500 for r in results]),
        'M_vis_r500': np.array([r.M_vis_inferred_r500 for r in results]),
        'M_vis_half_r500': np.array([r.M_vis_inferred_half_r500 for r in results]),
        'M_exc_lens_r500': np.array([r.M_exc_lens_r500 for r in results]),
        'M_exc_lens_half_r500': np.array([r.M_exc_lens_half_r500 for r in results]),
        'M_exc_dynX_r500': np.array([r.M_exc_dynX_r500 for r in results]),
        'M_exc_dynX_half_r500': np.array([r.M_exc_dynX_half_r500 for r in results]),
        'M_exc_dynV_r500': np.array([r.M_exc_dynV_r500 for r in results]),
        'M_exc_dynV_half_r500': np.array([r.M_exc_dynV_half_r500 for r in results]),
        'R_X_r500': np.array([r.R_X_r500 for r in results]),
        'R_X_half_r500': np.array([r.R_X_half_r500 for r in results]),
        'R_V_r500': np.array([r.R_V_r500 for r in results]),
        'R_V_half_r500': np.array([r.R_V_half_r500 for r in results]),
        'b_hse_true': np.array([dynamics_obs[i].b_hse_true for i in range(N)]),
        'b_aniso_true': np.array([dynamics_obs[i].b_aniso_true for i in range(N)]),
        'proj_boost': np.array([clusters[i].proj_boost for i in range(N)]),
    }

    return output
