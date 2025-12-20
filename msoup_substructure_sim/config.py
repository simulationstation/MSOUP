"""
Configuration for Msoup Substructure Simulation

All parameters for Poisson, Cox, and Cluster models are defined here.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class HostConfig:
    """Configuration for host lens properties."""
    # Host mass proxy H_i ~ Normal(0, 1)
    # Redshift proxy z_i ~ Uniform(z_min, z_max)
    z_min: float = 0.2
    z_max: float = 1.0

    # Coefficients for log-intensity:
    # log λ_i = log λ0 + a_H * H_i + a_z * (z_i - 0.6)
    a_H: float = 0.2  # Host mass dependence
    a_z: float = 0.1  # Redshift dependence


@dataclass
class PoissonConfig:
    """Configuration for Poisson baseline model (CDM-like)."""
    # Baseline intensity (mean perturbers per lens)
    lambda0: float = 3.0


@dataclass
class CoxConfig:
    """Configuration for Cox (doubly-stochastic Poisson) model.

    This creates over-dispersion without spatial clustering.
    The intensity multiplier u_i ~ LogNormal(0, sigma_u).
    """
    # Log-normal sigma for intensity multiplier
    # Higher sigma_u = more over-dispersion
    sigma_u: float = 0.8

    # Base lambda (will be adjusted to match mean)
    lambda0: float = 3.0


@dataclass
class ClusterConfig:
    """Configuration for cluster process model (domain pockets).

    This creates over-dispersion + spatial clustering along the arc.
    """
    # Number of hotspots per lens: K_i ~ Poisson(kappa0)
    kappa0: float = 1.5

    # Offspring per hotspot: m_k ~ Poisson(mu_off)
    mu_off: float = 2.0

    # Offspring angular spread (radians)
    sigma_theta: float = 0.25

    # Background fraction: bkg_rate = epsilon * lambda_expected
    epsilon: float = 0.1

    # Expected total rate (for normalization)
    # Will be calibrated to match Poisson mean
    lambda0: float = 3.0


@dataclass
class DetectionConfig:
    """Configuration for detection threshold model.

    Each perturber gets strength s ~ power law, detected if s > s_det.
    """
    # Power law index for strength distribution: p(s) ~ s^{-beta}
    beta: float = 2.0

    # Strength range
    s_min: float = 0.1
    s_max: float = 10.0

    # Detection threshold
    s_det: float = 0.5

    # Host-dependent detection variation (optional)
    # s_det_i = s_det * (1 + delta_det * H_i)
    delta_det: float = 0.05


@dataclass
class StatsConfig:
    """Configuration for statistical analysis."""
    # Clustering metric: fraction of pairs with angular separation < theta0
    theta0: float = 0.3  # radians

    # Tail probability threshold: P(N >= mean + k * sqrt(mean))
    tail_k: float = 3.0

    # Bootstrap samples for confidence intervals
    n_bootstrap: int = 100

    # Debiasing resamples for null distribution
    n_debias_resamples: int = 200


@dataclass
class WindowConfig:
    """Configuration for arc sensitivity windows.

    Models unequal detection probability around the Einstein ring.
    """
    # Enable window effects
    enabled: bool = False

    # Number of bright arc segments: K ~ Poisson(kappa_arc) or fixed
    kappa_arc: float = 2.0
    fixed_k_arc: bool = False  # If True, use kappa_arc as fixed count

    # Arc segment width (radians)
    sigma_arc: float = 0.4

    # Amplitude variation (log-normal sigma)
    sigma_A: float = 0.3

    # Floor sensitivity (never exactly zero)
    epsilon_floor: float = 0.02

    # Detection sharpness: higher = sharper cutoff
    gamma: float = 1.5

    # Detection mode: "thin" (probabilistic) or "threshold" (strength modulation)
    detection_mode: str = "thin"


@dataclass
class SimConfig:
    """Master simulation configuration."""
    n_lenses: int = 10000
    seed: int = 42
    model: str = "poisson"  # "poisson", "cox", "cluster", or with "_window" suffix

    host: HostConfig = field(default_factory=HostConfig)
    poisson: PoissonConfig = field(default_factory=PoissonConfig)
    cox: CoxConfig = field(default_factory=CoxConfig)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    window: WindowConfig = field(default_factory=WindowConfig)

    def __post_init__(self):
        assert self.n_lenses > 0
        valid_models = ["poisson", "cox", "cluster",
                        "poisson_window", "cox_window", "cluster_window"]
        assert self.model in valid_models, f"Unknown model: {self.model}"

        # Auto-enable windows for *_window models
        if self.model.endswith("_window"):
            self.window.enabled = True


def get_default_config(n_lenses: int = 10000,
                       seed: int = 42,
                       model: str = "poisson") -> SimConfig:
    """Get default simulation configuration."""
    return SimConfig(n_lenses=n_lenses, seed=seed, model=model)


def calibrate_models_to_match_mean(config: SimConfig) -> SimConfig:
    """
    Adjust Cox and Cluster model parameters so they have the same
    expected count as the Poisson baseline.

    This is CRITICAL: we want to test dispersion/clustering differences,
    not mean differences.
    """
    # The Poisson model has E[N] = lambda0 (ignoring host modulation for now)
    target_mean = config.poisson.lambda0

    # Cox model: E[N] = lambda0 * E[u] where u ~ LogNormal(0, sigma)
    # E[u] = exp(sigma^2 / 2)
    # So we need lambda0_cox = target_mean / exp(sigma^2 / 2)
    sigma_u = config.cox.sigma_u
    e_u = np.exp(sigma_u**2 / 2)
    config.cox.lambda0 = target_mean / e_u

    # Cluster model:
    # E[N] = E[K] * E[m] + bkg_rate
    # E[N] = kappa0 * mu_off + epsilon * lambda0
    # Solve for kappa0 given target mean and other params:
    # target = kappa0 * mu_off + epsilon * target
    # kappa0 = (target * (1 - epsilon)) / mu_off
    mu_off = config.cluster.mu_off
    epsilon = config.cluster.epsilon
    config.cluster.kappa0 = target_mean * (1 - epsilon) / mu_off
    config.cluster.lambda0 = target_mean

    return config


# Avoid import at module level; import numpy only when needed
import numpy as np
