"""
Perturber Models for Msoup Substructure Simulation

Three models:
1. Poisson: Independent M2-ish perturbers (CDM-like baseline)
2. Cox: Doubly-stochastic Poisson (over-dispersion, no spatial clustering)
3. Cluster: Domain pockets spawn clusters of perturbers (over-dispersion + clustering)
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np

from .config import SimConfig


@dataclass
class Lens:
    """A single lens with its host properties and detected perturbers."""
    lens_id: int
    H: float  # Host mass proxy
    z: float  # Redshift proxy
    lambda_i: float  # Expected intensity (host-modulated)

    # Perturber data
    n_perturbers: int  # Number of detected perturbers
    theta: np.ndarray  # Angular positions in [0, 2π)
    strengths: np.ndarray  # Perturber strengths (only detected ones)


def compute_intensity(H: np.ndarray, z: np.ndarray,
                      lambda0: float, a_H: float, a_z: float) -> np.ndarray:
    """
    Compute host-modulated intensity for each lens.

    log λ_i = log λ0 + a_H * H_i + a_z * (z_i - 0.6)
    """
    log_lambda = np.log(lambda0) + a_H * H + a_z * (z - 0.6)
    return np.exp(log_lambda)


def draw_strengths(n: int, beta: float, s_min: float, s_max: float,
                   rng: np.random.Generator) -> np.ndarray:
    """
    Draw perturber strengths from a power law distribution.

    p(s) ∝ s^{-beta} for s in [s_min, s_max]
    Uses inverse CDF sampling.
    """
    if n == 0:
        return np.array([])

    if beta == 1:
        # Special case: log-uniform
        log_s = rng.uniform(np.log(s_min), np.log(s_max), n)
        return np.exp(log_s)
    else:
        # General power law
        u = rng.uniform(0, 1, n)
        exp = 1 - beta
        s = ((s_max**exp - s_min**exp) * u + s_min**exp)**(1/exp)
        return s


def apply_detection(theta: np.ndarray, strengths: np.ndarray,
                    s_det: float) -> Tuple[np.ndarray, np.ndarray]:
    """Apply detection threshold: keep only perturbers with s > s_det."""
    mask = strengths > s_det
    return theta[mask], strengths[mask]


class PoissonModel:
    """
    Poisson baseline model (CDM-like).

    Each lens i has N_i ~ Poisson(λ_i) perturbers placed uniformly on the ring.
    This represents independent M2-ish subhalos.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.pc = config.poisson
        self.hc = config.host
        self.dc = config.detection

    def generate(self, rng: np.random.Generator) -> List[Lens]:
        """Generate a sample of lenses with perturbers."""
        N = self.config.n_lenses

        # Generate host properties
        H = rng.normal(0, 1, N)
        z = rng.uniform(self.hc.z_min, self.hc.z_max, N)

        # Compute intensities
        lambda_i = compute_intensity(H, z, self.pc.lambda0,
                                     self.hc.a_H, self.hc.a_z)

        lenses = []
        for i in range(N):
            # Draw number of perturbers (before detection)
            n_raw = rng.poisson(lambda_i[i])

            if n_raw > 0:
                # Positions uniform on the ring
                theta_raw = rng.uniform(0, 2 * np.pi, n_raw)

                # Draw strengths
                strengths_raw = draw_strengths(n_raw, self.dc.beta,
                                               self.dc.s_min, self.dc.s_max, rng)

                # Apply detection threshold (host-dependent)
                s_det_i = self.dc.s_det * (1 + self.dc.delta_det * H[i])
                theta, strengths = apply_detection(theta_raw, strengths_raw, s_det_i)
            else:
                theta = np.array([])
                strengths = np.array([])

            lens = Lens(
                lens_id=i,
                H=H[i],
                z=z[i],
                lambda_i=lambda_i[i],
                n_perturbers=len(theta),
                theta=theta,
                strengths=strengths
            )
            lenses.append(lens)

        return lenses


class CoxModel:
    """
    Cox (doubly-stochastic Poisson) model.

    Each lens has an intensity multiplier u_i ~ LogNormal(0, sigma_u).
    Effective intensity Λ_i = λ_i * u_i.
    Positions are still uniform (no spatial clustering).

    This creates over-dispersion without spatial clustering,
    representing lens-to-lens domain state variation.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.cc = config.cox
        self.hc = config.host
        self.dc = config.detection

    def generate(self, rng: np.random.Generator) -> List[Lens]:
        """Generate a sample of lenses with perturbers."""
        N = self.config.n_lenses

        # Generate host properties
        H = rng.normal(0, 1, N)
        z = rng.uniform(self.hc.z_min, self.hc.z_max, N)

        # Compute base intensities
        lambda_base = compute_intensity(H, z, self.cc.lambda0,
                                        self.hc.a_H, self.hc.a_z)

        # Draw intensity multipliers (log-normal)
        log_u = rng.normal(0, self.cc.sigma_u, N)
        u = np.exp(log_u)

        # Effective intensities
        lambda_i = lambda_base * u

        lenses = []
        for i in range(N):
            # Draw number of perturbers
            n_raw = rng.poisson(lambda_i[i])

            if n_raw > 0:
                # Positions uniform on the ring
                theta_raw = rng.uniform(0, 2 * np.pi, n_raw)

                # Draw strengths
                strengths_raw = draw_strengths(n_raw, self.dc.beta,
                                               self.dc.s_min, self.dc.s_max, rng)

                # Apply detection threshold
                s_det_i = self.dc.s_det * (1 + self.dc.delta_det * H[i])
                theta, strengths = apply_detection(theta_raw, strengths_raw, s_det_i)
            else:
                theta = np.array([])
                strengths = np.array([])

            lens = Lens(
                lens_id=i,
                H=H[i],
                z=z[i],
                lambda_i=lambda_base[i],  # Store base intensity for reference
                n_perturbers=len(theta),
                theta=theta,
                strengths=strengths
            )
            lenses.append(lens)

        return lenses


class ClusterModel:
    """
    Cluster process model (domain pockets).

    Each lens has:
    - K_i ~ Poisson(κ0) hotspots (domain pockets)
    - Each hotspot spawns m_k ~ Poisson(μ_off) offspring
    - Offspring positions: Normal(θ_k, σ_θ) wrapped around circle
    - Plus a background Poisson(ε * λ_i) component

    This creates both over-dispersion AND spatial clustering.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.clc = config.cluster
        self.hc = config.host
        self.dc = config.detection

    def generate(self, rng: np.random.Generator) -> List[Lens]:
        """Generate a sample of lenses with perturbers."""
        N = self.config.n_lenses

        # Generate host properties
        H = rng.normal(0, 1, N)
        z = rng.uniform(self.hc.z_min, self.hc.z_max, N)

        # Compute base intensity (for background)
        lambda_base = compute_intensity(H, z, self.clc.lambda0,
                                        self.hc.a_H, self.hc.a_z)

        lenses = []
        for i in range(N):
            theta_list = []
            strength_list = []

            # Draw number of hotspots
            n_hotspots = rng.poisson(self.clc.kappa0)

            for _ in range(n_hotspots):
                # Hotspot center
                theta_center = rng.uniform(0, 2 * np.pi)

                # Number of offspring
                n_offspring = rng.poisson(self.clc.mu_off)

                if n_offspring > 0:
                    # Offspring positions (wrapped normal)
                    offsets = rng.normal(0, self.clc.sigma_theta, n_offspring)
                    theta_offspring = (theta_center + offsets) % (2 * np.pi)
                    theta_list.append(theta_offspring)

                    # Offspring strengths
                    s_offspring = draw_strengths(n_offspring, self.dc.beta,
                                                 self.dc.s_min, self.dc.s_max, rng)
                    strength_list.append(s_offspring)

            # Background component
            bkg_rate = self.clc.epsilon * lambda_base[i]
            n_bkg = rng.poisson(bkg_rate)

            if n_bkg > 0:
                theta_bkg = rng.uniform(0, 2 * np.pi, n_bkg)
                s_bkg = draw_strengths(n_bkg, self.dc.beta,
                                       self.dc.s_min, self.dc.s_max, rng)
                theta_list.append(theta_bkg)
                strength_list.append(s_bkg)

            # Concatenate all perturbers
            if theta_list:
                theta_raw = np.concatenate(theta_list)
                strengths_raw = np.concatenate(strength_list)
            else:
                theta_raw = np.array([])
                strengths_raw = np.array([])

            # Apply detection threshold
            if len(theta_raw) > 0:
                s_det_i = self.dc.s_det * (1 + self.dc.delta_det * H[i])
                theta, strengths = apply_detection(theta_raw, strengths_raw, s_det_i)
            else:
                theta = np.array([])
                strengths = np.array([])

            lens = Lens(
                lens_id=i,
                H=H[i],
                z=z[i],
                lambda_i=lambda_base[i],
                n_perturbers=len(theta),
                theta=theta,
                strengths=strengths
            )
            lenses.append(lens)

        return lenses


class PoissonWindowModel:
    """
    Poisson model with arc sensitivity window.

    Tests whether unequal arc sensitivity can create spurious clustering
    even when true perturbers are uniformly distributed.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.pc = config.poisson
        self.hc = config.host
        self.dc = config.detection
        self.wc = config.window

    def generate(self, rng: np.random.Generator) -> List[Lens]:
        """Generate lenses with window-modulated detection."""
        from .windows import (generate_sensitivity_window, apply_window_thinning,
                              apply_window_threshold, WindowConfig)

        N = self.config.n_lenses

        # Generate host properties
        H = rng.normal(0, 1, N)
        z = rng.uniform(self.hc.z_min, self.hc.z_max, N)

        # Compute intensities
        lambda_i = compute_intensity(H, z, self.pc.lambda0,
                                     self.hc.a_H, self.hc.a_z)

        # Create window config from SimConfig
        win_cfg = WindowConfig(
            kappa_arc=self.wc.kappa_arc,
            fixed_k_arc=self.wc.fixed_k_arc,
            sigma_arc=self.wc.sigma_arc,
            sigma_A=self.wc.sigma_A,
            epsilon_floor=self.wc.epsilon_floor,
            gamma=self.wc.gamma,
            detection_mode=self.wc.detection_mode
        )

        lenses = []
        windows = []

        for i in range(N):
            # Generate sensitivity window for this lens
            window = generate_sensitivity_window(i, win_cfg, rng)
            windows.append(window)

            # Draw true perturbers (uniform on ring)
            n_raw = rng.poisson(lambda_i[i])

            if n_raw > 0:
                theta_raw = rng.uniform(0, 2 * np.pi, n_raw)
                strengths_raw = draw_strengths(n_raw, self.dc.beta,
                                               self.dc.s_min, self.dc.s_max, rng)

                # Apply standard detection first
                s_det_i = self.dc.s_det * (1 + self.dc.delta_det * H[i])
                theta_det, strengths_det = apply_detection(theta_raw, strengths_raw, s_det_i)

                # Then apply window effects
                if len(theta_det) > 0:
                    if self.wc.detection_mode == "thin":
                        theta, strengths = apply_window_thinning(
                            theta_det, strengths_det, window, self.wc.gamma, rng)
                    else:
                        theta, strengths = apply_window_threshold(
                            theta_det, strengths_det, window, self.wc.gamma, s_det_i)
                else:
                    theta = np.array([])
                    strengths = np.array([])
            else:
                theta = np.array([])
                strengths = np.array([])

            lens = Lens(
                lens_id=i,
                H=H[i],
                z=z[i],
                lambda_i=lambda_i[i],
                n_perturbers=len(theta),
                theta=theta,
                strengths=strengths
            )
            lenses.append(lens)

        # Store windows for debiasing
        self._windows = windows

        return lenses

    def get_windows(self):
        """Return stored sensitivity windows."""
        return getattr(self, '_windows', None)


class CoxWindowModel:
    """
    Cox model with arc sensitivity window.

    Tests whether over-dispersion + windows can mimic pocket clustering.
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.cc = config.cox
        self.hc = config.host
        self.dc = config.detection
        self.wc = config.window

    def generate(self, rng: np.random.Generator) -> List[Lens]:
        """Generate lenses with Cox dispersion and window detection."""
        from .windows import (generate_sensitivity_window, apply_window_thinning,
                              apply_window_threshold, WindowConfig)

        N = self.config.n_lenses

        H = rng.normal(0, 1, N)
        z = rng.uniform(self.hc.z_min, self.hc.z_max, N)

        lambda_base = compute_intensity(H, z, self.cc.lambda0,
                                        self.hc.a_H, self.hc.a_z)

        log_u = rng.normal(0, self.cc.sigma_u, N)
        u = np.exp(log_u)
        lambda_i = lambda_base * u

        win_cfg = WindowConfig(
            kappa_arc=self.wc.kappa_arc,
            fixed_k_arc=self.wc.fixed_k_arc,
            sigma_arc=self.wc.sigma_arc,
            sigma_A=self.wc.sigma_A,
            epsilon_floor=self.wc.epsilon_floor,
            gamma=self.wc.gamma,
            detection_mode=self.wc.detection_mode
        )

        lenses = []
        windows = []

        for i in range(N):
            window = generate_sensitivity_window(i, win_cfg, rng)
            windows.append(window)

            n_raw = rng.poisson(lambda_i[i])

            if n_raw > 0:
                theta_raw = rng.uniform(0, 2 * np.pi, n_raw)
                strengths_raw = draw_strengths(n_raw, self.dc.beta,
                                               self.dc.s_min, self.dc.s_max, rng)

                s_det_i = self.dc.s_det * (1 + self.dc.delta_det * H[i])
                theta_det, strengths_det = apply_detection(theta_raw, strengths_raw, s_det_i)

                if len(theta_det) > 0:
                    if self.wc.detection_mode == "thin":
                        theta, strengths = apply_window_thinning(
                            theta_det, strengths_det, window, self.wc.gamma, rng)
                    else:
                        theta, strengths = apply_window_threshold(
                            theta_det, strengths_det, window, self.wc.gamma, s_det_i)
                else:
                    theta = np.array([])
                    strengths = np.array([])
            else:
                theta = np.array([])
                strengths = np.array([])

            lens = Lens(
                lens_id=i,
                H=H[i],
                z=z[i],
                lambda_i=lambda_base[i],
                n_perturbers=len(theta),
                theta=theta,
                strengths=strengths
            )
            lenses.append(lens)

        self._windows = windows
        return lenses

    def get_windows(self):
        return getattr(self, '_windows', None)


class ClusterWindowModel:
    """
    Cluster model with arc sensitivity window.

    Stress test: do real pockets remain detectable under selection?
    """

    def __init__(self, config: SimConfig):
        self.config = config
        self.clc = config.cluster
        self.hc = config.host
        self.dc = config.detection
        self.wc = config.window

    def generate(self, rng: np.random.Generator) -> List[Lens]:
        """Generate lenses with cluster structure and window detection."""
        from .windows import (generate_sensitivity_window, apply_window_thinning,
                              apply_window_threshold, WindowConfig)

        N = self.config.n_lenses

        H = rng.normal(0, 1, N)
        z = rng.uniform(self.hc.z_min, self.hc.z_max, N)

        lambda_base = compute_intensity(H, z, self.clc.lambda0,
                                        self.hc.a_H, self.hc.a_z)

        win_cfg = WindowConfig(
            kappa_arc=self.wc.kappa_arc,
            fixed_k_arc=self.wc.fixed_k_arc,
            sigma_arc=self.wc.sigma_arc,
            sigma_A=self.wc.sigma_A,
            epsilon_floor=self.wc.epsilon_floor,
            gamma=self.wc.gamma,
            detection_mode=self.wc.detection_mode
        )

        lenses = []
        windows = []

        for i in range(N):
            window = generate_sensitivity_window(i, win_cfg, rng)
            windows.append(window)

            theta_list = []
            strength_list = []

            n_hotspots = rng.poisson(self.clc.kappa0)

            for _ in range(n_hotspots):
                theta_center = rng.uniform(0, 2 * np.pi)
                n_offspring = rng.poisson(self.clc.mu_off)

                if n_offspring > 0:
                    offsets = rng.normal(0, self.clc.sigma_theta, n_offspring)
                    theta_offspring = (theta_center + offsets) % (2 * np.pi)
                    theta_list.append(theta_offspring)

                    s_offspring = draw_strengths(n_offspring, self.dc.beta,
                                                 self.dc.s_min, self.dc.s_max, rng)
                    strength_list.append(s_offspring)

            bkg_rate = self.clc.epsilon * lambda_base[i]
            n_bkg = rng.poisson(bkg_rate)

            if n_bkg > 0:
                theta_bkg = rng.uniform(0, 2 * np.pi, n_bkg)
                s_bkg = draw_strengths(n_bkg, self.dc.beta,
                                       self.dc.s_min, self.dc.s_max, rng)
                theta_list.append(theta_bkg)
                strength_list.append(s_bkg)

            if theta_list:
                theta_raw = np.concatenate(theta_list)
                strengths_raw = np.concatenate(strength_list)
            else:
                theta_raw = np.array([])
                strengths_raw = np.array([])

            if len(theta_raw) > 0:
                s_det_i = self.dc.s_det * (1 + self.dc.delta_det * H[i])
                theta_det, strengths_det = apply_detection(theta_raw, strengths_raw, s_det_i)

                if len(theta_det) > 0:
                    if self.wc.detection_mode == "thin":
                        theta, strengths = apply_window_thinning(
                            theta_det, strengths_det, window, self.wc.gamma, rng)
                    else:
                        theta, strengths = apply_window_threshold(
                            theta_det, strengths_det, window, self.wc.gamma, s_det_i)
                else:
                    theta = np.array([])
                    strengths = np.array([])
            else:
                theta = np.array([])
                strengths = np.array([])

            lens = Lens(
                lens_id=i,
                H=H[i],
                z=z[i],
                lambda_i=lambda_base[i],
                n_perturbers=len(theta),
                theta=theta,
                strengths=strengths
            )
            lenses.append(lens)

        self._windows = windows
        return lenses

    def get_windows(self):
        return getattr(self, '_windows', None)


def get_model(config: SimConfig):
    """Get the appropriate model based on config."""
    if config.model == "poisson":
        return PoissonModel(config)
    elif config.model == "cox":
        return CoxModel(config)
    elif config.model == "cluster":
        return ClusterModel(config)
    elif config.model == "poisson_window":
        return PoissonWindowModel(config)
    elif config.model == "cox_window":
        return CoxWindowModel(config)
    elif config.model == "cluster_window":
        return ClusterWindowModel(config)
    else:
        raise ValueError(f"Unknown model: {config.model}")
