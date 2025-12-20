"""
Arc Sensitivity Window Module for Msoup Substructure Simulation

Implements per-lens arc sensitivity functions S_i(θ) that represent
unequal detection probability around the Einstein ring.

This tests whether selection effects alone can create spurious clustering.
"""

from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np


@dataclass
class WindowConfig:
    """Configuration for arc sensitivity windows."""
    # Number of bright arc segments
    kappa_arc: float = 2.0  # Mean if Poisson, or fixed count
    fixed_k_arc: bool = False  # If True, use kappa_arc as fixed count

    # Arc segment width (radians)
    sigma_arc: float = 0.4

    # Amplitude variation
    sigma_A: float = 0.3  # LogNormal sigma for amplitudes

    # Floor sensitivity (never exactly zero)
    epsilon_floor: float = 0.02

    # Detection sharpness
    gamma: float = 1.5  # Higher = sharper detection cutoff

    # Detection mode: "thin" (probabilistic) or "threshold" (strength modulation)
    detection_mode: str = "thin"


@dataclass
class SensitivityWindow:
    """Sensitivity function for a single lens."""
    lens_id: int
    theta_centers: np.ndarray  # Centers of bright segments
    amplitudes: np.ndarray  # Amplitudes of each segment
    sigma_arc: float
    epsilon_floor: float
    normalization: float  # Factor to make mean = 1

    def evaluate(self, theta: np.ndarray) -> np.ndarray:
        """
        Evaluate sensitivity S(θ) at given angles.

        Returns normalized sensitivity (mean over θ ≈ 1).
        """
        if len(theta) == 0:
            return np.array([])

        S = np.full(len(theta), self.epsilon_floor)

        for k, (center, amp) in enumerate(zip(self.theta_centers, self.amplitudes)):
            # Circular distance
            d = circular_distance_vec(theta, center)
            S += amp * np.exp(-d**2 / (2 * self.sigma_arc**2))

        return S / self.normalization

    def evaluate_grid(self, n_grid: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Evaluate on a regular grid for visualization."""
        theta_grid = np.linspace(0, 2 * np.pi, n_grid)
        S_grid = self.evaluate(theta_grid)
        return theta_grid, S_grid


def circular_distance_vec(theta1: np.ndarray, theta2: float) -> np.ndarray:
    """Vectorized circular distance on [0, 2π)."""
    diff = np.abs(theta1 - theta2)
    return np.minimum(diff, 2 * np.pi - diff)


def generate_sensitivity_window(lens_id: int, config: WindowConfig,
                                 rng: np.random.Generator) -> SensitivityWindow:
    """
    Generate a sensitivity window for a single lens.

    S_i(θ) = sum_{k=1..K} A_k * exp[-d_circ(θ, θ_k)^2/(2σ^2)] + ε_floor

    Normalized so mean over θ is 1.
    """
    # Number of bright segments
    if config.fixed_k_arc:
        K = int(config.kappa_arc)
    else:
        K = rng.poisson(config.kappa_arc)
        K = max(K, 1)  # Ensure at least one segment

    # Segment centers (uniform on circle)
    theta_centers = rng.uniform(0, 2 * np.pi, K)

    # Segment amplitudes (log-normal)
    log_A = rng.normal(0, config.sigma_A, K)
    amplitudes = np.exp(log_A)

    # Compute normalization factor (mean S over θ = 1)
    # Evaluate on a fine grid
    theta_grid = np.linspace(0, 2 * np.pi, 500)
    S_unnorm = np.full(len(theta_grid), config.epsilon_floor)
    for center, amp in zip(theta_centers, amplitudes):
        d = circular_distance_vec(theta_grid, center)
        S_unnorm += amp * np.exp(-d**2 / (2 * config.sigma_arc**2))

    normalization = np.mean(S_unnorm)

    return SensitivityWindow(
        lens_id=lens_id,
        theta_centers=theta_centers,
        amplitudes=amplitudes,
        sigma_arc=config.sigma_arc,
        epsilon_floor=config.epsilon_floor,
        normalization=normalization
    )


def apply_window_thinning(theta: np.ndarray, strengths: np.ndarray,
                           window: SensitivityWindow, gamma: float,
                           rng: np.random.Generator
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply probabilistic thinning based on sensitivity window.

    Detection probability: p_det(θ) = min(1, (S(θ)/S_max)^γ)

    Parameters
    ----------
    theta : array
        True perturber positions
    strengths : array
        True perturber strengths
    window : SensitivityWindow
        Lens sensitivity function
    gamma : float
        Detection sharpness (1=mild, 2=strong)
    rng : Generator

    Returns
    -------
    theta_det, strengths_det : arrays
        Detected perturbers after thinning
    """
    if len(theta) == 0:
        return np.array([]), np.array([])

    # Evaluate sensitivity at perturber positions
    S = window.evaluate(theta)
    S_max = np.max(S) if len(S) > 0 else 1.0

    # Detection probability
    p_det = np.minimum(1.0, (S / S_max)**gamma)

    # Accept/reject
    u = rng.uniform(0, 1, len(theta))
    mask = u < p_det

    return theta[mask], strengths[mask]


def apply_window_threshold(theta: np.ndarray, strengths: np.ndarray,
                            window: SensitivityWindow, gamma: float,
                            s_det_base: float
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply threshold modulation based on sensitivity window.

    Effective threshold: s_det(θ) = s_det_base / S(θ)^γ
    Detect if s > s_det(θ)

    Parameters
    ----------
    theta : array
        True perturber positions
    strengths : array
        True perturber strengths
    window : SensitivityWindow
        Lens sensitivity function
    gamma : float
        Threshold modulation power
    s_det_base : float
        Base detection threshold

    Returns
    -------
    theta_det, strengths_det : arrays
        Detected perturbers
    """
    if len(theta) == 0:
        return np.array([]), np.array([])

    # Evaluate sensitivity at perturber positions
    S = window.evaluate(theta)

    # Position-dependent threshold
    s_det_theta = s_det_base / (S**gamma + 1e-10)

    # Detection criterion
    mask = strengths > s_det_theta

    return theta[mask], strengths[mask]


def sample_from_sensitivity(window: SensitivityWindow, n_samples: int,
                             rng: np.random.Generator) -> np.ndarray:
    """
    Sample angular positions from the sensitivity distribution.

    Uses rejection sampling with a uniform proposal.
    """
    if n_samples == 0:
        return np.array([])

    # Evaluate sensitivity on grid to find max
    theta_grid = np.linspace(0, 2 * np.pi, 200)
    S_grid = window.evaluate(theta_grid)
    S_max = np.max(S_grid) * 1.1  # Safety margin

    samples = []
    while len(samples) < n_samples:
        # Propose from uniform
        n_propose = min(n_samples * 3, 10000)
        theta_prop = rng.uniform(0, 2 * np.pi, n_propose)
        S_prop = window.evaluate(theta_prop)

        # Accept with probability S/S_max
        u = rng.uniform(0, 1, n_propose)
        accepted = theta_prop[u < S_prop / S_max]
        samples.extend(accepted.tolist())

    return np.array(samples[:n_samples])


def compute_null_clustering_distribution(window: SensitivityWindow,
                                          n_obs: int,
                                          theta0: float,
                                          n_resamples: int,
                                          rng: np.random.Generator
                                          ) -> Tuple[float, float]:
    """
    Compute expected clustering under null (uniform truth + selection).

    Parameters
    ----------
    window : SensitivityWindow
        Lens sensitivity function
    n_obs : int
        Number of observed perturbers
    theta0 : float
        Clustering threshold
    n_resamples : int
        Number of Monte Carlo samples
    rng : Generator

    Returns
    -------
    mu : float
        Mean null clustering
    sigma : float
        Std of null clustering
    """
    if n_obs < 2:
        return np.nan, np.nan

    from .stats import compute_pairwise_clustering

    C_samples = []
    for _ in range(n_resamples):
        # Sample from sensitivity distribution
        theta_sample = sample_from_sensitivity(window, n_obs, rng)
        C = compute_pairwise_clustering(theta_sample, theta0)
        C_samples.append(C)

    C_samples = np.array(C_samples)
    return np.mean(C_samples), np.std(C_samples)
