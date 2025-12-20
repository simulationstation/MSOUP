"""
Simulation runner for Msoup Substructure Simulation

Generates synthetic lens samples and extracts summary statistics.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np

from .config import SimConfig, calibrate_models_to_match_mean
from .models import get_model, Lens


@dataclass
class SimulationResult:
    """Results from a single simulation run."""
    model_name: str
    n_lenses: int
    seed: int

    # Per-lens data
    counts: np.ndarray  # N_i for each lens
    H: np.ndarray  # Host mass proxy
    z: np.ndarray  # Redshift proxy
    lambda_i: np.ndarray  # Expected intensity

    # Lists of angular positions (ragged)
    theta_list: List[np.ndarray]

    # Summary statistics (computed during simulation)
    mean_count: float
    var_count: float
    fano_factor: float

    # Sensitivity windows (only for *_window models)
    windows: Optional[List] = None


def run_simulation(config: SimConfig,
                   calibrate: bool = True,
                   rng: Optional[np.random.Generator] = None
                   ) -> SimulationResult:
    """
    Run a single simulation.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration
    calibrate : bool
        If True, calibrate Cox/Cluster to match Poisson mean
    rng : Generator, optional
        Random number generator

    Returns
    -------
    SimulationResult
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    # Calibrate models to have same mean
    if calibrate:
        config = calibrate_models_to_match_mean(config)

    # Get model and generate lenses
    model = get_model(config)
    lenses = model.generate(rng)

    # Extract arrays
    N = len(lenses)
    counts = np.array([lens.n_perturbers for lens in lenses])
    H = np.array([lens.H for lens in lenses])
    z = np.array([lens.z for lens in lenses])
    lambda_i = np.array([lens.lambda_i for lens in lenses])
    theta_list = [lens.theta for lens in lenses]

    # Extract windows if available (for *_window models)
    windows = None
    if hasattr(model, 'get_windows'):
        windows = model.get_windows()

    # Compute summary statistics
    mean_count = np.mean(counts)
    var_count = np.var(counts)
    fano_factor = var_count / mean_count if mean_count > 0 else 0.0

    return SimulationResult(
        model_name=config.model,
        n_lenses=N,
        seed=config.seed,
        counts=counts,
        H=H,
        z=z,
        lambda_i=lambda_i,
        theta_list=theta_list,
        mean_count=mean_count,
        var_count=var_count,
        fano_factor=fano_factor,
        windows=windows
    )


def run_comparison(n_lenses: int = 10000,
                   seed: int = 42,
                   verbose: bool = True) -> Dict[str, SimulationResult]:
    """
    Run all three models and return results for comparison.

    Parameters
    ----------
    n_lenses : int
        Number of lenses per model
    seed : int
        Random seed (same for all models for fair comparison)
    verbose : bool
        Print progress

    Returns
    -------
    Dict mapping model name to SimulationResult
    """
    results = {}

    for model_name in ["poisson", "cox", "cluster"]:
        if verbose:
            print(f"Running {model_name} model with {n_lenses} lenses...")

        config = SimConfig(n_lenses=n_lenses, seed=seed, model=model_name)
        result = run_simulation(config, calibrate=True)
        results[model_name] = result

        if verbose:
            print(f"  Mean count: {result.mean_count:.3f}")
            print(f"  Fano factor: {result.fano_factor:.3f}")

    return results


def run_sweep(n_lenses_list: List[int],
              seed: int = 42,
              verbose: bool = True) -> Dict[str, List[SimulationResult]]:
    """
    Run all models for a range of sample sizes.

    Parameters
    ----------
    n_lenses_list : list of int
        Sample sizes to test
    seed : int
        Base random seed
    verbose : bool
        Print progress

    Returns
    -------
    Dict mapping model name to list of SimulationResults
    """
    results = {model: [] for model in ["poisson", "cox", "cluster"]}

    for i, n_lenses in enumerate(n_lenses_list):
        if verbose:
            print(f"\n=== N_lenses = {n_lenses} ===")

        for model_name in ["poisson", "cox", "cluster"]:
            # Use different seed for each sample size to ensure independence
            run_seed = seed + i * 1000
            config = SimConfig(n_lenses=n_lenses, seed=run_seed, model=model_name)
            result = run_simulation(config, calibrate=True)
            results[model_name].append(result)

            if verbose:
                print(f"  {model_name}: mean={result.mean_count:.2f}, Fano={result.fano_factor:.3f}")

    return results
