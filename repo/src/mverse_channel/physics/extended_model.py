"""Extended model with hidden-sector correlation channel."""

from __future__ import annotations

import numpy as np

from mverse_channel.config import SimulationConfig
from mverse_channel.physics import modulation as modulation_proxy
from mverse_channel.physics import topology as topology_proxy
from mverse_channel.physics.baseline_model import simulate_baseline
from mverse_channel.physics.noise import ou_process


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def _gated_factors(config: SimulationConfig) -> tuple[float, float, float, float]:
    coherence = modulation_proxy.coherence_proxy(
        config.omega_a,
        config.kappa_a,
        mode=config.coherence_mode,
        q_ref=config.q_ref,
        kappa_ref=config.kappa_ref,
    )
    topology = topology_proxy.PRESETS.get(config.topology.preset)
    if topology:
        topo_value = topology_proxy.topology_index(
            topology.n_nodes,
            topology.n_edges,
            topology.cycle_count,
            topology.max_cycles,
        )
    else:
        topo_value = topology_proxy.topology_index(
            config.topology.n_nodes,
            config.topology.n_edges,
            config.topology.cycle_count,
            config.topology.max_cycles,
        )

    boundary = config.modulation.boundary_index(config.omega_a)

    sig_c = _sigmoid((coherence - config.hidden_channel.coherence_threshold) / config.hidden_channel.threshold_width)
    sig_t = _sigmoid((topo_value - config.hidden_channel.topology_threshold) / config.hidden_channel.threshold_width)
    sig_b = _sigmoid((boundary - config.hidden_channel.boundary_threshold) / config.hidden_channel.threshold_width)

    epsilon_eff = (
        config.hidden_channel.epsilon0
        + config.hidden_channel.epsilon_max * sig_c * sig_t * sig_b
    )
    return epsilon_eff, coherence, topo_value, boundary


def simulate_extended(config: SimulationConfig, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Simulate extended model with hidden correlation channel."""
    base = simulate_baseline(config)
    if not config.hidden_channel.enabled:
        return base

    epsilon_eff, coherence, topo_value, boundary = _gated_factors(config)
    n_steps = base["ia"].shape[0]

    x_common = ou_process(n_steps, config.dt, config.hidden_channel.tau_x, rng)
    x_ind = ou_process(n_steps, config.dt, config.hidden_channel.tau_x, rng)

    rho = config.hidden_channel.rho
    x_b = rho * x_common + np.sqrt(max(0.0, 1 - rho**2)) * x_ind

    if config.hidden_channel.phenomenology == "parametric":
        phase = epsilon_eff * x_common
        base["ia"] = base["ia"] * np.cos(phase)
        base["qa"] = base["qa"] * np.sin(phase) + base["qa"]
        base["ib"] = base["ib"] * np.cos(phase)
        base["qb"] = base["qb"] * np.sin(phase) + base["qb"]
    else:
        base["ia"] = base["ia"] + epsilon_eff * x_common
        base["qa"] = base["qa"] + epsilon_eff * x_common
        base["ib"] = base["ib"] + epsilon_eff * x_b
        base["qb"] = base["qb"] + epsilon_eff * x_b

    base["hidden_epsilon"] = np.array([epsilon_eff])
    base["hidden_coherence"] = np.array([coherence])
    base["hidden_topology"] = np.array([topo_value])
    base["hidden_boundary"] = np.array([boundary])
    return base
