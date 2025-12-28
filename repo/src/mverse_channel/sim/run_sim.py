"""Simulation entrypoints for protocol runs."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np

from mverse_channel.config import ProtocolConfig, SimulationConfig
from mverse_channel.physics.topology import PRESETS
from mverse_channel.rng import get_rng
from mverse_channel.sim.generate_data import compute_metrics, generate_and_save, generate_time_series, save_run


def _apply_protocol(config: SimulationConfig, protocol: ProtocolConfig, mode: str) -> SimulationConfig:
    if mode == "treatment":
        config.kappa_a = (1 - protocol.coherence_high) * config.kappa_ref
        config.topology.preset = protocol.topology_high
        config.modulation.delta_omega = protocol.boundary_high * config.omega_a
    elif mode == "coherence_low":
        config.kappa_a = (1 - protocol.coherence_low) * config.kappa_ref
    elif mode == "topology_low":
        config.topology.preset = protocol.topology_low
    elif mode == "boundary_low":
        config.modulation.delta_omega = protocol.boundary_low * config.omega_a
    return config


def simulate_protocol(
    config: SimulationConfig,
    protocol: ProtocolConfig,
    out_dir: Path,
) -> Dict[str, Dict[str, float]]:
    outputs: Dict[str, Dict[str, float]] = {}
    for mode in [
        "treatment",
        "coherence_low",
        "topology_low",
        "boundary_low",
        "chain_swap",
        "drive_reversal",
    ]:
        sim_config = config.copy(deep=True)
        if mode == "drive_reversal":
            sim_config.drive_phase += 3.14159
        elif mode != "chain_swap":
            _apply_protocol(sim_config, protocol, mode)
        if sim_config.topology.preset in PRESETS:
            preset = PRESETS[sim_config.topology.preset]
            sim_config.topology.n_nodes = preset.n_nodes
            sim_config.topology.n_edges = preset.n_edges
            sim_config.topology.cycle_count = preset.cycle_count
            sim_config.topology.max_cycles = preset.max_cycles
        if mode == "chain_swap":
            series = generate_time_series(sim_config, rng=get_rng(sim_config.seed))
            series["ia"], series["ib"] = series["ib"], series["ia"]
            series["qa"], series["qb"] = series["qb"], series["qa"]
            metrics = compute_metrics(series, sim_config)
            save_run(series, metrics, sim_config, out_dir / mode)
            outputs[mode] = metrics
        else:
            outputs[mode] = generate_and_save(sim_config, out_dir / mode)
    return outputs
