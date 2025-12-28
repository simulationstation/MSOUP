"""Topology proxies for loop dominance."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopologyPreset:
    n_nodes: int
    n_edges: int
    cycle_count: int
    max_cycles: int


PRESETS = {
    "tree": TopologyPreset(n_nodes=4, n_edges=3, cycle_count=0, max_cycles=6),
    "single_loop": TopologyPreset(n_nodes=4, n_edges=4, cycle_count=1, max_cycles=6),
    "multi_loop": TopologyPreset(n_nodes=6, n_edges=7, cycle_count=3, max_cycles=6),
    "linked_loops": TopologyPreset(n_nodes=8, n_edges=10, cycle_count=6, max_cycles=6),
}


def topology_index(
    n_nodes: int,
    n_edges: int,
    cycle_count: int,
    max_cycles: int,
) -> float:
    """Normalize cycle count to [0, 1] as a loop dominance proxy."""
    if max_cycles <= 0:
        return 0.0
    normalized = cycle_count / max_cycles
    return max(0.0, min(1.0, normalized))
