import importlib

import pytest

from mverse_channel.config import SimulationConfig
from mverse_channel.physics.baseline_model import simulate_baseline


def test_baseline_shapes():
    if importlib.util.find_spec("qutip") is None:
        pytest.skip("qutip not available")
    config = SimulationConfig(n_levels=3, duration=1.0, dt=0.2)
    output = simulate_baseline(config)
    assert output["ia"].shape == output["qa"].shape
    assert output["ib"].shape == output["qb"].shape
    assert output["ia"].shape[0] == len(output["t"])
