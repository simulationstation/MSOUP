import importlib

import numpy as np
import pytest

from mverse_channel.config import SimulationConfig
from mverse_channel.physics.extended_model import simulate_extended
from mverse_channel.rng import get_rng


def test_extended_toggle_matches_baseline_when_disabled():
    if importlib.util.find_spec("qutip") is None:
        pytest.skip("qutip not available")
    config = SimulationConfig(n_levels=3, duration=1.0, dt=0.2)
    config.hidden_channel.enabled = False
    rng = get_rng(123)
    base = simulate_extended(config, rng)
    config.hidden_channel.enabled = True
    config.hidden_channel.epsilon_max = 0.0
    rng = get_rng(123)
    extended = simulate_extended(config, rng)
    assert np.allclose(base["ia"], extended["ia"])
