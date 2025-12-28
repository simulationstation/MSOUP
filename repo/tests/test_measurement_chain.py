import numpy as np

from mverse_channel.config import MeasurementChainConfig
from mverse_channel.physics.measurement_chain import apply_measurement_chain
from mverse_channel.rng import get_rng


def test_measurement_chain_noise_addition():
    rng = get_rng(123)
    base = np.zeros(100)
    config = MeasurementChainConfig(amp_noise=0.5)
    ia, qa, ib, qb = apply_measurement_chain(base, base, base, base, config, rng)
    assert np.std(ia) > 0.0
    assert ia.shape == qa.shape == ib.shape == qb.shape
