import numpy as np

from mverse_channel.rng import get_rng


def test_rng_reproducibility():
    rng1 = get_rng(123)
    rng2 = get_rng(123)
    assert np.allclose(rng1.normal(size=5), rng2.normal(size=5))
