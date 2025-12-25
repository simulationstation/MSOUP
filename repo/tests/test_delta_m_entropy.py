import numpy as np

from msoup.constraints import build_constraint_pair
from msoup.entropy import delta_entropy


def test_delta_m_entropy_relation():
    rng = np.random.default_rng(42)
    pair = build_constraint_pair(rng, n=16, m_base=8, delta_m=3, density=0.3)
    delta_s = delta_entropy(pair.peeled, pair.pinned)
    expected = -pair.delta_m * np.log(2)
    assert np.isclose(delta_s, expected)
