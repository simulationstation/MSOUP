import numpy as np

from msoup.gf2 import gf2_is_consistent, gf2_rank
from msoup.xor_model import generate_constraint_pair


def test_rank_increase_matches_delta_m() -> None:
    rng = np.random.default_rng(42)
    pair = generate_constraint_pair(rng, n=12, m_base=10, delta_m=3)
    assert gf2_rank(pair.a_plus) - gf2_rank(pair.a_minus) == 3
    assert np.array_equal(pair.a_plus[: pair.a_minus.shape[0]], pair.a_minus)
    assert gf2_is_consistent(pair.a_minus, pair.b_minus)
    assert gf2_is_consistent(pair.a_plus, pair.b_plus)
