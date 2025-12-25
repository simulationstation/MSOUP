import numpy as np

from msoup.constraints import build_constraint_pair
from msoup.coarsegrain import coarsegrain_blocks, coarsegrain_random


def test_coarsegrain_monotone_nvars():
    rng = np.random.default_rng(7)
    pair = build_constraint_pair(rng, n=64, m_base=32, delta_m=4, density=0.3)
    block_sizes = (1, 2, 4, 8, 16)
    cg1 = coarsegrain_blocks(pair, block_sizes)
    cg2 = coarsegrain_random(pair, block_sizes, rng=rng)

    n_vars_cg1 = [entry.n_vars for entry in cg1]
    n_vars_cg2 = [entry.n_vars for entry in cg2]

    assert n_vars_cg1 == sorted(n_vars_cg1, reverse=True)
    assert n_vars_cg2 == sorted(n_vars_cg2, reverse=True)
