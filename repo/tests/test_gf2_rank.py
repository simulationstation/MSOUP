import numpy as np

from msoup.gf2 import gf2_rank, gf2_row_reduce


def test_gf2_rank_invariants():
    rng = np.random.default_rng(123)
    for _ in range(5):
        mat = rng.integers(0, 2, size=(12, 10), dtype=np.uint8)
        rank = gf2_rank(mat)
        reduced = gf2_row_reduce(mat)
        assert rank == reduced.rank
        assert rank == gf2_rank(reduced.matrix)

        # Add linear combination of rows; rank should not increase.
        combo = np.zeros(mat.shape[1], dtype=np.uint8)
        for row in mat[:3]:
            combo ^= row
        mat_aug = np.vstack([mat, combo])
        assert gf2_rank(mat_aug) == rank
