import numpy as np

from bao_overlap.density_field import DensityField
from bao_overlap.overlap_metric import compute_environment


def test_environment_normalization():
    grid = np.zeros((4, 4, 4), dtype="f4")
    grid[1:3, 1:3, 1:3] = 1.0
    field = DensityField(grid=grid, origin=np.array([0.0, 0.0, 0.0]), cell_size=1.0)
    galaxy_xyz = np.array([[1.5, 1.5, 1.5], [2.5, 2.5, 2.5]])
    pairs = np.array([
        [[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]],
        [[0.5, 0.5, 0.5], [3.0, 3.0, 3.0]],
    ])
    rng = np.random.default_rng(0)
    env = compute_environment(
        field=field,
        galaxy_xyz=galaxy_xyz,
        pair_xyz=pairs,
        step=0.5,
        rng=rng,
        subsample=1.0,
        delta_threshold=0.5,
        min_volume=1,
        normalize_output=True,
        normalization_method="mean_std",
        primary="E1",
    )
    assert env.per_pair.shape[0] == pairs.shape[0]
    assert np.isclose(env.per_pair.mean(), 0.0, atol=1e-6)
