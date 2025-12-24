import numpy as np

from bao_overlap.density_field import DensityField
from bao_overlap.overlap_metric import compute_per_galaxy_mean_e1


def test_per_galaxy_mean_e1_constant_field():
    grid = np.ones((4, 4, 4), dtype="f4") * 2.0
    field = DensityField(grid=grid, origin=np.array([0.0, 0.0, 0.0]), cell_size=1.0)
    galaxy_xyz = np.array(
        [
            [0.5, 0.5, 0.5],
            [2.5, 0.5, 0.5],
            [0.5, 2.5, 0.5],
        ]
    )
    rng = np.random.default_rng(0)
    e_raw, meta = compute_per_galaxy_mean_e1(
        field=field,
        galaxy_xyz=galaxy_xyz,
        s_min=0.0,
        s_max=10.0,
        step=0.5,
        rng=rng,
        pair_subsample_fraction=1.0,
        max_outside_fraction=0.0,
    )
    assert np.allclose(e_raw[np.isfinite(e_raw)], 2.0, atol=1e-6)
    assert meta["valid_pairs"].sum() > 0
