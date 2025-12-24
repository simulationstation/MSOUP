import numpy as np

from bao_overlap.density_field import DensityField
from bao_overlap.overlap_metric import compute_per_galaxy_mean_e1


def test_e1_not_all_invalid_toy() -> None:
    grid = np.zeros((20, 20, 20), dtype="f4")
    field = DensityField(
        grid=grid,
        origin=np.zeros(3, dtype="f4"),
        cell_sizes=np.ones(3, dtype="f4"),
        grid_shape=grid.shape,
    )

    rng = np.random.default_rng(123)
    galaxy_xyz = rng.uniform(1.0, 18.0, size=(30, 3)).astype("f4")

    per_galaxy, meta = compute_per_galaxy_mean_e1(
        field=field,
        galaxy_xyz=galaxy_xyz,
        s_min=1.0,
        s_max=5.0,
        step=1.0,
        rng=rng,
        pair_subsample_fraction=1.0,
        max_outside_fraction=0.2,
        max_pairs_per_galaxy=10,
    )

    assert np.any(np.isfinite(per_galaxy))
    assert int(np.sum(meta["valid_pairs"])) > 0
