import numpy as np

from bao_overlap.density_field import build_grid_spec


def test_grid_covers_span() -> None:
    data_xyz = np.array([[0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]])
    rand_xyz = np.array([[0.0, 0.0, 0.0], [1000.0, 1000.0, 1000.0]])

    grid_spec = build_grid_spec(
        data_xyz=data_xyz,
        random_xyz=rand_xyz,
        target_cell_size=10.0,
        padding=0.0,
        max_n_per_axis=512,
    )

    coverage = grid_spec.cell_sizes * np.array(grid_spec.grid_shape)
    span = grid_spec.maxs - grid_spec.origin

    assert np.all(coverage >= span - 1e-6)
    assert np.allclose(grid_spec.cell_sizes, 10.0, atol=1e-6)
