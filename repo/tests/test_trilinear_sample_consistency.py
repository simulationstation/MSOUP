import numpy as np

from bao_overlap.density_field import DensityField, trilinear_sample


def test_trilinear_sample_consistency() -> None:
    grid = np.zeros((2, 2, 2), dtype="f4")
    idx = np.indices(grid.shape).astype("f4")
    grid = idx[0] + idx[1] + idx[2]

    field = DensityField(
        grid=grid,
        origin=np.zeros(3, dtype="f4"),
        cell_sizes=np.ones(3, dtype="f4"),
        grid_shape=grid.shape,
    )

    points = np.array([[0.5, 0.5, 0.5], [0.25, 0.25, 0.25]], dtype="f4")
    values = trilinear_sample(field, points)

    assert np.allclose(values[0], 1.5, atol=1e-6)
    assert np.allclose(values[1], 0.75, atol=1e-6)

    outside = trilinear_sample(field, np.array([[-0.1, 0.0, 0.0]], dtype="f4"))
    assert np.isnan(outside[0])
