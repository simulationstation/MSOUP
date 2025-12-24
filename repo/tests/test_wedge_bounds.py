import numpy as np

from bao_overlap.correlation import parse_wedge_bounds, wedge_xi


def test_parse_wedge_bounds_numeric():
    wedge_cfg = {"mu_min": 0.0, "mu_max": 0.2}
    assert parse_wedge_bounds(wedge_cfg) == (0.0, 0.2)


def test_wedge_xi_applies_bounds():
    xi = np.arange(12.0).reshape(3, 4)
    mu_edges = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    wedge = (0.0, 0.5)
    values = wedge_xi(xi, mu_edges, wedge)
    expected = np.mean(xi[:, :2], axis=1)
    assert np.allclose(values, expected)
