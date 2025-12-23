import numpy as np

from bao_overlap.correlation import compute_pair_counts, landy_szalay


def test_correlation_shapes():
    data = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    rand = np.array([[0.0, 1.0, 0.0], [1.0, 1.0, 0.0]])
    s_edges = np.array([0.0, 2.0])
    mu_edges = np.array([0.0, 1.0])
    counts = compute_pair_counts(data, rand, s_edges, mu_edges)
    xi = landy_szalay(counts)
    assert xi.shape == (1, 1)
