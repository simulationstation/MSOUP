import numpy as np

from bao_overlap.correlation import compute_pair_counts_by_environment, compute_pair_counts_simple


def test_pair_counts_by_bin_shapes():
    data = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
    ])
    rand = np.array([
        [0.5, 0.5, 0.0],
        [1.5, 0.5, 0.0],
        [0.5, 1.5, 0.0],
        [1.5, 1.5, 0.0],
    ])
    data_bins = np.array([0, 1, 0, 1])
    rand_bins = np.array([0, 1, 0, 1])
    s_edges = np.array([0.0, 2.0])
    mu_edges = np.array([0.0, 1.0])

    counts_by_bin = compute_pair_counts_by_environment(
        data,
        rand,
        data_bins,
        rand_bins,
        n_bins=2,
        s_edges=s_edges,
        mu_edges=mu_edges,
        pair_counter=compute_pair_counts_simple,
        verbose=False,
    )

    stacked = np.stack([counts_by_bin[b].dd for b in range(2)], axis=0)
    assert stacked.shape == (2, 1, 1)
