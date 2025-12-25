import numpy as np

from bao_overlap.bao_template import bao_template
from bao_overlap.correlation import compute_pair_counts_bruteforce, landy_szalay


def test_landy_szalay_zero_uniform():
    rng = np.random.default_rng(0)
    data_xyz = rng.uniform(0.0, 1.0, size=(80, 3))
    rand_xyz = rng.uniform(0.0, 1.0, size=(80, 3))
    s_edges = np.linspace(0.0, 1.8, 6)
    mu_edges = np.linspace(0.0, 1.0, 5)

    counts = compute_pair_counts_bruteforce(
        data_xyz=data_xyz,
        rand_xyz=rand_xyz,
        s_edges=s_edges,
        mu_edges=mu_edges,
    )
    xi = landy_szalay(counts)
    assert abs(float(np.mean(xi))) < 0.1
    assert float(np.median(np.abs(xi))) < 0.2


def test_paircount_decomposition():
    rng = np.random.default_rng(1)
    data_xyz = rng.uniform(0.0, 1.0, size=(12, 3))
    data_weights = rng.uniform(0.5, 1.5, size=12)
    data_bins = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 1, 2])
    s_edges = np.linspace(0.0, 1.8, 5)
    mu_edges = np.linspace(0.0, 1.0, 4)

    dd_all = compute_pair_counts_bruteforce(
        data_xyz=data_xyz,
        rand_xyz=data_xyz,
        s_edges=s_edges,
        mu_edges=mu_edges,
        data_weights=data_weights,
        rand_weights=data_weights,
    ).dd

    dd_within = np.zeros_like(dd_all)
    dd_cross = np.zeros_like(dd_all)
    for i in range(3):
        mask_i = data_bins == i
        dd_within += compute_pair_counts_bruteforce(
            data_xyz=data_xyz[mask_i],
            rand_xyz=data_xyz[mask_i],
            s_edges=s_edges,
            mu_edges=mu_edges,
            data_weights=data_weights[mask_i],
            rand_weights=data_weights[mask_i],
        ).dd
        for j in range(i + 1, 3):
            mask_j = data_bins == j
            cross_counts = compute_pair_counts_bruteforce(
                data_xyz=data_xyz[mask_i],
                rand_xyz=data_xyz[mask_j],
                s_edges=s_edges,
                mu_edges=mu_edges,
                data_weights=data_weights[mask_i],
                rand_weights=data_weights[mask_j],
            )
            dd_cross += cross_counts.dr

    dd_reconstructed = dd_within + dd_cross
    assert np.allclose(dd_reconstructed, dd_all, rtol=1.0e-6, atol=1.0e-8)


def test_template_scale_reasonable():
    s = np.linspace(60.0, 160.0, 30)
    xi = bao_template(
        s,
        r_d=147.09,
        sigma_nl=10.0,
        omega_m=0.31,
        omega_b=0.049,
        h=0.676,
        n_s=0.97,
        sigma8=0.81,
    )
    assert 1.0e-5 < np.max(np.abs(xi)) < 1.0
