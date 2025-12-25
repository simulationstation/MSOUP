import numpy as np
import pytest

from bao_overlap import correlation
from bao_overlap.correlation import compute_xi_s_mu, wedge_xi


pytestmark = pytest.mark.skipif(
    not correlation.HAS_TREECORR, reason="TreeCorr not available"
)


def _treecorr_xi_1d(data_xyz, rand_xyz, s_edges, data_w, rand_w):
    import treecorr

    config = {
        "min_sep": s_edges[0],
        "max_sep": s_edges[-1],
        "nbins": len(s_edges) - 1,
        "bin_type": "Linear",
        "num_threads": 1,
        "verbose": 0,
    }

    data_cat = treecorr.Catalog(
        x=data_xyz[:, 0],
        y=data_xyz[:, 1],
        z=data_xyz[:, 2],
        w=data_w,
    )
    rand_cat = treecorr.Catalog(
        x=rand_xyz[:, 0],
        y=rand_xyz[:, 1],
        z=rand_xyz[:, 2],
        w=rand_w,
    )

    dd = treecorr.NNCorrelation(**config)
    dr = treecorr.NNCorrelation(**config)
    rr = treecorr.NNCorrelation(**config)
    dd.process(data_cat)
    dr.process(data_cat, rand_cat)
    rr.process(rand_cat)
    dd.calculateXi(rr=rr, dr=dr)
    return dd.xi


def test_randoms_as_data_null_tail():
    rng = np.random.default_rng(123)
    xyz = rng.uniform(0.0, 1000.0, size=(400, 3))
    weights = np.ones(len(xyz))

    s_edges = np.arange(30.0, 210.0, 10.0)
    mu_edges = np.linspace(0.0, 1.0, 11)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])

    xi = compute_xi_s_mu(
        xyz,
        xyz,
        s_edges=s_edges,
        mu_edges=mu_edges,
        data_weights=weights,
        rand_weights=weights,
        verbose=False,
    ).xi

    tangential = wedge_xi(xi, mu_edges, (0.0, 0.2))
    monopole = np.mean(xi, axis=1)

    tail_mask = s_centers > 155.0
    assert float(np.max(np.abs(tangential[tail_mask]))) < 0.02
    assert float(np.max(np.abs(monopole[tail_mask]))) < 0.02


def test_treecorr_monopole_consistency():
    rng = np.random.default_rng(456)
    data_xyz = rng.uniform(0.0, 800.0, size=(400, 3))
    rand_xyz = rng.uniform(0.0, 800.0, size=(800, 3))
    data_w = np.ones(len(data_xyz))
    rand_w = np.ones(len(rand_xyz))

    s_edges = np.arange(30.0, 210.0, 15.0)
    mu_edges = np.linspace(0.0, 1.0, 11)

    xi_s_mu = compute_xi_s_mu(
        data_xyz,
        rand_xyz,
        s_edges=s_edges,
        mu_edges=mu_edges,
        data_weights=data_w,
        rand_weights=rand_w,
        verbose=False,
    ).xi
    monopole = np.mean(xi_s_mu, axis=1)

    xi_1d = _treecorr_xi_1d(data_xyz, rand_xyz, s_edges, data_w, rand_w)

    max_abs_diff = float(np.max(np.abs(monopole - xi_1d)))
    assert max_abs_diff < 0.02
