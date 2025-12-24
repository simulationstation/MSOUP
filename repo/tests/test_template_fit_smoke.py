import numpy as np

from bao_overlap.fitting import fit_wedge


def test_fit_wedge_smoke():
    s = np.linspace(60, 160, 10)
    xi = np.sin(s / 100.0)
    cov = np.eye(len(s)) * 0.01
    result = fit_wedge(
        s=s,
        xi=xi,
        covariance=cov,
        fit_range=(60.0, 160.0),
        nuisance_terms=["a0", "a1/s"],
        template_params={
            "r_d": 147.0,
            "sigma_nl": 10.0,
            "omega_m": 0.31,
            "omega_b": 0.049,
            "h": 0.676,
            "n_s": 0.97,
        },
    )
    assert 0.8 <= result.alpha <= 1.2
