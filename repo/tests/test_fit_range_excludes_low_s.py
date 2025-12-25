import numpy as np

from bao_overlap.fitting import fit_wedge


def test_fit_range_excludes_low_s():
    s = np.linspace(40.0, 180.0, 29)
    fit_range = (60.0, 160.0)
    xi = np.sin(s / 100.0)
    cov = np.eye(len(s)) * 0.01

    base_result = fit_wedge(
        s=s,
        xi=xi,
        covariance=cov,
        fit_range=fit_range,
        nuisance_terms=["a0", "a1/s"],
        template_params={
            "r_d": 147.0,
            "sigma_nl": 10.0,
            "omega_m": 0.31,
            "omega_b": 0.049,
            "h": 0.676,
            "n_s": 0.97,
            "sigma8": 0.81,
        },
    )

    low_mask = s < fit_range[0]
    xi_modified = xi.copy()
    xi_modified[low_mask] += 100.0

    modified_result = fit_wedge(
        s=s,
        xi=xi_modified,
        covariance=cov,
        fit_range=fit_range,
        nuisance_terms=["a0", "a1/s"],
        template_params={
            "r_d": 147.0,
            "sigma_nl": 10.0,
            "omega_m": 0.31,
            "omega_b": 0.049,
            "h": 0.676,
            "n_s": 0.97,
            "sigma8": 0.81,
        },
    )

    np.testing.assert_allclose(modified_result.alpha, base_result.alpha, rtol=0.0, atol=1.0e-12)
