import numpy as np

from bao_overlap.bao_template import bao_wiggle_components
from bao_overlap.diagnostic_bao import fit_wiggle_only_monopole


def test_wiggle_only_injection_recovery():
    s = np.linspace(60.0, 160.0, 30)
    template_params = {
        "r_d": 147.0,
        "sigma_nl": 8.0,
        "omega_m": 0.31,
        "omega_b": 0.049,
        "h": 0.676,
        "n_s": 0.97,
        "sigma8": 0.81,
    }

    _, _, xi_wiggle = bao_wiggle_components(
        s,
        r_d=template_params["r_d"],
        omega_m=template_params["omega_m"],
        omega_b=template_params["omega_b"],
        h=template_params["h"],
        n_s=template_params["n_s"],
        sigma8=template_params["sigma8"],
        sigma_nl=template_params["sigma_nl"],
    )

    b2 = 2.0
    poly = 0.01 + (-1.0e-4) * s + (1.0e-7) * s**2
    xi = b2 * xi_wiggle + poly
    cov = np.eye(len(s)) * 1.0e-4

    result = fit_wiggle_only_monopole(
        s=s,
        xi=xi,
        covariance=cov,
        fit_range=(60.0, 160.0),
        template_params=template_params,
        alpha_bounds=(0.6, 1.4),
        alpha_scan_points=41,
    )

    assert abs(result.alpha - 1.0) < 0.05
    assert result.alpha_hit_bounds is False
