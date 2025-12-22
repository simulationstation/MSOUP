import numpy as np
import pytest

from msoup_purist_closure.config import DEFAULT_H_EARLY, DEFAULT_OMEGA_L0, DEFAULT_OMEGA_M0, DistanceConfig
from msoup_purist_closure.observables import bao_predict

COSMO_KWARGS = {
    "h_early": DEFAULT_H_EARLY,
    "omega_m0": DEFAULT_OMEGA_M0,
    "omega_L0": DEFAULT_OMEGA_L0,
}


@pytest.mark.parametrize(
    "z,expected_dm,tolerance",
    [
        (0.38, 10.48, 0.03),
        (0.51, 13.40, 0.03),
        (0.70, 17.29, 0.03),
    ],
)
def test_dm_rd_regression(z, expected_dm, tolerance):
    rd = DistanceConfig().rd_mpc
    pred = bao_predict(z, "DM/rd", delta_m=0.0, rd_mpc=rd, **COSMO_KWARGS)
    assert pred == pytest.approx(expected_dm, rel=tolerance)


@pytest.mark.parametrize(
    "z,expected_dh,tolerance",
    [
        (0.38, 24.73, 0.03),
        (0.51, 22.87, 0.03),
        (0.70, 20.36, 0.03),
        (2.33, 8.67, 0.05),
    ],
)
def test_dh_rd_regression(z, expected_dh, tolerance):
    rd = DistanceConfig().rd_mpc
    pred = bao_predict(z, "DH/rd", delta_m=0.0, rd_mpc=rd, **COSMO_KWARGS)
    assert pred == pytest.approx(expected_dh, rel=tolerance)


def test_dv_rd_low_z_positive_and_plausible():
    rd = DistanceConfig().rd_mpc
    pred = bao_predict(0.15, "DV/rd", delta_m=0.0, rd_mpc=rd, **COSMO_KWARGS)
    assert pred > 0
    assert 4.0 <= pred <= 5.5


def test_rd_fid_scaling_helpers_apply_expected_factor():
    rd = DistanceConfig().rd_mpc
    rd_fid = rd * 1.05
    base = bao_predict(0.38, "DM/rd", delta_m=0.0, rd_mpc=rd, **COSMO_KWARGS)
    scaled = bao_predict(
        0.38,
        "DM/rd",
        delta_m=0.0,
        rd_mpc=rd,
        rd_fid_mpc=rd_fid,
        rd_scaling="multiply_by_rd_over_rdfid",
        **COSMO_KWARGS,
    )
    assert scaled == pytest.approx(base * (rd / rd_fid))
