import numpy as np
import pytest

from msoup_purist_closure.constants import SPEED_OF_LIGHT_KM_S
from msoup_purist_closure.config import DEFAULT_H_EARLY, DEFAULT_OMEGA_L0, DEFAULT_OMEGA_M0
from msoup_purist_closure.model import h_eff
from msoup_purist_closure.distances import angular_diameter_distance
from msoup_purist_closure.observables import bao_dh, bao_dm, bao_dv


COSMO_KWARGS = {
    "h_early": DEFAULT_H_EARLY,
    "omega_m0": DEFAULT_OMEGA_M0,
    "omega_L0": DEFAULT_OMEGA_L0,
}
DIST_KWARGS = {**COSMO_KWARGS, "c_km_s": SPEED_OF_LIGHT_KM_S}


def test_dh_matches_c_over_h():
    z = np.array([0.1, 0.5, 1.0])
    expected = SPEED_OF_LIGHT_KM_S / h_eff(z, delta_m=0.0, **COSMO_KWARGS)
    computed = bao_dh(z, delta_m=0.0, **DIST_KWARGS)
    assert np.allclose(computed, expected, rtol=1e-10, atol=0.0)


def test_dm_equals_one_plus_z_times_da():
    z = np.array([0.1, 0.5, 1.0])
    da = angular_diameter_distance(z, delta_m=0.0, **DIST_KWARGS)
    dm_direct = (1 + z) * da
    dm_from_fn = bao_dm(z, delta_m=0.0, **DIST_KWARGS)
    assert np.allclose(dm_from_fn, dm_direct, rtol=1e-12, atol=0.0)


def test_dv_identity():
    z = np.array([0.15, 0.5])
    da = angular_diameter_distance(z, delta_m=0.0, **DIST_KWARGS)
    h_vals = h_eff(z, delta_m=0.0, **COSMO_KWARGS)
    manual = ((1 + z) ** 2 * da ** 2 * (SPEED_OF_LIGHT_KM_S * z / h_vals)) ** (1.0 / 3.0)
    dv = bao_dv(z, delta_m=0.0, **DIST_KWARGS)
    assert np.allclose(dv, manual, rtol=1e-12, atol=0.0)
