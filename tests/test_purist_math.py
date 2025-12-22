import numpy as np

from msoup_purist_closure import model
from msoup_purist_closure.config import DEFAULT_H_EARLY, DEFAULT_OMEGA_L0, DEFAULT_OMEGA_M0


def test_pinned_fraction_limits():
    z = np.array([0.0, 0.5, 1.0])
    val = model.pinned_fraction(z, delta_m=0.0)
    assert np.isclose(val[0], 0.5)
    assert val[-1] > val[0]


def test_leak_fraction_monotonic_delta_m():
    z = np.linspace(0, 1, 5)
    f = model.leak_fraction(z, delta_m=1.5)
    assert np.all((f >= 0) & (f <= 1))
    assert f[0] >= f[-1]


def test_leak_fraction_zero_when_delta_zero():
    z = np.linspace(0, 3, 50)
    f = model.leak_fraction(z, delta_m=0.0)
    assert np.allclose(f, 0.0, atol=1e-12)


def test_h_eff_matches_lcdm_at_delta_zero():
    z = np.linspace(0, 3, 25)
    lcdm = DEFAULT_H_EARLY * np.sqrt(DEFAULT_OMEGA_M0 * (1 + z) ** 3 + DEFAULT_OMEGA_L0)
    h_val = model.h_eff(z, delta_m=0.0)
    assert np.allclose(h_val, lcdm, rtol=1e-10, atol=0.0)
