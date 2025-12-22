import numpy as np

from msoup_purist_closure import model


def test_pinned_fraction_limits():
    z = np.array([0.0, 0.5, 1.0])
    val = model.pinned_fraction(z, delta_m=0.0)
    assert np.allclose(val, 0.5)


def test_leak_fraction_monotonic_delta_m():
    z = np.linspace(0, 1, 5)
    f = model.leak_fraction(z, delta_m=1.5)
    assert np.all((f >= 0) & (f <= 1))
    assert f[0] >= f[-1]
