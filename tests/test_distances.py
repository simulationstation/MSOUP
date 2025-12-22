import numpy as np

from msoup_purist_closure.distances import angular_diameter_distance, comoving_distance, luminosity_distance


def test_comoving_distance_increases():
    z = np.linspace(0, 1, 5)
    chi = comoving_distance(z, delta_m=0.0)
    assert np.all(np.diff(chi) >= 0)


def test_luminosity_distance_scaling():
    z = np.array([0.5])
    dl = luminosity_distance(z, delta_m=0.0)
    da = angular_diameter_distance(z, delta_m=0.0)
    assert dl[0] > da[0]
