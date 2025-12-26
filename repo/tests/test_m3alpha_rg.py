import math

from m3alpha.rg_bkt import RGConfig, find_critical_k


def test_rg_critical_k_close_to_bkt() -> None:
    config = RGConfig(l_max=6.0, n_steps=600, k_min=0.4, k_max=1.2, critical_y0=0.02)
    k_c = find_critical_k(config.critical_y0, config)
    target = 2.0 / math.pi
    assert abs(k_c - target) < 0.08
