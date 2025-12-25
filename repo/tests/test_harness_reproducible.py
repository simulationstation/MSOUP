import numpy as np

from msoup.harness import HarnessConfig, run_harness


def test_harness_reproducible() -> None:
    config = HarnessConfig(
        n_values=(16,),
        densities=(0.5,),
        delta_m=2,
        j_over_t_grid=tuple(np.logspace(-1, 1, 5)),
        instances_per_setting=4,
        coarsegrain_scales=(1, 2, 4),
        n_samples=128,
        seed=7,
    )
    first = run_harness(config).to_dict()
    second = run_harness(config).to_dict()
    assert first == second
