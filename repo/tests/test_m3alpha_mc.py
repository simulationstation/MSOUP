import numpy as np

from m3alpha.xy_mc import MCConfig, run_mc


def test_mc_deterministic_seed() -> None:
    config = MCConfig(
        lattice_sizes=(8,),
        temperatures=(0.9, 1.0),
        thermalization_sweeps=50,
        measurement_sweeps=60,
        sweep_stride=5,
        seed=7,
    )
    result_a = run_mc(config)
    result_b = run_mc(config)

    for l_size in config.lattice_sizes:
        np.testing.assert_allclose(
            result_a.helicity_modulus[l_size],
            result_b.helicity_modulus[l_size],
            rtol=1e-6,
            atol=1e-6,
        )
