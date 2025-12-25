from msoup.experiments import ExperimentConfig, run_experiments


def test_reproducibility_small_config():
    config = ExperimentConfig(
        seed=11,
        sizes=(32,),
        densities=(0.7,),
        delta_m=3,
        j_over_t_grid=(0.1, 1.0),
        n_samples=256,
        coarse_block_sizes=(1, 2, 4),
        coarse_pairs=2,
    )
    out1 = run_experiments(config)
    out2 = run_experiments(config)

    assert out1.combined == out2.combined
    assert out1.entropy_checks == out2.entropy_checks
    assert out1.dual_sweep == out2.dual_sweep
