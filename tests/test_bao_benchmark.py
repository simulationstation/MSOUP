from msoup_purist_closure.bao_benchmark import BENCHMARK_POINTS, run_bao_benchmark


def test_bao_benchmark_creates_outputs(tmp_path):
    df, output_dir = run_bao_benchmark(tmp_path)
    assert not df.empty
    assert (output_dir / "bao_benchmark.csv").exists()
    assert (output_dir / "bao_benchmark.md").exists()
    # All rows should be within the specified tolerance
    assert df["within_tolerance"].all()
    # Benchmarks should match the embedded table length
    assert len(df) == len(BENCHMARK_POINTS)
