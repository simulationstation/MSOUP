import numpy as np

from mverse_channel.metrics.correlations import cross_spectral_density


def test_metrics_coherence_increases_with_correlation():
    rng = np.random.default_rng(42)
    base = rng.normal(size=256)
    correlated = base + 0.1 * rng.normal(size=256)
    _, _, _, coherence = cross_spectral_density(base, correlated, fs=50.0)
    assert float(np.max(coherence)) > 0.1
