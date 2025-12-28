"""Tests for correlation and coherence metrics."""

import numpy as np

from mverse_channel.metrics.correlations import (
    cross_spectral_density,
    coherence_peak_magnitude,
    coherence_magnitude,
)


def test_metrics_coherence_increases_with_correlation():
    """Test that coherence increases with signal correlation."""
    rng = np.random.default_rng(42)
    base = rng.normal(size=256)
    correlated = base + 0.1 * rng.normal(size=256)
    _, _, _, coherence = cross_spectral_density(base, correlated, fs=50.0)
    assert float(np.max(coherence)) > 0.1


def test_coherence_magnitude_phase_invariant():
    """Test that coherence_magnitude is phase-invariant."""
    rng = np.random.default_rng(123)
    n = 200

    # Base signal
    t = np.linspace(0, 2 * np.pi, n)
    xa = np.sin(t) + 0.3 * rng.normal(size=n)

    # In-phase correlated signal
    xb_inphase = 0.8 * xa + 0.3 * rng.normal(size=n)

    # Out-of-phase correlated signal (shifted by pi/2)
    xb_outphase = 0.8 * np.roll(xa, n // 4) + 0.3 * rng.normal(size=n)

    coh_inphase = coherence_peak_magnitude(xa, xb_inphase, fs=50.0)
    coh_outphase = coherence_peak_magnitude(xa, xb_outphase, fs=50.0)

    # Both should have similar coherence magnitude despite phase difference
    assert abs(coh_inphase - coh_outphase) < 0.3, (
        f"Coherence magnitude not phase-invariant: {coh_inphase:.3f} vs {coh_outphase:.3f}"
    )


def test_coherence_magnitude_increases_with_correlation_strength():
    """Test that coherence_mag monotonically increases with correlation."""
    rng = np.random.default_rng(456)
    n = 200
    xa = rng.normal(size=n)

    coh_values = []
    for coupling in [0.0, 0.3, 0.6, 0.9]:
        xb = coupling * xa + (1 - coupling) * rng.normal(size=n)
        coh = coherence_peak_magnitude(xa, xb, fs=50.0)
        coh_values.append(coh)

    # Should be approximately monotonic
    for i in range(len(coh_values) - 1):
        assert coh_values[i + 1] >= coh_values[i] - 0.1, (
            f"Coherence not monotonic: {coh_values}"
        )
