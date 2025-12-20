import numpy as np

from cowls_field_study.highpass import highpass_fourier


def test_highpass_removes_low_modes():
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    signal = np.sin(theta) + 0.2 * np.sin(7 * theta)
    hp, _ = highpass_fourier(signal, valid_mask=np.ones_like(signal, dtype=bool), m_cut=3, theta=theta)
    target = 0.2 * np.sin(7 * theta)
    # Low-frequency content should be suppressed
    low_mode_overlap = np.abs(np.nanmean(hp * np.sin(theta)))
    assert low_mode_overlap < 1e-2
    # High-frequency component should be retained
    assert np.nanstd(hp - target) < 0.05


def test_highpass_robust_to_missing_bins():
    rng = np.random.default_rng(0)
    theta = np.linspace(0, 2 * np.pi, 180, endpoint=False)
    signal = np.sin(theta) + 0.2 * np.sin(7 * theta)
    mask = rng.random(signal.size) > 0.3
    hp, _ = highpass_fourier(signal, valid_mask=mask, m_cut=3, theta=theta)
    target = 0.2 * np.sin(7 * theta)
    assert np.nanstd(hp[mask] - target[mask]) < 0.07
