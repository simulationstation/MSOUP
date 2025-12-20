import numpy as np
import pytest
from astropy.io import fits
from pathlib import Path

from cowls_pocket_study import candidates, io, preprocess, sensitivity, stats_real


def _make_fake_lens(tmp_path: Path, lens_id: str = "M00", amp277: float = 5.0, amp444: float = 8.0) -> io.LensRecord:
    lens_dir = tmp_path / lens_id
    lens_dir.mkdir(parents=True)
    shape = (64, 64)
    y, x = np.indices(shape)
    center = (32, 32)
    r = np.hypot(x - center[0], y - center[1])
    base_noise = 0.5
    ring = np.exp(-(r - 20) ** 2 / (2 * 3**2))

    for band, amp in (("F277W", amp277), ("F444W", amp444)):
        image = amp * ring + np.random.default_rng(0).normal(0, base_noise, size=shape)
        noise = np.full(shape, base_noise)
        fits.writeto(lens_dir / f"{lens_id}_{band}_science.fits", image, overwrite=True)
        fits.writeto(lens_dir / f"{lens_id}_{band}_noise.fits", noise, overwrite=True)
    return io.discover_lenses(data_root=tmp_path)[0]


def test_auto_band_selection_prefers_high_snr(tmp_path):
    lens = _make_fake_lens(tmp_path, amp277=5.0, amp444=15.0)
    band = io.choose_band_for_lens(lens, band="auto")
    assert band == "F444W"


def test_sensitivity_and_candidates_pipeline(tmp_path):
    lens = _make_fake_lens(tmp_path, lens_id="S10")
    band_used, image, noise, _ = io.load_lens_data(lens, band="auto")
    prep = preprocess.build_arc_mask(image, noise, lens)
    window = sensitivity.compute_sensitivity(prep.snr_map, prep.arc_mask, prep.center, lens_id=lens.lens_id)
    cand = candidates.detect_candidates(
        image=image,
        noise=noise,
        preprocess=prep,
        cache_root=tmp_path,
        lens_id=lens.lens_id,
    )

    assert band_used in lens.available_bands
    assert np.isclose(np.mean(window.s_grid), 1.0, atol=1e-3)
    assert prep.arc_mask.sum() > 0
    assert cand.theta.ndim == 1


def test_stats_real_controls():
    theta_by_lens = {"A": np.array([0.0, 0.1, 1.0]), "B": np.array([0.2, 2.5])}
    theta_grid = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    s_grid = np.ones_like(theta_grid)
    windows = {lid: sensitivity.SensitivityWindowReal(theta_grid=theta_grid, s_grid=s_grid, lens_id=lid) for lid in theta_by_lens}
    agg = stats_real.aggregate_clustering(theta_by_lens, windows, theta0=0.5, n_resamples=50, rng=np.random.default_rng(1))
    assert agg.lens_stats, "should compute per-lens stats"
    shuffle = stats_real.shuffle_control(theta_by_lens, windows, theta0=0.5, rng=np.random.default_rng(2), n_resamples=30)
    assert abs(shuffle) < 1.0, "shuffle control should be near zero-ish"
