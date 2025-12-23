import pathlib

import numpy as np
import pandas as pd

from msoup_td_maxid.config import ComputeConfig, MaxIDConfig, PathsConfig, PriorConfig
from msoup_td_maxid.inference import _delta_grid, run_inference
from msoup_td_maxid.likelihoods import histogram_logpdf_from_samples, robust_residual_loglike
from msoup_td_maxid.model import D_dt_pred


def _make_cfg(tmp_path: pathlib.Path, filename: str = "td_master.csv", **kwargs) -> MaxIDConfig:
    paths = PathsConfig(td_master=tmp_path / filename, posterior_dir=tmp_path, output_dir=tmp_path / "out")
    priors = PriorConfig(**kwargs) if kwargs else PriorConfig()
    compute = ComputeConfig(max_workers=1, max_rss_gb=1.5, rss_check_interval=2, chunk_size=1024)
    return MaxIDConfig(paths=paths, priors=priors, compute=compute)


def test_histogram_and_student_t_behave():
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=0.0, scale=1.0, size=500)
    grid = np.linspace(-1, 1, 5)
    logpdf = histogram_logpdf_from_samples(samples, grid)
    assert np.all(np.isfinite(logpdf))
    center_idx = len(grid) // 2
    assert logpdf[center_idx] > logpdf[0]

    loglike = robust_residual_loglike(np.array([0.0, 1.0]), np.array([1.0, 2.0]), df=4)
    assert loglike.shape == (2,)
    assert loglike[0] > loglike[1]


def test_base_inference_recovers_delta_m(tmp_path):
    cfg = _make_cfg(tmp_path)
    delta_true = 0.12
    rows = []
    for i, (zl, zs) in enumerate([(0.3, 1.4), (0.5, 1.8), (0.7, 2.2)]):
        pred = D_dt_pred(zl, zs, delta_true, cfg.h_early, cfg.omega_m0, cfg.omega_L0)
        rows.append({"lens_id": f"L{i}", "z_lens": zl, "z_source": zs, "observable_type": "D_dt", "value": pred, "sigma": 0.05 * pred})
    pd.DataFrame(rows).to_csv(cfg.paths.td_master, index=False)

    result = run_inference(cfg, mode="base")
    assert abs(result["summary"]["median"] - delta_true) < 0.05
    assert result["kill_table"]["K-PSIS"]


def test_hier_mode_flags_dominance(tmp_path):
    cfg = _make_cfg(tmp_path)
    delta_true = 0.2
    rows = []
    for i, (zl, zs, scale) in enumerate([(0.4, 1.6, 0.02), (0.6, 2.0, 0.5), (0.8, 2.4, 0.6)]):
        pred = D_dt_pred(zl, zs, delta_true, cfg.h_early, cfg.omega_m0, cfg.omega_L0)
        rows.append({"lens_id": f"L{i}", "z_lens": zl, "z_source": zs, "observable_type": "D_dt", "value": pred, "sigma": scale * pred})
    pd.DataFrame(rows).to_csv(cfg.paths.td_master, index=False)

    result = run_inference(cfg, mode="hier")
    assert result["verdict"] in {"DOMINATED", "ROBUST"}
    assert not result["kill_table"]["K-DOM"] or not result["kill_table"]["K-LOO"]


def test_contam_mode_handles_outlier(tmp_path):
    cfg = _make_cfg(tmp_path)
    delta_true = 0.1
    rows = []
    base = [(0.35, 1.4), (0.55, 1.8), (0.75, 2.2)]
    for i, (zl, zs) in enumerate(base):
        pred = D_dt_pred(zl, zs, delta_true, cfg.h_early, cfg.omega_m0, cfg.omega_L0)
        sigma = 0.04 * pred
        value = pred * (1.3 if i == 2 else 1.0)
        rows.append({"lens_id": f"L{i}", "z_lens": zl, "z_source": zs, "observable_type": "D_dt", "value": value, "sigma": sigma})
    pd.DataFrame(rows).to_csv(cfg.paths.td_master, index=False)

    result = run_inference(cfg, mode="contam")
    assert result["summary"]["median"] > -0.2
    assert result["summary"]["median"] < 0.4
    assert "kill_table" in result


def test_edge_mass_not_truncated(tmp_path):
    cfg = _make_cfg(tmp_path, delta_m_min=-1.0, delta_m_max=3.0)
    delta_true = 0.8
    rows = []
    for i, (zl, zs) in enumerate([(0.3, 1.2), (0.5, 1.6), (0.7, 1.9), (0.9, 2.4)]):
        pred = D_dt_pred(zl, zs, delta_true, cfg.h_early, cfg.omega_m0, cfg.omega_L0)
        rows.append({"lens_id": f"L{i}", "z_lens": zl, "z_source": zs, "observable_type": "D_dt", "value": pred, "sigma": 0.03 * pred})
    pd.DataFrame(rows).to_csv(cfg.paths.td_master, index=False)
    result = run_inference(cfg, mode="base")
    assert result["edge_mass_high"] < 0.01
    assert not result["bound_truncation_flag"]


def test_loo_refits_written(tmp_path):
    cfg = _make_cfg(tmp_path, delta_m_min=-1.0, delta_m_max=3.0)
    delta_true = 0.2
    rows = []
    for i in range(7):
        zl = 0.3 + 0.1 * i
        zs = 1.5 + 0.1 * i
        pred = D_dt_pred(zl, zs, delta_true, cfg.h_early, cfg.omega_m0, cfg.omega_L0)
        rows.append({"lens_id": f"L{i}", "z_lens": zl, "z_source": zs, "observable_type": "D_dt", "value": pred, "sigma": 0.05 * pred})
    pd.DataFrame(rows).to_csv(cfg.paths.td_master, index=False)
    result = run_inference(cfg, mode="hier")
    csv_path = cfg.paths.output_dir / "hier" / "loo_refits_hier.csv"
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    assert len(df) == 7
    assert "loo_delta_m_median" in df.columns
    assert result["loo_refits"]


def test_delta_m_grid_bounds():
    cfg = MaxIDConfig()
    grid = _delta_grid(cfg)
    assert np.isclose(grid.min(), -1.0)
    assert np.isclose(grid.max(), 3.0)
