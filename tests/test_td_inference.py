import pathlib

import numpy as np
import pandas as pd

from msoup_purist_closure.config import DistanceConfig, F0Config, FitConfig, MsoupConfig, ProbeConfig
from msoup_purist_closure.data_utils import compute_ddt_prediction
from msoup_purist_closure.distances import h_eff_ratio
from msoup_purist_closure.td_inference import f0_from_delta_m, load_td_master, run_td_inference


def _make_cfg(tmp_path: pathlib.Path, csv_path: pathlib.Path) -> tuple[MsoupConfig, ProbeConfig]:
    probe = ProbeConfig(
        name="td_test",
        path=str(csv_path),
        type="td",
        z_column="z_lens",
        sigma_column="sigma",
        obs_column="value",
        observable_column="observable_type",
    )
    fit_cfg = FitConfig(h_local=73.0, sigma_local=1.0, delta_m_bounds=(-1, 1), fit_sn_m=False)
    f0_cfg = F0Config(delta_m_min=0.0, delta_m_max=0.6, num_points=81, include_sn=False, include_bao=False, include_td=True)
    cfg = MsoupConfig(
        probes=[probe],
        fit=fit_cfg,
        distance=DistanceConfig(),
        f0=f0_cfg,
        results_dir=tmp_path / "results",
    )
    return cfg, probe


def test_ddt_prediction_positive_and_leak_free(tmp_path):
    csv_path = tmp_path / "td.csv"
    csv_path.write_text("z_lens,z_source,observable_type,value,sigma\n0.5,2.0,D_dt,1,1\n", encoding="utf-8")
    cfg, _ = _make_cfg(tmp_path, csv_path)

    ddt_low = compute_ddt_prediction(0.5, 1.5, 0.0, cfg)
    ddt_high = compute_ddt_prediction(0.5, 2.5, 0.0, cfg)
    assert ddt_low > 0
    assert ddt_high > 0

    h_ratio = h_eff_ratio(np.array([0.3, 1.0]), 0.0, cfg.h_early, cfg.omega_m0, cfg.omega_L0)
    np.testing.assert_allclose(h_ratio, np.ones_like(h_ratio), rtol=1e-6)


def test_td_inference_recovers_delta_m(tmp_path):
    csv_path = tmp_path / "td.csv"
    # Generate synthetic Gaussian measurements around delta_m_true
    delta_m_true = 0.2
    z_lens = [0.3, 0.6]
    z_source = [1.5, 2.2]
    rows = []
    for zl, zs in zip(z_lens, z_source):
        cfg_base, _ = _make_cfg(tmp_path, csv_path)
        ddt_pred = compute_ddt_prediction(zl, zs, delta_m_true, cfg_base)
        rows.append({"z_lens": zl, "z_source": zs, "observable_type": "D_dt", "value": ddt_pred, "sigma": 0.02 * ddt_pred})
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    cfg, probe = _make_cfg(tmp_path, csv_path)
    output_dir = tmp_path / "out"
    result = run_td_inference(cfg, output_dir, probe)
    assert abs(result["summary"]["median"] - delta_m_true) < 0.05
    assert result["f0_summary"]["mean"] >= 0.0


def test_histogram_loader_and_likelihood(tmp_path):
    csv_path = tmp_path / "td_master.csv"
    posterior_path = tmp_path / "posterior.csv"
    rng = np.random.default_rng(0)
    posterior_values = rng.normal(loc=5000.0, scale=200.0, size=500)
    pd.DataFrame({"D_dt": posterior_values}).to_csv(posterior_path, index=False)
    pd.DataFrame(
        [
            {
                "lens_id": "mock",
                "z_lens": 0.5,
                "z_source": 2.0,
                "observable_type": "D_dt",
                "file_ref": posterior_path.name,
            }
        ]
    ).to_csv(csv_path, index=False)

    probe = ProbeConfig(
        name="td_hist",
        path=str(csv_path),
        type="td",
        z_column="z_lens",
        sigma_column="sigma",
        obs_column="value",
        observable_column="observable_type",
        file_ref_column="file_ref",
    )
    lenses, status = load_td_master(probe, data_dir=tmp_path)
    assert status == "OK"
    assert len(lenses) == 1
    assert lenses[0].kind == "hist"
    assert lenses[0].histogram is not None
    assert lenses[0].histogram.sample_std > 0


def test_loader_missing_columns(tmp_path):
    csv_path = tmp_path / "td_missing.csv"
    pd.DataFrame([{"z_lens": 0.5, "observable_type": "D_dt", "value": 1.0, "sigma": 0.1}]).to_csv(csv_path, index=False)
    probe = ProbeConfig(
        name="td_missing",
        path=str(csv_path),
        type="td",
        z_column="z_lens",
        sigma_column="sigma",
        obs_column="value",
        observable_column="observable_type",
    )
    lenses, status = load_td_master(probe)
    assert lenses == []
    assert "missing" in status or "no usable" in status


def test_f0_mapping_monotonic():
    deltas = np.array([0.0, 0.3, 0.6])
    f0_vals = f0_from_delta_m(deltas, 0.315, 0.685)
    assert f0_vals[0] == 0.0
    assert np.all(np.diff(f0_vals) >= 0)
    assert np.all(f0_vals <= 1.0)
