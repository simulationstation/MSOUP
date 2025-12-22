import json
import pathlib

import numpy as np
import pandas as pd

from msoup_purist_closure.config import DistanceConfig, F0Config, FitConfig, MsoupConfig, ProbeConfig
from msoup_purist_closure.infer_f0 import _combine_inverse_variance, _z_score_consistency, run_inference
from msoup_purist_closure.observables import bao_predict, distance_modulus


def _basic_config(tmp_path: pathlib.Path) -> MsoupConfig:
    sn_path = tmp_path / "sn.csv"
    z = np.array([0.01, 0.05, 0.1])
    mu = distance_modulus(z, delta_m=0.0, h_early=67.0)
    pd.DataFrame({"z": z, "mu_obs": mu, "sigma": 0.1}).to_csv(sn_path, index=False)

    fb_path = tmp_path / "fb.csv"
    pd.DataFrame({"fb_mean": [0.01], "fb_sigma": [0.02]}).to_csv(fb_path, index=False)

    probes = [
        ProbeConfig(
            name="sn_mock",
            path=str(sn_path),
            type="sn",
            z_column="z",
            sigma_column="sigma",
            obs_column="mu_obs",
        )
    ]
    fit_cfg = FitConfig(h_local=73.0, sigma_local=1.0)
    distance_cfg = DistanceConfig(nuisance_M=0.0)
    f0_cfg = F0Config(
        delta_m_min=0.0,
        delta_m_max=0.5,
        num_points=50,
        include_sn=True,
        include_bao=False,
        include_td=False,
        fb_reference=str(fb_path),
        output_subdir="test_out",
    )
    return MsoupConfig(
        probes=probes,
        fit=fit_cfg,
        distance=distance_cfg,
        f0=f0_cfg,
        results_dir=tmp_path / "results",
    )


def test_infer_f0_returns_finite_results(tmp_path):
    cfg = _basic_config(tmp_path)
    output_dir = cfg.results_dir / "run"
    output_dir.mkdir(parents=True, exist_ok=True)

    results = run_inference(cfg, output_dir)
    sn_summary = results["summary"]["SN"]

    assert np.isfinite(sn_summary["f0_hat"])
    assert np.isfinite(sn_summary["sigma_f0"])
    summary_json = json.loads((output_dir / "summary.json").read_text())
    assert "subsets" in summary_json


def test_dominance_check_flags_large_weight():
    estimates = {"A": (0.1, 0.1), "B": (0.2, 1.0)}
    combined, filtered, dominance = _combine_inverse_variance(estimates, dominance_threshold=0.5)
    assert dominance["A"] > 0.5
    assert combined[1] < filtered[1] or np.isclose(combined[1], filtered[1])


def test_z_score_uses_combined_uncertainty():
    z, p = _z_score_consistency(0.2, 0.1, 0.1, 0.1)
    expected_sigma = np.sqrt(0.1 ** 2 + 0.1 ** 2)
    assert np.isclose(z, (0.2 - 0.1) / expected_sigma)
    assert 0 < p < 1


def test_bao_regression_delta_m_zero(tmp_path):
    bao_path = tmp_path / "bao.csv"
    rd = 147.09
    rows = []
    for z in [0.3, 0.6]:
        val = bao_predict(z, "DV/rd", delta_m=0.0, rd_mpc=rd, h_early=67.0, omega_m0=0.315, omega_L0=0.685)
        rows.append({"z": z, "observable": "DV/rd", "value": val, "sigma": 0.05 * val})
    pd.DataFrame(rows).to_csv(bao_path, index=False)

    fb_path = tmp_path / "fb.csv"
    pd.DataFrame({"fb_mean": [0.0], "fb_sigma": [0.1]}).to_csv(fb_path, index=False)

    probes = [
        ProbeConfig(name="bao_mock", path=str(bao_path), type="bao", z_column="z", sigma_column="sigma", obs_column="value", observable_column="observable")
    ]
    fit_cfg = FitConfig(h_local=73.0, sigma_local=1.0)
    f0_cfg = F0Config(
        delta_m_min=0.0,
        delta_m_max=0.2,
        num_points=40,
        include_sn=False,
        include_bao=True,
        include_td=False,
        fb_reference=str(fb_path),
        output_subdir="test_out",
    )
    cfg = MsoupConfig(
        probes=probes,
        fit=fit_cfg,
        f0=f0_cfg,
        distance=DistanceConfig(),
        results_dir=tmp_path / "results",
    )
    output_dir = cfg.results_dir / "run"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = run_inference(cfg, output_dir)
    bao_summary = results["summary"]["BAO"]
    assert abs(bao_summary["delta_m_hat"]) < 1e-2
