import pathlib

import pandas as pd

from msoup_purist_closure.config import FitConfig, MsoupConfig, ProbeConfig
from msoup_purist_closure.fit import fit_delta_m


def test_fit_delta_m_returns_value(tmp_path):
    data_path = tmp_path / "probe.csv"
    pd.DataFrame({"z": [0.01, 0.02, 0.03], "sigma": [0.1, 0.1, 0.1]}).to_csv(data_path, index=False)

    cfg = MsoupConfig(
        probes=[ProbeConfig(name="local", path=str(data_path), kernel="K1", is_local=True)],
        fit=FitConfig(h_local=73.0, sigma_local=1.0),
    )
    delta_m_star, probe_results = fit_delta_m(cfg)
    assert "local" in probe_results
    assert abs(delta_m_star) < 10
