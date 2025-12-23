import numpy as np
import pandas as pd

from bao_overlap.io import load_catalog


def test_load_catalog_parquet(tmp_path):
    data = pd.DataFrame({
        "RA": [0.0, 1.0],
        "DEC": [0.0, 1.0],
        "Z": [0.5, 0.6],
        "WEIGHT_FKP": [1.0, 2.0],
        "WEIGHT_SYSTOT": [1.0, 1.0],
        "WEIGHT_CP": [1.0, 1.0],
    })
    rand = pd.DataFrame({
        "RA": [0.1, 1.1],
        "DEC": [0.1, 1.1],
        "Z": [0.55, 0.65],
        "WEIGHT_FKP": [1.0, 1.0],
        "WEIGHT_SYSTOT": [1.0, 1.0],
        "WEIGHT_CP": [1.0, 1.0],
    })
    data_path = tmp_path / "data.parquet"
    rand_path = tmp_path / "rand.parquet"
    data.to_parquet(data_path)
    rand.to_parquet(rand_path)

    datasets_cfg = {
        "catalogs": {
            "test": {
                "data": {"path": str(data_path), "format": "parquet"},
                "randoms": {"path": str(rand_path), "format": "parquet"},
                "weights": {"total": "WEIGHT_FKP * WEIGHT_SYSTOT * WEIGHT_CP"},
            }
        }
    }

    data_cat, rand_cat = load_catalog(datasets_cfg, "test")
    assert data_cat.ra.shape[0] == 2
    assert rand_cat.w.shape[0] == 2
    assert np.isclose(data_cat.w[0], 1.0)
