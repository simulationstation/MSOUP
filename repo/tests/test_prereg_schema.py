from pathlib import Path

import pytest

from bao_overlap.prereg import load_prereg


def test_load_prereg_top_level_keys():
    prereg = load_prereg(Path(__file__).parent.parent / "configs" / "preregistration.yaml")
    assert "analysis" not in prereg
    assert "primary_endpoint" in prereg


def test_load_prereg_missing_key(tmp_path):
    path = tmp_path / "prereg.yaml"
    path.write_text("version: '1.0.0'\nstatus: 'locked'\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_prereg(path)
