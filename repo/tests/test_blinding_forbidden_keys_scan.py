import json
from pathlib import Path

import pytest

from bao_overlap.reporting import scan_forbidden_keys_in_dir


def test_forbidden_keys_scan_blocks_beta(tmp_path: Path):
    payload = {"beta": 1.23}
    output = tmp_path / "results.json"
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError):
        scan_forbidden_keys_in_dir(tmp_path)
