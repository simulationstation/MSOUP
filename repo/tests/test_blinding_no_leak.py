import json
from pathlib import Path

import pytest

from bao_overlap.blinding import BlindState
from bao_overlap.reporting import write_results


def test_write_results_blocks_forbidden_keys(tmp_path: Path):
    results = {"beta": 1.0, "sigma_beta": 0.1}
    blind_state = BlindState(unblind=False)
    with pytest.raises(ValueError):
        write_results(tmp_path / "results.json", results, blind_state)


def test_write_results_allows_safe_payload(tmp_path: Path):
    results = {"alpha_bins": [1.0, 1.1]}
    blind_state = BlindState(unblind=False)
    output = tmp_path / "results.json"
    write_results(output, results, blind_state)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["alpha_bins"] == [1.0, 1.1]
