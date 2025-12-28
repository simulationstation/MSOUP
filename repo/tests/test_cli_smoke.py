import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "mverse_channel.cli", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "mverse-channel" in result.stdout.lower() or "simulate" in result.stdout.lower()


def test_cli_end_to_end_demo():
    if importlib.util.find_spec("qutip") is None:
        pytest.skip("qutip not available")
    result = subprocess.run(
        [sys.executable, "-m", "mverse_channel.cli", "end-to-end", "--preset", "demo"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert (Path("outputs") / "mverse_demo" / "data.npz").exists()
