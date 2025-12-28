"""Export helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict


def write_json(data: Dict, path: Path) -> None:
    path.write_text(json.dumps(data, indent=2))
