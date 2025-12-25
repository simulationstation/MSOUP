"""Run the decision harness and emit reports."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from msoup.harness import HarnessConfig, run_harness
from msoup.report import write_harness_report


def main() -> int:
    config = HarnessConfig()
    results = run_harness(config)
    output_dir = Path("outputs")
    write_harness_report(results.to_dict(), output_dir, executed=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
