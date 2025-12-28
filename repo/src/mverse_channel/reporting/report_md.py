"""Markdown report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict


def _format_value(v) -> str:
    """Format a metric value for display."""
    if isinstance(v, float):
        return f"{v:.3f}"
    return str(v)


def generate_report(
    config_json: str,
    metrics: Dict[str, float],
    figures: Dict[str, str],
    out_path: Path,
) -> None:
    summary = "\n".join(f"- **{k}**: {_format_value(v)}" for k, v in metrics.items())
    fig_md = "\n".join(f"![{name}]({path})" for name, path in figures.items())
    report = f"""# Mverse Channel Report

## Configuration
```json
{config_json}
```

## Metrics Summary
{summary}

## Figures
{fig_md}
"""
    out_path.write_text(report)
