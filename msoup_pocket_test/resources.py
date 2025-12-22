"""
Resource monitoring and environment detection helpers.

These utilities enforce RSS limits to prevent host freezes and expose a
lightweight monitor that can be polled during long-running operations.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


def is_wsl() -> bool:
    """Detect if running under WSL2 by inspecting /proc/version."""
    try:
        with open("/proc/version", "r", encoding="utf-8") as f:
            return "microsoft" in f.read().lower()
    except OSError:
        return False


def get_rss_gb() -> float:
    """Return current process RSS in gigabytes."""
    try:
        import psutil

        rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        return rss_bytes / (1024 ** 3)
    except Exception:
        return 0.0


@dataclass
class ResourceMonitor:
    """Track peak RSS and enforce a hard ceiling via check calls."""

    max_rss_gb: float
    peak_rss_gb: float = 0.0

    def check(self, context: str = "") -> None:
        """Raise if RSS exceeds the configured ceiling."""
        current = get_rss_gb()
        self.peak_rss_gb = max(self.peak_rss_gb, current)
        if self.max_rss_gb > 0 and current > self.max_rss_gb:
            msg = (
                f"RSS {current:.2f} GB exceeded limit {self.max_rss_gb:.2f} GB"
                " — aborting to protect the host. "
                "Try reducing null resamples, setting max_workers=1, "
                "using binned pair mode, or increasing chunk_days."
            )
            if context:
                msg = f"[{context}] " + msg
            raise RuntimeError(msg)

    def snapshot(self) -> dict:
        return {"peak_rss_gb": self.peak_rss_gb, "max_rss_gb": self.max_rss_gb}

