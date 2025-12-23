from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional


def get_rss_gb() -> float:
    try:
        import psutil

        rss_bytes = psutil.Process(os.getpid()).memory_info().rss
        return rss_bytes / (1024 ** 3)
    except Exception:
        return 0.0


class MemoryLimitExceeded(RuntimeError):
    """Raised when RSS memory limit is exceeded."""


@dataclass
class ResourceMonitor:
    max_rss_gb: float
    rss_check_interval: int = 5
    peak_rss_gb: float = 0.0
    checks: int = 0
    aborted: bool = False
    abort_reason: Optional[str] = None
    partial_report: dict = field(default_factory=dict)

    def should_check(self, idx: int) -> bool:
        return idx % max(1, self.rss_check_interval) == 0

    def check(self, context: str = "") -> float:
        self.checks += 1
        current = get_rss_gb()
        self.peak_rss_gb = max(self.peak_rss_gb, current)
        if self.max_rss_gb > 0 and current > self.max_rss_gb:
            self.aborted = True
            self.abort_reason = f"RSS {current:.2f} GB exceeded limit {self.max_rss_gb:.2f} GB ({context})"
            raise MemoryLimitExceeded(self.abort_reason)
        return current

    def snapshot(self) -> dict:
        return {
            "max_rss_gb": self.max_rss_gb,
            "peak_rss_gb": self.peak_rss_gb,
            "checks": self.checks,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
        }
