"""
Resource monitoring and environment detection helpers.

These utilities enforce RSS limits to prevent host freezes and expose a
lightweight monitor that can be polled during long-running operations.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Optional


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


def get_system_memory_gb() -> float:
    """Return total system memory in gigabytes."""
    try:
        import psutil
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


def check_rss(max_rss_gb: float, context: str = "") -> float:
    """
    Check current RSS against limit. Raises RuntimeError if exceeded.

    This is a standalone function for use in tight loops where a monitor
    instance may not be convenient.

    Args:
        max_rss_gb: Maximum allowed RSS in gigabytes (0 = no limit)
        context: Description of current operation for error message

    Returns:
        Current RSS in GB

    Raises:
        RuntimeError: If RSS exceeds the limit
    """
    current = get_rss_gb()
    if max_rss_gb > 0 and current > max_rss_gb:
        msg = (
            f"RSS {current:.2f} GB exceeded limit {max_rss_gb:.2f} GB"
            " — aborting to protect the host.\n"
            "Suggested mitigations:\n"
            "  1. Reduce null_realizations or resamples\n"
            "  2. Use --mode sanity for validation runs\n"
            "  3. Set max_workers=1 (default)\n"
            "  4. Ensure pair_mode=binned (default)\n"
            "  5. Increase chunk_days to reduce concurrent data\n"
            "  6. Run with reduced max_*_files temporarily"
        )
        if context:
            msg = f"[{context}] " + msg
        raise MemoryLimitExceeded(msg)
    return current


class MemoryLimitExceeded(RuntimeError):
    """Raised when RSS memory limit is exceeded."""
    pass


class PipelineAborted(Exception):
    """Raised when pipeline is aborted due to resource constraints or other fatal errors."""
    def __init__(self, reason: str, last_chunk: Optional[str] = None, partial_results: Optional[dict] = None):
        self.reason = reason
        self.last_chunk = last_chunk
        self.partial_results = partial_results or {}
        super().__init__(reason)


@dataclass
class ResourceMonitor:
    """Track peak RSS and enforce a hard ceiling via check calls."""

    max_rss_gb: float
    rss_check_interval: int = 5  # Check every N operations
    peak_rss_gb: float = 0.0
    check_count: int = 0
    aborted: bool = False
    abort_reason: Optional[str] = None
    last_completed_chunk: Optional[str] = None

    # Tracking for partial results
    _partial_results: dict = field(default_factory=dict)

    def check(self, context: str = "") -> float:
        """
        Raise if RSS exceeds the configured ceiling.

        Args:
            context: Description of current operation

        Returns:
            Current RSS in GB

        Raises:
            MemoryLimitExceeded: If RSS exceeds limit
        """
        self.check_count += 1
        current = get_rss_gb()
        self.peak_rss_gb = max(self.peak_rss_gb, current)

        if self.max_rss_gb > 0 and current > self.max_rss_gb:
            self.aborted = True
            self.abort_reason = f"RSS {current:.2f} GB exceeded limit {self.max_rss_gb:.2f} GB at {context}"
            msg = (
                f"RSS {current:.2f} GB exceeded limit {self.max_rss_gb:.2f} GB"
                " — aborting to protect the host.\n"
                "Suggested mitigations:\n"
                "  1. Reduce null_realizations or resamples\n"
                "  2. Use --mode sanity for validation runs\n"
                "  3. Set max_workers=1 (default)\n"
                "  4. Ensure pair_mode=binned (default)\n"
                "  5. Increase chunk_days to reduce concurrent data\n"
                "  6. Run with reduced max_*_files temporarily"
            )
            if context:
                msg = f"[{context}] " + msg
            raise MemoryLimitExceeded(msg)
        return current

    def should_check(self, step: int) -> bool:
        """Return True if a memory check should be performed at this step."""
        return step % self.rss_check_interval == 0

    def mark_chunk_complete(self, chunk_id: str) -> None:
        """Record that a chunk was successfully processed."""
        self.last_completed_chunk = chunk_id

    def store_partial(self, key: str, value) -> None:
        """Store partial results for recovery."""
        self._partial_results[key] = value

    def snapshot(self) -> dict:
        """Return current resource state for reporting."""
        return {
            "peak_rss_gb": self.peak_rss_gb,
            "max_rss_gb": self.max_rss_gb,
            "check_count": self.check_count,
            "aborted": self.aborted,
            "abort_reason": self.abort_reason,
            "last_completed_chunk": self.last_completed_chunk,
        }

    def abort_report(self) -> dict:
        """Generate report data for an aborted run."""
        return {
            "status": "ABORTED",
            "reason": self.abort_reason or "Unknown",
            "last_completed_chunk": self.last_completed_chunk,
            "peak_rss_gb": self.peak_rss_gb,
            "partial_results": self._partial_results,
        }


def estimate_memory_requirements(
    n_files: int,
    n_sensors: int,
    n_resamples: int,
    chunk_hours: float,
    pair_mode: str,
) -> dict:
    """
    Estimate memory requirements for a run.

    Returns dict with estimates for each phase and recommendations.
    """
    # Rough estimates based on typical data
    bytes_per_clk_record = 100  # time, satellite, bias
    records_per_file = 10000  # rough estimate
    bytes_per_candidate = 200  # DataFrame row
    candidates_per_sensor_per_day = 50  # typical

    # File loading phase
    hours_per_file = 24  # typical CLK file spans 24 hours
    concurrent_files = max(1, int(chunk_hours / hours_per_file))
    file_load_mb = (concurrent_files * records_per_file * bytes_per_clk_record) / (1024 ** 2)

    # Candidate phase
    days = (n_files * hours_per_file) / 24
    total_candidates = n_sensors * candidates_per_sensor_per_day * days
    candidate_mb = (total_candidates * bytes_per_candidate) / (1024 ** 2)

    # Null resampling (streaming, so only one at a time)
    null_mb = candidate_mb  # One null catalog at a time

    # Clustering
    if pair_mode == "exact":
        # O(n log n) KDTree
        clustering_mb = total_candidates * 8 * 3 / (1024 ** 2)  # 3 floats per candidate
    else:
        # O(n) binned
        clustering_mb = total_candidates * 16 / (1024 ** 2)  # Just bin counts

    peak_estimate_mb = file_load_mb + candidate_mb + null_mb + clustering_mb
    peak_estimate_gb = peak_estimate_mb / 1024

    system_gb = get_system_memory_gb()

    return {
        "file_load_mb": file_load_mb,
        "candidate_mb": candidate_mb,
        "null_mb": null_mb,
        "clustering_mb": clustering_mb,
        "peak_estimate_gb": peak_estimate_gb,
        "system_memory_gb": system_gb,
        "safe": peak_estimate_gb < 0.7 * system_gb,
        "recommendations": _memory_recommendations(peak_estimate_gb, system_gb, pair_mode, n_resamples),
    }


def _memory_recommendations(peak_gb: float, system_gb: float, pair_mode: str, n_resamples: int) -> list:
    """Generate recommendations based on memory estimates."""
    recs = []

    if peak_gb > 0.8 * system_gb:
        recs.append("CRITICAL: Estimated memory usage exceeds safe threshold")

    if peak_gb > 0.5 * system_gb and n_resamples > 128:
        recs.append(f"Consider reducing null_realizations from {n_resamples} to 128")

    if peak_gb > 0.5 * system_gb and pair_mode == "exact":
        recs.append("Consider using pair_mode='binned' for O(n) memory scaling")

    if is_wsl() and system_gb < 16:
        recs.append("WSL2 detected with limited RAM - use conservative settings")

    return recs
