"""
Configuration handling for the MV-O1B pocket test.

The configuration is expressed as nested dataclasses and persisted as YAML to
keep provenance alongside results. Sensible defaults target a scientifically
meaningful run; the CLI can toggle a lighter-weight fast mode without
compromising correctness checks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass
class DataConfig:
    """Paths to GNSS and magnetometer products."""

    clk_glob: str = "data/igs/clk/*.clk"
    sp3_glob: str = "data/igs/sp3/*.sp3"
    magnetometer_glob: str = "data/magnetometer/*.csv"
    chunk_hours: float = 6.0
    cache_dir: str = "results/cache"


@dataclass
class PreprocessConfig:
    """Quality control and detrending choices."""

    detrend: str = "median"
    detrend_window_minutes: float = 60.0
    highpass_minutes: float = 5.0
    max_gap_factor: float = 5.0
    min_window_minutes: float = 15.0
    cadence_seconds: float = 30.0


@dataclass
class CandidateConfig:
    """Candidate extraction settings."""

    threshold_sigma: float = 4.0
    subthreshold_sigma: float = 2.0
    min_duration_seconds: float = 30.0
    neighbor_time_seconds: float = 120.0
    neighbor_angle_degrees: float = 25.0
    keep_subthreshold: bool = True
    max_candidates_per_sensor: int = 200000


@dataclass
class WindowConfig:
    """Per-sensor sensitivity windows."""

    cadence_tolerance: float = 1.5
    angular_response: bool = True
    mask_path: Optional[str] = None
    require_fractional_coverage: float = 0.6


@dataclass
class StatsConfig:
    """Statistical settings for nulls and summary statistics."""

    null_realizations: int = 256
    block_length: int = 128
    random_seed: int = 13
    n_processes: int = 4
    enforce_determinism: bool = True
    allow_time_shift_null: bool = False
    max_empirical_pvalue: float = 1.0


@dataclass
class GeometryConfig:
    """Propagation geometry settings."""

    enable_geometry_test: bool = True
    n_shuffles: int = 200
    min_sensors: int = 4
    speed_prior_kmps: float = 300.0


@dataclass
class OutputConfig:
    """Output locations."""

    results_dir: str = "results/msoup_pocket_test"
    report_name: str = "REPORT.md"
    cache_dir: str = "results/cache"


@dataclass
class PocketConfig:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    run_label: str = "default"
    fast_mode: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "PocketConfig":
        """Load configuration from YAML."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PocketConfig":
        """Construct config from nested dictionaries."""
        def build(subcls, key):
            if key in data:
                return subcls(**data[key])
            return subcls()

        return cls(
            data=build(DataConfig, "data"),
            preprocess=build(PreprocessConfig, "preprocess"),
            candidates=build(CandidateConfig, "candidates"),
            windows=build(WindowConfig, "windows"),
            stats=build(StatsConfig, "stats"),
            geometry=build(GeometryConfig, "geometry"),
            output=build(OutputConfig, "output"),
            run_label=data.get("run_label", "default"),
            fast_mode=data.get("fast_mode", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict suitable for YAML."""
        return {
            "data": vars(self.data),
            "preprocess": vars(self.preprocess),
            "candidates": vars(self.candidates),
            "windows": vars(self.windows),
            "stats": vars(self.stats),
            "geometry": vars(self.geometry),
            "output": vars(self.output),
            "run_label": self.run_label,
            "fast_mode": self.fast_mode,
        }

    def save(self, path: Path) -> None:
        """Persist configuration to YAML."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def adjusted_for_fast_mode(self) -> "PocketConfig":
        """
        Return a copy with reduced workload for sanity checks.

        Fast mode keeps statistical structure intact but lowers the number of
        null realizations and block lengths to shorten runtime. This avoids
        silent subsampling of sensors or time.
        """
        if not self.fast_mode:
            return self

        fast = PocketConfig.from_dict(self.to_dict())
        fast.stats.null_realizations = min(self.stats.null_realizations, 32)
        fast.stats.block_length = min(self.stats.block_length, 32)
        fast.stats.n_processes = min(self.stats.n_processes, 2)
        fast.run_label = f"{self.run_label}_fast"
        return fast


def load_config(path: Optional[str]) -> PocketConfig:
    """Load configuration, falling back to defaults when not provided."""
    if path is None:
        return PocketConfig()
    return PocketConfig.from_yaml(Path(path))
