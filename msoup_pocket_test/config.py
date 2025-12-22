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

from .nulls import NullType
@dataclass
class DataConfig:
    """Paths to GNSS and magnetometer products."""

    clk_glob: str = "data/igs/clk/*.clk"
    sp3_glob: str = "data/igs/sp3/*.sp3"
    magnetometer_glob: str = "data/magnetometer/*.csv"
    chunk_hours: float = 6.0
    chunk_days: float = 7.0
    cache_dir: str = "results/cache"
    # Memory-safety limits: 0 means no limit
    max_clk_files: int = 0
    max_mag_files: int = 0


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
    robustness_thresholds: Optional[list] = None


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
    resamples_sanity_default: int = 128
    resamples_full_default: int = 512
    block_length: int = 128
    random_seed: int = 13
    n_processes: int = 1
    enforce_determinism: bool = True
    allow_time_shift_null: bool = False
    max_empirical_pvalue: float = 1.0
    null_type: NullType = NullType.BLOCK_BOOTSTRAP
    robustness_null_variants: Optional[list] = None
    pair_mode: str = "binned"
    time_bin_seconds: int = 60
    max_rss_gb: float = 9.0
    max_candidates_in_memory: int = 750000
    max_workers: int = 1


@dataclass
class ResourceConfig:
    """Execution safety settings."""

    max_workers: int = 1
    chunk_days: float = 7.0
    resamples_sanity_default: int = 128
    resamples_full_default: int = 512
    max_rss_gb: float = 9.0
    pair_mode: str = "binned"
    time_bin_seconds: int = 60
    max_candidates_in_memory: int = 750000
    rss_check_interval_steps: int = 5  # Check RSS every N resamples/chunks


@dataclass
class GeometryConfig:
    """Propagation geometry settings."""

    enable_geometry_test: bool = True
    n_shuffles: int = 200
    min_sensors: int = 4
    speed_prior_kmps: float = 300.0
    min_events: int = 20


@dataclass
class OutputConfig:
    """Output locations."""

    results_dir: str = "results/msoup_pocket_test"
    report_name: str = "REPORT.md"
    cache_dir: str = "results/cache"


@dataclass
class RobustnessConfig:
    """Robustness sweep controls."""

    thresholds: Optional[list] = None
    mask_variants: Optional[list] = None
    null_variants: Optional[list] = None
    n_resamples: Optional[int] = None


@dataclass
class PocketConfig:
    """Top-level configuration container."""

    data: DataConfig = field(default_factory=DataConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    candidates: CandidateConfig = field(default_factory=CandidateConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)
    stats: StatsConfig = field(default_factory=StatsConfig)
    resources: ResourceConfig = field(default_factory=ResourceConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
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
                payload = data[key]
                if subcls is StatsConfig and "null_type" in payload and isinstance(payload["null_type"], str):
                    payload = dict(payload)
                    payload["null_type"] = NullType(payload["null_type"])
                return subcls(**payload)
            return subcls()

        resources = build(ResourceConfig, "resources")
        stats = build(StatsConfig, "stats")
        stats.pair_mode = getattr(stats, "pair_mode", resources.pair_mode)
        stats.time_bin_seconds = getattr(stats, "time_bin_seconds", resources.time_bin_seconds)
        stats.max_rss_gb = getattr(stats, "max_rss_gb", resources.max_rss_gb)
        stats.max_candidates_in_memory = getattr(
            stats, "max_candidates_in_memory", resources.max_candidates_in_memory
        )
        stats.resamples_sanity_default = getattr(
            stats, "resamples_sanity_default", resources.resamples_sanity_default
        )
        stats.resamples_full_default = getattr(
            stats, "resamples_full_default", resources.resamples_full_default
        )
        stats.n_processes = getattr(stats, "n_processes", resources.max_workers)

        data_cfg = build(DataConfig, "data")
        if hasattr(data_cfg, "chunk_days"):
            data_cfg.chunk_hours = data_cfg.chunk_days * 24

        return cls(
            data=data_cfg,
            preprocess=build(PreprocessConfig, "preprocess"),
            candidates=build(CandidateConfig, "candidates"),
            windows=build(WindowConfig, "windows"),
            stats=stats,
            resources=resources,
            geometry=build(GeometryConfig, "geometry"),
            output=build(OutputConfig, "output"),
            robustness=build(RobustnessConfig, "robustness"),
            run_label=data.get("run_label", "default"),
            fast_mode=data.get("fast_mode", False),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict suitable for YAML."""
        stats_dict = vars(self.stats).copy()
        if isinstance(stats_dict.get("null_type"), NullType):
            stats_dict["null_type"] = stats_dict["null_type"].value
        return {
            "data": vars(self.data),
            "preprocess": vars(self.preprocess),
            "candidates": vars(self.candidates),
            "windows": vars(self.windows),
            "stats": stats_dict,
            "resources": vars(self.resources),
            "geometry": vars(self.geometry),
            "output": vars(self.output),
            "robustness": vars(self.robustness),
            "run_label": self.run_label,
            "fast_mode": self.fast_mode,
        }

    def save(self, path: Path) -> None:
        """Persist configuration to YAML."""
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    def clone(self) -> "PocketConfig":
        """Return a deep copy of the configuration."""
        return PocketConfig.from_dict(self.to_dict())

    def adjusted_for_fast_mode(self) -> "PocketConfig":
        """
        Return a copy with reduced workload for sanity checks.

        Fast mode keeps statistical structure intact but lowers the number of
        null realizations and block lengths to shorten runtime. This avoids
        silent subsampling of sensors or time.
        """
        if not self.fast_mode:
            return self

        fast = self.clone()
        fast.stats.null_realizations = min(self.stats.null_realizations, 32)
        fast.stats.block_length = min(self.stats.block_length, 32)
        fast.stats.n_processes = min(self.stats.n_processes, 2)
        fast.resources.max_workers = min(self.resources.max_workers, 1)
        fast.stats.null_realizations = min(
            self.resources.resamples_sanity_default, self.stats.null_realizations
        )
        fast.run_label = f"{self.run_label}_fast"
        return fast


def load_config(path: Optional[str]) -> PocketConfig:
    """Load configuration, falling back to defaults when not provided."""
    if path is None:
        return PocketConfig()
    return PocketConfig.from_yaml(Path(path))
