from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

DEFAULT_DATA_ROOT = Path("data_ignored")
DEFAULT_POSTERIOR_DIR = DEFAULT_DATA_ROOT / "td_posteriors"
DEFAULT_MASTER = DEFAULT_DATA_ROOT / "td_master.csv"
DEFAULT_RESULTS = Path("results") / "td_maxid"


@dataclass
class PathsConfig:
    """Paths for inputs and outputs."""

    td_master: Path = DEFAULT_MASTER
    posterior_dir: Path = DEFAULT_POSTERIOR_DIR
    output_dir: Path = DEFAULT_RESULTS


@dataclass
class PriorConfig:
    """Priors and grids for the inference."""

    delta_m_min: float = -1.0
    delta_m_max: float = 3.0
    delta_m_points: int = 121
    tau0: float = 0.05
    student_df: float = 6.0
    sigma_floor_frac: float = 0.0
    mixture_eps: float = 0.1
    mixture_scale: float = 6.0


@dataclass
class ComputeConfig:
    """Compute and guardrail settings."""

    max_workers: int = 12
    max_rss_gb: float = 200.0
    rss_check_interval: int = 5
    bandwidth: Optional[float] = None
    seed: int = 1_234
    chunk_size: int = 10_000


@dataclass
class MaxIDConfig:
    """Top-level configuration for TD-only max-ID inference."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    priors: PriorConfig = field(default_factory=PriorConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    h_early: float = 67.0
    omega_m0: float = 0.315
    omega_L0: float = 0.685
    summary_only: bool = False
    posterior_column: str = "D_dt"

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "MaxIDConfig":
        paths = data.get("paths", {}) or {}
        priors = data.get("priors", {}) or {}
        compute = data.get("compute", {}) or {}
        return MaxIDConfig(
            paths=PathsConfig(
                td_master=Path(paths.get("td_master", PathsConfig.td_master)),
                posterior_dir=Path(paths.get("posterior_dir", PathsConfig.posterior_dir)),
                output_dir=Path(paths.get("output_dir", PathsConfig.output_dir)),
            ),
            priors=PriorConfig(
                delta_m_min=float(priors.get("delta_m_min", PriorConfig.delta_m_min)),
                delta_m_max=float(priors.get("delta_m_max", PriorConfig.delta_m_max)),
                delta_m_points=int(priors.get("delta_m_points", PriorConfig.delta_m_points)),
                tau0=float(priors.get("tau0", PriorConfig.tau0)),
                student_df=float(priors.get("student_df", PriorConfig.student_df)),
                sigma_floor_frac=float(priors.get("sigma_floor_frac", PriorConfig.sigma_floor_frac)),
                mixture_eps=float(priors.get("mixture_eps", PriorConfig.mixture_eps)),
                mixture_scale=float(priors.get("mixture_scale", PriorConfig.mixture_scale)),
            ),
            compute=ComputeConfig(
                max_workers=int(compute.get("max_workers", ComputeConfig.max_workers)),
                max_rss_gb=float(compute.get("max_rss_gb", ComputeConfig.max_rss_gb)),
                rss_check_interval=int(compute.get("rss_check_interval", ComputeConfig.rss_check_interval)),
                bandwidth=compute.get("bandwidth", ComputeConfig.bandwidth),
                seed=int(compute.get("seed", ComputeConfig.seed)),
                chunk_size=int(compute.get("chunk_size", ComputeConfig.chunk_size)),
            ),
            h_early=float(data.get("h_early", MaxIDConfig.h_early)),
            omega_m0=float(data.get("omega_m0", MaxIDConfig.omega_m0)),
            omega_L0=float(data.get("omega_L0", MaxIDConfig.omega_L0)),
            summary_only=bool(data.get("summary_only", False)),
            posterior_column=str(data.get("posterior_column", "D_dt")),
        )


def load_config(path: str | Path) -> MaxIDConfig:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return MaxIDConfig.from_dict(raw)
