from __future__ import annotations

import dataclasses
import datetime as _dt
import pathlib
from dataclasses import dataclass, field
from typing import List, Optional

import yaml


DEFAULT_OMEGA_M0 = 0.315
DEFAULT_OMEGA_L0 = 0.685
DEFAULT_H_EARLY = 67.0


@dataclass
class F0Config:
    """Configuration block for f0 inference.

    The scan defaults are deliberately conservative to avoid runaway memory use
    on WSL2 environments, while still allowing users to request denser grids
    through the YAML file.
    """

    delta_m_min: float = 0.0
    delta_m_max: float = 2.0
    num_points: int = 400
    include_sn: bool = True
    include_bao: bool = True
    include_td: bool = True
    fb_reference: Optional[str] = None
    output_subdir: str = "msoup_f0_inference"
    max_rss_mb: int = 6000
    dominance_threshold: float = 0.5


@dataclass
class ProbeConfig:
    """Configuration for a single probe dataset."""

    name: str
    path: str
    kernel: str = "K1"
    type: str = "generic"
    is_local: bool = False
    sigma_column: Optional[str] = "sigma"
    z_column: str = "z"
    label: Optional[str] = None
    covariance_path: Optional[str] = None
    obs_column: Optional[str] = None
    observable_column: Optional[str] = None  # for BAO: which observable (DV/rd, etc.)


@dataclass
class FitConfig:
    """Configuration for the Delta_m fit."""

    h_local: float
    sigma_local: float
    local_probe: Optional[str] = None
    delta_m_bounds: tuple = (-6.0, 6.0)
    fit_sn_m: bool = False


@dataclass
class DistanceConfig:
    """Configuration for distance-mode calculations."""

    z_max: float = 2.0
    num_z_samples: int = 512
    speed_of_light: float = 299792.458  # km/s
    nuisance_M: float = 0.0
    # BAO sound horizon at drag epoch in Mpc (Planck 2018 fiducial, fixed, not fitted)
    rd_mpc: float = 147.09


@dataclass
class MsoupConfig:
    """Top-level configuration."""

    probes: List[ProbeConfig]
    fit: FitConfig
    omega_m0: float = DEFAULT_OMEGA_M0
    omega_L0: float = DEFAULT_OMEGA_L0
    h_early: float = DEFAULT_H_EARLY
    results_dir: pathlib.Path = pathlib.Path("results/msoup_purist_closure")
    data_dir: Optional[pathlib.Path] = None
    distance: DistanceConfig = field(default_factory=DistanceConfig)
    f0: F0Config = field(default_factory=F0Config)
    timestamp: str = field(
        default_factory=lambda: _dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    )

    def resolve_paths(self) -> "MsoupConfig":
        """Resolve probe paths relative to config location."""
        for probe in self.probes:
            probe.path = str(pathlib.Path(probe.path).expanduser())
        if self.data_dir is not None:
            self.data_dir = pathlib.Path(self.data_dir).expanduser()
        self.results_dir = pathlib.Path(self.results_dir)
        return self


def load_config(path: str | pathlib.Path) -> MsoupConfig:
    """Load configuration from YAML file."""
    config_path = pathlib.Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as f:
        cfg_dict = yaml.safe_load(f)

    probes = [ProbeConfig(**p) for p in cfg_dict["probes"]]
    fit_cfg = FitConfig(**cfg_dict["fit"])
    distance_cfg = DistanceConfig(**cfg_dict.get("distance", {}))
    f0_cfg = F0Config(**cfg_dict.get("f0", {}))
    msoup_cfg = MsoupConfig(
        probes=probes,
        fit=fit_cfg,
        omega_m0=cfg_dict.get("omega_m0", DEFAULT_OMEGA_M0),
        omega_L0=cfg_dict.get("omega_L0", DEFAULT_OMEGA_L0),
        h_early=cfg_dict.get("h_early", DEFAULT_H_EARLY),
        results_dir=pathlib.Path(cfg_dict.get("results_dir", "results/msoup_purist_closure")),
        data_dir=cfg_dict.get("data_dir"),
        distance=distance_cfg,
        f0=f0_cfg,
    )
    return msoup_cfg.resolve_paths()
