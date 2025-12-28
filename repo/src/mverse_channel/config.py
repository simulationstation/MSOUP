"""Configuration models for the mverse channel pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


class ModulationConfig(BaseModel):
    enabled: bool = True
    delta_omega: float = 0.05
    omega_mod: float = 0.8

    def boundary_index(self, omega_a0: float) -> float:
        if not self.enabled:
            return 0.0
        return _clamp01(abs(self.delta_omega) / max(omega_a0, 1e-9))


class TopologyConfig(BaseModel):
    preset: Literal[
        "tree",
        "single_loop",
        "multi_loop",
        "linked_loops",
        "custom",
    ] = "tree"
    n_nodes: int = 4
    n_edges: int = 3
    cycle_count: int = 0
    max_cycles: int = 6


class MeasurementChainConfig(BaseModel):
    sample_rate: float = 50.0
    amp_noise: float = 0.2
    shared_crosstalk: float = 0.0
    digitize_bits: Optional[int] = None
    bandpass_low: Optional[float] = None
    bandpass_high: Optional[float] = None


class HiddenChannelConfig(BaseModel):
    enabled: bool = False
    phenomenology: Literal["additive", "parametric"] = "additive"
    epsilon0: float = 0.0
    epsilon_max: float = 0.2
    beta: float = 0.4
    gamma: float = 0.4
    eta: float = 0.4
    rho: float = 0.8
    tau_x: float = 1.5
    coherence_threshold: float = 0.5
    topology_threshold: float = 0.3
    boundary_threshold: float = 0.2
    threshold_width: float = 0.1


class SimulationConfig(BaseModel):
    seed: int = 123
    n_levels: int = 6
    duration: float = 6.0
    dt: float = 0.05
    omega_a: float = 5.0
    omega_b: float = 5.6
    coupling_j: float = 0.08
    kappa_a: float = 0.06
    kappa_b: float = 0.08
    nth_a: float = 0.05
    nth_b: float = 0.05
    drive_amp_a: float = 0.1
    drive_amp_b: float = 0.07
    drive_freq: float = 5.0
    drive_phase: float = 0.0
    kerr_a: float = 0.0
    kerr_b: float = 0.0
    coherence_mode: Literal["q_factor", "kappa"] = "q_factor"
    q_ref: float = 50.0
    kappa_ref: float = 0.2

    modulation: ModulationConfig = Field(default_factory=ModulationConfig)
    topology: TopologyConfig = Field(default_factory=TopologyConfig)
    measurement: MeasurementChainConfig = Field(default_factory=MeasurementChainConfig)
    hidden_channel: HiddenChannelConfig = Field(default_factory=HiddenChannelConfig)

    @field_validator("dt")
    @classmethod
    def _dt_positive(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("dt must be positive")
        return value

    def time_grid(self) -> tuple[list[float], int]:
        n_steps = int(self.duration / self.dt) + 1
        tlist = [i * self.dt for i in range(n_steps)]
        return tlist, n_steps


class ProtocolConfig(BaseModel):
    coherence_high: float = 0.9
    coherence_low: float = 0.2
    topology_high: str = "linked_loops"
    topology_low: str = "tree"
    boundary_high: float = 0.4
    boundary_low: float = 0.0


class InferenceConfig(BaseModel):
    metric_sigma: float = 0.2
    restarts: int = 3
    max_iter: int = 80


class ReportConfig(BaseModel):
    anomaly_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "coherence_peak": 0.4,
            "non_gaussian": 0.3,
            "memory": 0.3,
        }
    )


class RunConfig(BaseModel):
    simulation: SimulationConfig = Field(default_factory=SimulationConfig)
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)


class SweepConfig(BaseModel):
    base: RunConfig = Field(default_factory=RunConfig)
    epsilon_values: list[float] = Field(default_factory=lambda: [0.0, 0.05, 0.1, 0.2])
    tau_values: list[float] = Field(default_factory=lambda: [0.5, 1.5])
    rho_values: list[float] = Field(default_factory=lambda: [0.4, 0.8])
    replicates: int = 5


def load_config(path: str | Path, model: type[BaseModel] = RunConfig) -> BaseModel:
    """Load a config from JSON into the specified model."""
    config_path = Path(path)
    data = json.loads(config_path.read_text())
    return model.model_validate(data)
