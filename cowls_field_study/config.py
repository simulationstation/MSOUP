"""Configuration helpers for the COWLS field-level study."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Sequence

DEFAULT_DATA_ROOT = Path("data") / "jwst_cowls" / "repo"
DEFAULT_RESULTS_ROOT = Path("results") / "cowls_field_study"


def _default_bins() -> List[float]:
    return list(map(float, range(0, 360, 4)))


@dataclass
class FieldStudyConfig:
    """
    Pipeline configuration parameters.

    The defaults are conservative and tuned for a quick pilot run; callers may
    override individual fields via the CLI in :mod:`cowls_field_study.run`.
    """

    data_root: Path = DEFAULT_DATA_ROOT
    results_root: Path = DEFAULT_RESULTS_ROOT
    subset: Sequence[str] | None = None
    score_bins: Sequence[str] | None = None
    band: str = "auto"
    prefer_model_residuals: bool = True
    snr_threshold: float = 1.5
    annulus_width: float = 0.5
    theta_bins: List[float] = field(default_factory=_default_bins)
    lag_max: int = 6
    hf_fraction: float = 0.35
    null_mode: str = "both"
    null_draws: int = 300
    G_resample_global: int = 1000
    G_shift_global: int = 500
    allow_reduction: bool = False
    max_workers: int | None = None
    per_lens_null_B: int | None = None
    m_cut: int = 3
    fourier_m_max: int | None = None
    band_common_mask: str | None = None
    band_pair: str = "F150W,F277W"

    def theta_bin_edges(self) -> List[float]:
        """Return monotonically increasing theta bin edges in degrees."""
        if sorted(self.theta_bins) != list(self.theta_bins):
            return sorted(self.theta_bins)
        return list(self.theta_bins)

    @staticmethod
    def parse_subset(values: Iterable[str] | None) -> Sequence[str] | None:
        """Normalize CLI subset input."""
        if values is None:
            return None
        if isinstance(values, str):
            return [v.strip() for v in values.split(",") if v.strip()]
        return list(values)
