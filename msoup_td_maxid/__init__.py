"""
TD-only “maximally identifying” inference module.

This package focuses on robust, resource-aware time-delay inference with
dominance diagnostics. See `python -m msoup_td_maxid.run --help` for CLI usage.
"""

from .config import MaxIDConfig, PathsConfig, PriorConfig, ComputeConfig
from .inference import run_inference

__all__ = ["MaxIDConfig", "PathsConfig", "PriorConfig", "ComputeConfig", "run_inference"]
