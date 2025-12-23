"""Blinding utilities."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BlindState:
    unblind: bool


def require_unblind(state: BlindState) -> None:
    if not state.unblind:
        raise RuntimeError("Unblinding is not enabled. Set blinding.unblind=true to proceed.")
