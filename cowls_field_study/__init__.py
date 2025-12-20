"""
Field-level residual analysis for COWLS lenses.

This package implements a new pipeline that measures azimuthal residual
structure along the Einstein ring and evaluates whether short-scale
correlations exceed expectations from the arc-visibility window and
instrument noise.
"""

from importlib import metadata


def __version__() -> str:
    """Return the installed package version, if available."""
    try:
        return metadata.version("cowls_field_study")
    except metadata.PackageNotFoundError:
        return "0.0.0"


__all__ = ["__version__"]
