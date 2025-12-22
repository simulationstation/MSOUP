"""
MV-O1B pocket/intermittency test package.

This package implements a streaming, window-debiased analysis pipeline to
combine GNSS clock residuals and ground magnetometer data and test for
transient clustered signals consistent with domain-wall/clump traversals.
"""

__all__ = [
    "config",
    "io_igs",
    "io_mag",
    "preprocess",
    "windows",
    "candidates",
    "geometry",
    "stats",
    "nulls",
    "plots",
    "report",
    "run",
]

__version__ = "0.1.0"
