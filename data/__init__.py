"""
Data acquisition and management for Msoup pipeline.

Modules:
- sparc: SPARC rotation curve data
- lensing: Strong lensing substructure constraints
"""

from .sparc import download_sparc, load_sparc_catalog, load_rotation_curve
from .lensing import LensingConstraints

__all__ = [
    "download_sparc",
    "load_sparc_catalog",
    "load_rotation_curve",
    "LensingConstraints",
]
