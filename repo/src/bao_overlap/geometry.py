"""Geometry utilities for coordinate transforms."""

from __future__ import annotations

import numpy as np
from astropy.cosmology import FlatLambdaCDM


def fiducial_cosmology(omega_m: float, h: float) -> FlatLambdaCDM:
    return FlatLambdaCDM(H0=100.0 * h, Om0=omega_m)


def radec_to_cartesian(ra: np.ndarray, dec: np.ndarray, z: np.ndarray, omega_m: float, h: float) -> np.ndarray:
    cosmo = fiducial_cosmology(omega_m, h)
    chi = cosmo.comoving_distance(z).value * h
    ra_rad = np.deg2rad(ra)
    dec_rad = np.deg2rad(dec)
    x = chi * np.cos(dec_rad) * np.cos(ra_rad)
    y = chi * np.cos(dec_rad) * np.sin(ra_rad)
    zc = chi * np.sin(dec_rad)
    return np.vstack([x, y, zc]).T
