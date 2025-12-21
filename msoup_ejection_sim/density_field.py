"""Density field utilities for the ejection simulation."""
from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2, fftshift


def smoothstep(x: float) -> float:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def make_fourier_field(rng: np.random.Generator, grid: int, spectral_index: float, kd: float) -> np.ndarray:
    kx = fftshift(np.fft.fftfreq(grid)) * grid
    ky = fftshift(np.fft.fftfreq(grid)) * grid
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k = np.sqrt(kx ** 2 + ky ** 2)
    amp = np.where(k == 0, 0.0, (k ** (spectral_index / 2.0)) * np.exp(-(k / kd) ** 2))
    noise = rng.normal(size=(grid, grid)) + 1j * rng.normal(size=(grid, grid))
    field_k = noise * amp
    field = np.real(ifft2(fftshift(field_k)))
    field /= np.std(field) + 1e-8
    return field


def lognormal_density(base_field: np.ndarray, sigma: float) -> np.ndarray:
    log_rho = base_field * sigma
    rho = np.exp(log_rho - 0.5 * sigma ** 2)
    return rho


def evolve_sigma(t: float, sigma0: float, sigmaF: float) -> float:
    return sigma0 + (sigmaF - sigma0) * smoothstep(t)


def gaussian_smooth(field: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return field
    grid = field.shape[0]
    kx = fftshift(np.fft.fftfreq(grid)) * grid
    ky = fftshift(np.fft.fftfreq(grid)) * grid
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    k2 = kx ** 2 + ky ** 2
    kernel = np.exp(-0.5 * sigma ** 2 * k2)
    field_k = fft2(field)
    return np.real(ifft2(field_k * kernel))
