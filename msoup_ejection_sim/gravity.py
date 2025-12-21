"""Gravity proxy using FFT Poisson solve."""
from __future__ import annotations

import numpy as np
from numpy.fft import fft2, ifft2


class GravitySolver:
    def __init__(self, grid: int, eps: float = 1e-4):
        self.grid = grid
        self.eps = eps
        self.kx = np.fft.fftfreq(grid) * 2 * np.pi * grid
        self.ky = np.fft.fftfreq(grid) * 2 * np.pi * grid
        self.kx, self.ky = np.meshgrid(self.kx, self.ky, indexing="ij")
        self.k2 = self.kx ** 2 + self.ky ** 2

    def potential(self, rho_tot: np.ndarray) -> np.ndarray:
        rho_k = fft2(rho_tot)
        denom = self.k2 + self.eps
        phi_k = -rho_k / denom
        phi = np.real(ifft2(phi_k))
        return phi

    def binding(self, rho_tot: np.ndarray) -> np.ndarray:
        phi = self.potential(rho_tot)
        B = -phi
        B -= B.min()
        if B.max() > 0:
            B /= B.max()
        return B
