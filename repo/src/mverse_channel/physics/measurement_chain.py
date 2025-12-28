"""Measurement chain simulation.

Includes standard amplifier noise and optional nuisance models for
shared noise sources that can create spurious correlations.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy import signal

from mverse_channel.config import MeasurementChainConfig


def _quantize(values: np.ndarray, bits: int) -> np.ndarray:
    if bits <= 0:
        return values
    scale = 2 ** (bits - 1) - 1
    clipped = np.clip(values, -1.0, 1.0)
    return np.round(clipped * scale) / scale


def apply_measurement_chain(
    ia: np.ndarray,
    qa: np.ndarray,
    ib: np.ndarray,
    qb: np.ndarray,
    config: MeasurementChainConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply amplifier noise, cross-talk, optional digitization and filtering."""
    noise_a = rng.normal(0.0, config.amp_noise, size=ia.shape)
    noise_b = rng.normal(0.0, config.amp_noise, size=ib.shape)
    ia = ia + noise_a
    qa = qa + rng.normal(0.0, config.amp_noise, size=qa.shape)
    ib = ib + noise_b
    qb = qb + rng.normal(0.0, config.amp_noise, size=qb.shape)

    if config.shared_crosstalk != 0.0:
        ia = ia + config.shared_crosstalk * noise_b
        ib = ib + config.shared_crosstalk * noise_a

    if config.digitize_bits:
        ia = _quantize(ia, config.digitize_bits)
        qa = _quantize(qa, config.digitize_bits)
        ib = _quantize(ib, config.digitize_bits)
        qb = _quantize(qb, config.digitize_bits)

    if config.bandpass_low and config.bandpass_high:
        nyq = 0.5 * config.sample_rate
        low = config.bandpass_low / nyq
        high = config.bandpass_high / nyq
        b, a = signal.butter(2, [low, high], btype="band")
        ia = signal.filtfilt(b, a, ia)
        qa = signal.filtfilt(b, a, qa)
        ib = signal.filtfilt(b, a, ib)
        qb = signal.filtfilt(b, a, qb)

    return ia, qa, ib, qb


def apply_nuisance_model(
    ia: np.ndarray,
    qa: np.ndarray,
    ib: np.ndarray,
    qb: np.ndarray,
    nuisance_amplitude: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply nuisance-only correlation model (shared noise without gating).

    This model adds correlated noise to both channels without any gating by
    coherence, topology, or boundary conditions. It serves as an alternative
    hypothesis to distinguish genuine hidden-channel effects from simple
    amplifier cross-talk or shared environmental noise.

    Args:
        ia, qa, ib, qb: Input quadrature signals
        nuisance_amplitude: Amplitude of shared nuisance noise
        rng: Random generator

    Returns:
        Modified signals with shared nuisance noise added
    """
    if nuisance_amplitude <= 0.0:
        return ia, qa, ib, qb

    # Shared noise source affects both channels equally (no gating)
    shared_noise = rng.normal(0.0, nuisance_amplitude, size=ia.shape)

    ia = ia + shared_noise
    qa = qa + shared_noise
    ib = ib + shared_noise
    qb = qb + shared_noise

    return ia, qa, ib, qb


def apply_chain_swap(
    ia: np.ndarray,
    qa: np.ndarray,
    ib: np.ndarray,
    qb: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Swap readout chains A and B (null test diagnostic).

    Under the null hypothesis (no hidden channel), swapping chains should
    not change detection statistics. Under the alternative (hidden channel),
    swapping may affect results if the channel is asymmetric.

    Returns:
        Swapped signals: (ib, qb, ia, qa)
    """
    return ib.copy(), qb.copy(), ia.copy(), qa.copy()
