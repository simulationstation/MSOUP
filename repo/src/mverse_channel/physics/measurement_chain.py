"""Measurement chain simulation."""

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
