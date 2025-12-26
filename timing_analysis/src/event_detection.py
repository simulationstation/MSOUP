"""
Event Detection Module for Timing Analysis Pipeline

Implements robust changepoint detection for clock timing data:
- Phase-step detection (sudden offset changes)
- Frequency-step detection (slope changes)

Uses ruptures library for changepoint detection with outlier robustness.

Author: Claude Code
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Event:
    """Represents a detected timing event."""
    epoch: pd.Timestamp
    stream: str
    event_type: str  # 'phase_step' or 'freq_step'
    magnitude: float  # size of the step (seconds for phase, s/s for freq)
    confidence: float  # statistical confidence (0-1)
    snr: float  # signal-to-noise ratio
    index: int  # sample index in stream


def robust_mad(x: np.ndarray) -> float:
    """Compute Median Absolute Deviation."""
    return np.median(np.abs(x - np.median(x))) * 1.4826


def detect_phase_steps_cusum(
    epochs: np.ndarray,
    values: np.ndarray,
    threshold_sigma: float = 5.0,
    min_segment_samples: int = 10
) -> List[Tuple[int, float, float]]:
    """
    Detect phase steps using CUSUM (cumulative sum) method.

    Args:
        epochs: Array of epoch timestamps (as floats, seconds from start)
        values: Array of clock bias values (seconds)
        threshold_sigma: Detection threshold in sigma units
        min_segment_samples: Minimum samples between changepoints

    Returns:
        List of (index, magnitude, snr) tuples for detected steps
    """
    if len(values) < 2 * min_segment_samples:
        return []

    # Compute first differences
    diff = np.diff(values)

    # Robust scale estimate
    scale = robust_mad(diff)
    if scale < 1e-15:
        scale = np.std(diff)
    if scale < 1e-15:
        return []

    # Normalized differences
    z = diff / scale

    # CUSUM statistics
    cusum_pos = np.zeros(len(diff))
    cusum_neg = np.zeros(len(diff))

    for i in range(len(diff)):
        cusum_pos[i] = max(0, cusum_pos[i-1] + z[i] - 0.5) if i > 0 else max(0, z[i] - 0.5)
        cusum_neg[i] = max(0, cusum_neg[i-1] - z[i] - 0.5) if i > 0 else max(0, -z[i] - 0.5)

    # Find peaks exceeding threshold
    events = []
    threshold = threshold_sigma

    # Detect positive jumps
    peaks_pos = np.where(cusum_pos > threshold)[0]
    # Detect negative jumps
    peaks_neg = np.where(cusum_neg > threshold)[0]

    # Combine and find actual step locations
    all_peaks = set(peaks_pos.tolist() + peaks_neg.tolist())

    for peak in sorted(all_peaks):
        if peak < min_segment_samples or peak >= len(diff) - min_segment_samples:
            continue

        # Estimate step magnitude
        before = values[max(0, peak - min_segment_samples):peak]
        after = values[peak + 1:min(len(values), peak + min_segment_samples + 1)]

        if len(before) < 3 or len(after) < 3:
            continue

        mag = np.median(after) - np.median(before)
        snr = abs(mag) / scale

        if snr > threshold_sigma:
            events.append((peak + 1, mag, snr))  # +1 because diff shifts index

    return events


def detect_phase_steps_ruptures(
    epochs: np.ndarray,
    values: np.ndarray,
    threshold_sigma: float = 5.0,
    min_segment_samples: int = 10,
    max_changepoints: int = 50
) -> List[Tuple[int, float, float]]:
    """
    Detect phase steps using ruptures library (piecewise constant model).
    """
    try:
        import ruptures as rpt
    except ImportError:
        # Fall back to CUSUM
        return detect_phase_steps_cusum(epochs, values, threshold_sigma, min_segment_samples)

    if len(values) < 2 * min_segment_samples:
        return []

    # Robust scale estimate
    diff = np.diff(values)
    scale = robust_mad(diff)
    if scale < 1e-15:
        scale = np.std(diff)
    if scale < 1e-15:
        return []

    # Use PELT algorithm with L2 cost
    signal = values.reshape(-1, 1)
    penalty = threshold_sigma ** 2 * scale ** 2 * len(values)

    try:
        algo = rpt.Pelt(model="l2", min_size=min_segment_samples).fit(signal)
        changepoints = algo.predict(pen=penalty)
    except Exception:
        return detect_phase_steps_cusum(epochs, values, threshold_sigma, min_segment_samples)

    # Filter out end point and compute magnitudes
    events = []
    changepoints = [c for c in changepoints if c < len(values)]

    for cp in changepoints:
        if cp < min_segment_samples or cp >= len(values) - min_segment_samples:
            continue

        before = values[max(0, cp - min_segment_samples):cp]
        after = values[cp:min(len(values), cp + min_segment_samples)]

        if len(before) < 3 or len(after) < 3:
            continue

        mag = np.median(after) - np.median(before)
        snr = abs(mag) / scale

        if snr > threshold_sigma:
            events.append((cp, mag, snr))

    return events


def detect_frequency_steps(
    epochs: np.ndarray,
    values: np.ndarray,
    threshold_sigma: float = 5.0,
    min_segment_samples: int = 20
) -> List[Tuple[int, float, float]]:
    """
    Detect frequency steps (changes in clock rate) using piecewise linear model.

    A frequency step appears as a change in the slope of the phase time series.
    """
    if len(values) < 2 * min_segment_samples:
        return []

    # Compute local slopes using moving window
    window = min_segment_samples

    slopes = np.full(len(values), np.nan)
    for i in range(window, len(values) - window):
        t_local = epochs[i - window:i + window] - epochs[i]
        y_local = values[i - window:i + window]
        valid = np.isfinite(y_local)
        if np.sum(valid) >= window:
            coeffs = np.polyfit(t_local[valid], y_local[valid], 1)
            slopes[i] = coeffs[0]

    # Find slope changes
    valid_slopes = slopes[np.isfinite(slopes)]
    if len(valid_slopes) < 10:
        return []

    slope_diff = np.diff(slopes)
    scale = robust_mad(slope_diff[np.isfinite(slope_diff)])
    if scale < 1e-20:
        scale = np.nanstd(slope_diff)
    if scale < 1e-20:
        return []

    events = []
    for i in range(window + 1, len(slopes) - window):
        if not np.isfinite(slopes[i]) or not np.isfinite(slopes[i-1]):
            continue

        delta_slope = slopes[i] - slopes[i-1]
        snr = abs(delta_slope) / scale

        if snr > threshold_sigma:
            events.append((i, delta_slope, snr))

    return events


def detect_events_in_stream(
    epochs: pd.DatetimeIndex,
    values: pd.Series,
    stream_name: str,
    phase_threshold: float = 5.0,
    freq_threshold: float = 5.0,
    min_segment: int = 10
) -> List[Event]:
    """
    Detect all events (phase and frequency steps) in a single clock stream.
    """
    # Drop NaN values
    valid = values.dropna()
    if len(valid) < 2 * min_segment:
        return []

    # Convert epochs to seconds from start
    t0 = valid.index[0]
    t_seconds = np.array([(t - t0).total_seconds() for t in valid.index])
    y = valid.values

    events = []

    # Detect phase steps
    phase_events = detect_phase_steps_ruptures(t_seconds, y, phase_threshold, min_segment)
    for idx, mag, snr in phase_events:
        if idx < len(valid.index):
            confidence = min(1.0, stats.norm.sf(snr) * 2)  # Two-tailed p-value
            confidence = 1.0 - confidence  # Convert to confidence

            events.append(Event(
                epoch=valid.index[idx],
                stream=stream_name,
                event_type='phase_step',
                magnitude=mag,
                confidence=confidence,
                snr=snr,
                index=idx
            ))

    # Detect frequency steps
    freq_events = detect_frequency_steps(t_seconds, y, freq_threshold, min_segment * 2)
    for idx, mag, snr in freq_events:
        if idx < len(valid.index):
            confidence = min(1.0, stats.norm.sf(snr) * 2)
            confidence = 1.0 - confidence

            events.append(Event(
                epoch=valid.index[idx],
                stream=stream_name,
                event_type='freq_step',
                magnitude=mag,
                confidence=confidence,
                snr=snr,
                index=idx
            ))

    return events


def detect_all_events(
    residuals: pd.DataFrame,
    phase_threshold: float = 5.0,
    freq_threshold: float = 5.0,
    min_segment: int = 10
) -> List[Event]:
    """
    Detect events across all clock streams.

    Args:
        residuals: DataFrame with epoch index, columns are clock names
        phase_threshold: Detection threshold for phase steps (sigma)
        freq_threshold: Detection threshold for frequency steps (sigma)
        min_segment: Minimum samples between events

    Returns:
        List of all detected events
    """
    all_events = []

    for col in residuals.columns:
        events = detect_events_in_stream(
            residuals.index,
            residuals[col],
            stream_name=col,
            phase_threshold=phase_threshold,
            freq_threshold=freq_threshold,
            min_segment=min_segment
        )
        all_events.extend(events)

    # Sort by epoch
    all_events.sort(key=lambda e: e.epoch)

    return all_events


def events_to_dataframe(events: List[Event]) -> pd.DataFrame:
    """Convert list of events to DataFrame."""
    if not events:
        return pd.DataFrame(columns=['epoch', 'stream', 'event_type', 'magnitude', 'confidence', 'snr', 'index'])

    return pd.DataFrame([
        {
            'epoch': e.epoch,
            'stream': e.stream,
            'event_type': e.event_type,
            'magnitude': e.magnitude,
            'confidence': e.confidence,
            'snr': e.snr,
            'index': e.index
        }
        for e in events
    ])


def compute_event_rate(
    events: List[Event],
    n_attempts: int,
    event_type: Optional[str] = None
) -> Tuple[float, float, float]:
    """
    Compute event rate (epsilon) with confidence interval.

    Args:
        events: List of detected events
        n_attempts: Total number of samples (attempts)
        event_type: Filter by event type (None = all)

    Returns:
        (epsilon_hat, epsilon_lower, epsilon_upper) using Wilson score interval
    """
    if event_type:
        filtered = [e for e in events if e.event_type == event_type]
    else:
        filtered = events

    n_events = len(filtered)

    if n_attempts == 0:
        return 0.0, 0.0, 0.0

    epsilon_hat = n_events / n_attempts

    # Wilson score interval for binomial proportion
    z = 1.96  # 95% CI
    n = n_attempts
    p_hat = epsilon_hat

    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2*n)) / denom
    spread = z * np.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom

    epsilon_lower = max(0, center - spread)
    epsilon_upper = min(1, center + spread)

    return epsilon_hat, epsilon_lower, epsilon_upper


if __name__ == "__main__":
    # Test event detection
    np.random.seed(42)

    # Create synthetic data with a phase step
    n = 1000
    t = np.arange(n) * 30.0  # 30-second samples

    # Base signal with linear drift
    y = 1e-9 * t + np.random.normal(0, 1e-10, n)

    # Add phase step at sample 500
    y[500:] += 5e-9

    epochs = pd.date_range('2024-01-01', periods=n, freq='30s')

    print("Testing phase step detection...")
    events = detect_phase_steps_ruptures(t, y, threshold_sigma=5.0)
    print(f"Detected {len(events)} phase steps")
    for idx, mag, snr in events:
        print(f"  Index {idx}: magnitude={mag:.2e}s, SNR={snr:.1f}")
