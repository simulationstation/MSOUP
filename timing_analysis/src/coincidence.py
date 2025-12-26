"""
Coincidence Analysis Module for Timing Analysis Pipeline

Tests whether detected events show nonlocal coincidence across
independent clock systems beyond what's expected by chance.

Uses Monte Carlo permutation tests with proper null model.

Author: Claude Code
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict

from .event_detection import Event


@dataclass
class CoincidentEvent:
    """A group of coincident events across multiple streams."""
    reference_epoch: pd.Timestamp
    window_seconds: float
    participating_streams: List[str]
    events: List[Event]
    n_streams: int


def find_coincidences(
    events: List[Event],
    window_seconds: float = 2.0,
    min_streams: int = 3,
    exclude_same_site: bool = True
) -> List[CoincidentEvent]:
    """
    Find coincident events across multiple independent streams.

    Args:
        events: List of detected events
        window_seconds: Coincidence window (symmetric)
        min_streams: Minimum number of streams with events to count as coincident
        exclude_same_site: If True, exclude events from same-site clocks

    Returns:
        List of coincident event groups
    """
    if not events:
        return []

    # Sort events by epoch
    sorted_events = sorted(events, key=lambda e: e.epoch)

    coincidences = []
    used = set()

    for i, ref_event in enumerate(sorted_events):
        if i in used:
            continue

        # Find all events within window
        ref_time = ref_event.epoch
        window = pd.Timedelta(seconds=window_seconds)

        group = [ref_event]
        group_streams = {ref_event.stream}
        group_indices = {i}

        for j, other_event in enumerate(sorted_events):
            if j == i or j in used:
                continue

            time_diff = abs((other_event.epoch - ref_time).total_seconds())

            if time_diff <= window_seconds:
                # Check if from different stream
                if other_event.stream != ref_event.stream:
                    # Optionally check same-site
                    if exclude_same_site:
                        # Same site if first 4 chars match (station naming convention)
                        if ref_event.stream[:4] != other_event.stream[:4]:
                            group.append(other_event)
                            group_streams.add(other_event.stream)
                            group_indices.add(j)
                    else:
                        group.append(other_event)
                        group_streams.add(other_event.stream)
                        group_indices.add(j)

        if len(group_streams) >= min_streams:
            coincidences.append(CoincidentEvent(
                reference_epoch=ref_time,
                window_seconds=window_seconds,
                participating_streams=list(group_streams),
                events=group,
                n_streams=len(group_streams)
            ))
            used.update(group_indices)

    return coincidences


def compute_expected_coincidences(
    events: List[Event],
    window_seconds: float,
    min_streams: int,
    duration_seconds: float,
    n_streams: int
) -> float:
    """
    Compute expected number of coincidences under null hypothesis
    (independent Poisson processes).

    The probability of K or more streams having an event in any window is:
    P(K+ coincidence) ≈ n_windows * P(single window has K+ events)

    where P(single window) is computed from the binomial with p = rate * window.
    """
    if n_streams < min_streams or duration_seconds <= 0:
        return 0.0

    # Compute per-stream rates
    stream_rates = defaultdict(int)
    for event in events:
        stream_rates[event.stream] += 1

    if len(stream_rates) < min_streams:
        return 0.0

    # Average event probability per window per stream
    total_rate = len(events) / duration_seconds  # events per second overall
    per_stream_rate = total_rate / len(stream_rates)  # average per stream
    p_event = per_stream_rate * window_seconds  # P(event in window for one stream)

    # Number of independent windows
    n_windows = duration_seconds / window_seconds

    # P(at least min_streams out of n_streams have event in same window)
    # Using binomial approximation
    p_coinc = 0.0
    for k in range(min_streams, n_streams + 1):
        p_coinc += stats.binom.pmf(k, n_streams, p_event)

    expected = n_windows * p_coinc

    return expected


def monte_carlo_coincidence_test(
    events: List[Event],
    window_seconds: float = 2.0,
    min_streams: int = 3,
    n_permutations: int = 10000,
    seed: int = 42
) -> Tuple[int, float, float, np.ndarray]:
    """
    Monte Carlo test for coincidence significance.

    Permutes event times within each day while preserving daily rates,
    then counts coincidences under null.

    Args:
        events: List of detected events
        window_seconds: Coincidence window
        min_streams: Minimum streams for coincidence
        n_permutations: Number of Monte Carlo iterations
        seed: Random seed for reproducibility

    Returns:
        (observed_count, expected_count, p_value, null_distribution)
    """
    rng = np.random.RandomState(seed)

    # Observed coincidences
    observed_coinc = find_coincidences(events, window_seconds, min_streams)
    observed_count = len(observed_coinc)

    if not events:
        return 0, 0.0, 1.0, np.array([0])

    # Get time range
    epochs = [e.epoch for e in events]
    t_min = min(epochs)
    t_max = max(epochs)
    duration = (t_max - t_min).total_seconds()

    if duration <= 0:
        return observed_count, 0.0, 1.0, np.array([0])

    # Group events by day and stream
    events_by_day_stream = defaultdict(list)
    for e in events:
        day = e.epoch.date()
        key = (day, e.stream)
        events_by_day_stream[key].append(e)

    # Get all unique day-stream combinations and their time ranges
    day_ranges = {}
    for (day, stream), evts in events_by_day_stream.items():
        times = [e.epoch for e in evts]
        day_start = pd.Timestamp(day)
        day_end = day_start + pd.Timedelta(days=1)
        day_ranges[(day, stream)] = (day_start, day_end)

    # Monte Carlo permutations
    null_counts = []

    for _ in range(n_permutations):
        # Permute event times within day for each stream
        permuted_events = []

        for (day, stream), evts in events_by_day_stream.items():
            day_start, day_end = day_ranges[(day, stream)]
            day_duration = (day_end - day_start).total_seconds()

            for e in evts:
                # Random time within the day
                random_offset = rng.uniform(0, day_duration)
                new_epoch = day_start + pd.Timedelta(seconds=random_offset)

                # Create new event with permuted time
                permuted_events.append(Event(
                    epoch=new_epoch,
                    stream=e.stream,
                    event_type=e.event_type,
                    magnitude=e.magnitude,
                    confidence=e.confidence,
                    snr=e.snr,
                    index=e.index
                ))

        # Count coincidences in permuted data
        perm_coinc = find_coincidences(permuted_events, window_seconds, min_streams)
        null_counts.append(len(perm_coinc))

    null_distribution = np.array(null_counts)
    expected_count = np.mean(null_distribution)

    # P-value: fraction of permutations with >= observed count
    p_value = np.mean(null_distribution >= observed_count)

    return observed_count, expected_count, p_value, null_distribution


def analyze_coincidence_vs_window(
    events: List[Event],
    windows: List[float] = [1.0, 2.0, 5.0, 30.0, 300.0],
    min_streams: int = 3,
    n_permutations: int = 1000,
    seed: int = 42
) -> pd.DataFrame:
    """
    Analyze coincidence rates across different window sizes.

    Returns DataFrame with columns:
    - window_seconds
    - observed
    - expected
    - excess
    - p_value
    """
    results = []

    for w in windows:
        obs, exp, pval, _ = monte_carlo_coincidence_test(
            events, w, min_streams, n_permutations, seed
        )
        results.append({
            'window_seconds': w,
            'observed': obs,
            'expected': exp,
            'excess': obs - exp,
            'p_value': pval
        })

    return pd.DataFrame(results)


def compute_coincidence_significance(
    observed: int,
    expected: float,
    null_distribution: np.ndarray
) -> Dict:
    """
    Compute various significance measures for coincidence test.
    """
    if expected <= 0:
        sigma = 0.0
    else:
        sigma = (observed - expected) / np.sqrt(expected)

    p_value = np.mean(null_distribution >= observed)

    # Also compute using Poisson approximation
    p_poisson = 1.0 - stats.poisson.cdf(observed - 1, expected) if expected > 0 else 1.0

    return {
        'observed': observed,
        'expected': expected,
        'excess': observed - expected,
        'sigma': sigma,
        'p_value_mc': p_value,
        'p_value_poisson': p_poisson,
        'null_mean': np.mean(null_distribution),
        'null_std': np.std(null_distribution),
        'null_max': np.max(null_distribution) if len(null_distribution) > 0 else 0
    }


def filter_known_disturbances(
    events: List[Event],
    leap_second_dates: Optional[List[pd.Timestamp]] = None,
    excluded_streams: Optional[List[str]] = None
) -> List[Event]:
    """
    Remove events during known disturbance periods.

    Args:
        events: List of events
        leap_second_dates: Dates of leap seconds to exclude
        excluded_streams: Streams to exclude entirely
    """
    filtered = []

    # Default leap seconds in recent years
    if leap_second_dates is None:
        leap_second_dates = [
            pd.Timestamp('2016-12-31'),
            pd.Timestamp('2015-06-30'),
        ]

    excluded_streams = set(excluded_streams or [])

    for e in events:
        # Check excluded streams
        if e.stream in excluded_streams:
            continue

        # Check leap second windows (exclude +/- 1 day)
        skip = False
        for ls_date in leap_second_dates:
            if abs((e.epoch - ls_date).days) <= 1:
                skip = True
                break

        if not skip:
            filtered.append(e)

    return filtered


def coincidences_to_dataframe(coincidences: List[CoincidentEvent]) -> pd.DataFrame:
    """Convert coincident events to DataFrame."""
    if not coincidences:
        return pd.DataFrame(columns=['reference_epoch', 'window_seconds', 'n_streams', 'streams'])

    return pd.DataFrame([
        {
            'reference_epoch': c.reference_epoch,
            'window_seconds': c.window_seconds,
            'n_streams': c.n_streams,
            'streams': ','.join(sorted(c.participating_streams))
        }
        for c in coincidences
    ])


if __name__ == "__main__":
    # Test coincidence detection
    np.random.seed(42)

    # Create synthetic events
    n_events = 100
    n_streams = 10
    duration_hours = 24

    events = []
    base_time = pd.Timestamp('2024-01-01')

    for i in range(n_events):
        stream = f"STREAM{i % n_streams:02d}"
        offset_hours = np.random.uniform(0, duration_hours)
        epoch = base_time + pd.Timedelta(hours=offset_hours)

        events.append(Event(
            epoch=epoch,
            stream=stream,
            event_type='phase_step',
            magnitude=1e-9,
            confidence=0.99,
            snr=5.0,
            index=i
        ))

    # Add some coincident events
    coinc_time = base_time + pd.Timedelta(hours=12)
    for i in range(5):
        events.append(Event(
            epoch=coinc_time + pd.Timedelta(seconds=np.random.uniform(-1, 1)),
            stream=f"COINC{i:02d}",
            event_type='phase_step',
            magnitude=1e-9,
            confidence=0.99,
            snr=5.0,
            index=100 + i
        ))

    print("Testing coincidence detection...")
    coincidences = find_coincidences(events, window_seconds=2.0, min_streams=3)
    print(f"Found {len(coincidences)} coincident event groups")

    print("\nTesting Monte Carlo significance...")
    obs, exp, pval, null = monte_carlo_coincidence_test(events, 2.0, 3, 1000)
    print(f"Observed: {obs}, Expected: {exp:.2f}, p-value: {pval:.4f}")
