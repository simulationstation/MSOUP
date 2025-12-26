"""
Visualization Module for Timing Analysis Pipeline

Generates plots for event analysis and coincidence testing.

Author: Claude Code
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional

from .event_detection import Event
from .coincidence import CoincidentEvent


def setup_plot_style():
    """Set up consistent plot style."""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['figure.dpi'] = 100


def plot_event_magnitude_distribution(
    events: List[Event],
    output_path: Path,
    event_type: Optional[str] = None
) -> Path:
    """
    Plot histogram of event magnitudes.
    """
    setup_plot_style()

    if event_type:
        filtered = [e for e in events if e.event_type == event_type]
        title_suffix = f" ({event_type})"
    else:
        filtered = events
        title_suffix = " (all types)"

    if not filtered:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No events detected', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Event Magnitude Distribution{title_suffix}')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    magnitudes = np.array([abs(e.magnitude) for e in filtered])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Linear scale
    ax1 = axes[0]
    ax1.hist(magnitudes, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Event Magnitude (seconds)')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Magnitude Distribution (Linear){title_suffix}')
    ax1.axvline(np.median(magnitudes), color='r', linestyle='--', label=f'Median: {np.median(magnitudes):.2e}')
    ax1.legend()

    # Log scale
    ax2 = axes[1]
    log_mags = np.log10(magnitudes[magnitudes > 0])
    if len(log_mags) > 0:
        ax2.hist(log_mags, bins=50, edgecolor='black', alpha=0.7)
        ax2.set_xlabel('log₁₀(Magnitude) [seconds]')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Magnitude Distribution (Log){title_suffix}')

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


def plot_inter_event_times(
    events: List[Event],
    output_path: Path
) -> Path:
    """
    Plot inter-event time distribution and compare to exponential.
    """
    setup_plot_style()

    if len(events) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient events for inter-event analysis', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    # Sort by time
    sorted_events = sorted(events, key=lambda e: e.epoch)
    epochs = [e.epoch for e in sorted_events]

    # Compute inter-event times in seconds
    inter_times = []
    for i in range(1, len(epochs)):
        dt = (epochs[i] - epochs[i-1]).total_seconds()
        inter_times.append(dt)

    inter_times = np.array(inter_times)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    ax1 = axes[0]
    ax1.hist(inter_times, bins=50, density=True, edgecolor='black', alpha=0.7, label='Observed')

    # Expected exponential
    mean_rate = 1.0 / np.mean(inter_times)
    x = np.linspace(0, np.percentile(inter_times, 99), 100)
    ax1.plot(x, mean_rate * np.exp(-mean_rate * x), 'r-', linewidth=2, label=f'Exponential (λ={mean_rate:.2e})')

    ax1.set_xlabel('Inter-event Time (seconds)')
    ax1.set_ylabel('Probability Density')
    ax1.set_title('Inter-event Time Distribution')
    ax1.legend()

    # Q-Q plot against exponential
    ax2 = axes[1]
    sorted_times = np.sort(inter_times)
    n = len(sorted_times)
    theoretical = -np.log(1 - (np.arange(1, n+1) - 0.5) / n) / mean_rate

    ax2.scatter(theoretical, sorted_times, alpha=0.5, s=10)
    max_val = max(theoretical.max(), sorted_times.max())
    ax2.plot([0, max_val], [0, max_val], 'r--', label='Perfect exponential')
    ax2.set_xlabel('Theoretical Quantiles (seconds)')
    ax2.set_ylabel('Sample Quantiles (seconds)')
    ax2.set_title('Q-Q Plot vs Exponential Distribution')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


def plot_coincidence_vs_window(
    results_df: pd.DataFrame,
    output_path: Path
) -> Path:
    """
    Plot coincidence counts vs window size.
    """
    setup_plot_style()

    if results_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No coincidence data', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Observed vs Expected
    ax1 = axes[0]
    windows = results_df['window_seconds'].values
    ax1.plot(windows, results_df['observed'], 'bo-', label='Observed', markersize=8)
    ax1.plot(windows, results_df['expected'], 'rs--', label='Expected (null)', markersize=8)
    ax1.fill_between(windows,
                     results_df['expected'] - 2*np.sqrt(results_df['expected']),
                     results_df['expected'] + 2*np.sqrt(results_df['expected']),
                     alpha=0.2, color='red', label='±2σ (Poisson)')
    ax1.set_xscale('log')
    ax1.set_xlabel('Coincidence Window (seconds)')
    ax1.set_ylabel('Number of Coincidences')
    ax1.set_title('Coincidences vs Window Size')
    ax1.legend()

    # P-values
    ax2 = axes[1]
    ax2.plot(windows, results_df['p_value'], 'go-', markersize=8)
    ax2.axhline(0.05, color='r', linestyle='--', label='α = 0.05')
    ax2.axhline(0.01, color='r', linestyle=':', label='α = 0.01')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xlabel('Coincidence Window (seconds)')
    ax2.set_ylabel('P-value (Monte Carlo)')
    ax2.set_title('Statistical Significance')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


def plot_epsilon_over_time(
    events: List[Event],
    residuals: pd.DataFrame,
    output_path: Path,
    window_days: float = 1.0
) -> Path:
    """
    Plot rolling estimate of epsilon (event rate) over time.
    """
    setup_plot_style()

    if not events or residuals.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Insufficient data for rolling epsilon', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    # Get time range
    event_epochs = [e.epoch for e in events]
    t_min = min(event_epochs)
    t_max = max(event_epochs)

    # Create time bins
    window = pd.Timedelta(days=window_days)
    bins = pd.date_range(t_min, t_max, freq=window)

    if len(bins) < 2:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'Time range too short for rolling analysis', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    # Count events and samples per bin
    epsilon_values = []
    epsilon_times = []

    for i in range(len(bins) - 1):
        bin_start = bins[i]
        bin_end = bins[i + 1]

        # Count events in bin
        n_events = sum(1 for e in events if bin_start <= e.epoch < bin_end)

        # Count samples in bin
        n_samples = ((residuals.index >= bin_start) & (residuals.index < bin_end)).sum() * len(residuals.columns)
        n_samples = n_samples - residuals.loc[bin_start:bin_end].isna().sum().sum()

        if n_samples > 0:
            epsilon = n_events / n_samples
            epsilon_values.append(epsilon)
            epsilon_times.append(bin_start + window/2)

    fig, ax = plt.subplots(figsize=(12, 5))

    if epsilon_values:
        ax.plot(epsilon_times, epsilon_values, 'b.-')
        ax.axhline(np.mean(epsilon_values), color='r', linestyle='--',
                   label=f'Mean ε = {np.mean(epsilon_values):.2e}')

        # Target epsilon for reference
        ax.axhline(5.33e-5, color='g', linestyle=':',
                   label=f'Target ε = 5.33e-5')

    ax.set_xlabel('Time')
    ax.set_ylabel('ε (events per sample)')
    ax.set_title(f'Rolling Event Rate (window = {window_days} days)')
    ax.legend()
    ax.set_yscale('log')

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


def plot_stream_events(
    events: List[Event],
    residuals: pd.DataFrame,
    output_path: Path,
    max_streams: int = 10
) -> Path:
    """
    Plot time series with detected events for top streams.
    """
    setup_plot_style()

    if not events or residuals.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No data to plot', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    # Count events per stream
    stream_counts = {}
    for e in events:
        stream_counts[e.stream] = stream_counts.get(e.stream, 0) + 1

    # Get top streams
    top_streams = sorted(stream_counts.keys(), key=lambda s: stream_counts[s], reverse=True)[:max_streams]
    top_streams = [s for s in top_streams if s in residuals.columns]

    if not top_streams:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No matching streams', ha='center', va='center', transform=ax.transAxes)
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    n_streams = len(top_streams)
    fig, axes = plt.subplots(n_streams, 1, figsize=(14, 2*n_streams), sharex=True)
    if n_streams == 1:
        axes = [axes]

    for i, stream in enumerate(top_streams):
        ax = axes[i]

        # Plot time series
        series = residuals[stream].dropna()
        ax.plot(series.index, series.values * 1e9, 'b-', linewidth=0.5, alpha=0.7)

        # Mark events
        stream_events = [e for e in events if e.stream == stream]
        for e in stream_events:
            ax.axvline(e.epoch, color='r', linewidth=1, alpha=0.7)

        ax.set_ylabel(f'{stream}\n(ns)')
        ax.text(0.02, 0.95, f'{len(stream_events)} events', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    axes[-1].set_xlabel('Time')
    plt.suptitle('Clock Residuals with Detected Events (red lines)', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


def plot_null_distribution(
    null_distribution: np.ndarray,
    observed: int,
    output_path: Path
) -> Path:
    """
    Plot null distribution from Monte Carlo with observed value marked.
    """
    setup_plot_style()

    fig, ax = plt.subplots()

    ax.hist(null_distribution, bins=50, density=True, edgecolor='black', alpha=0.7, label='Null distribution')
    ax.axvline(observed, color='r', linewidth=2, label=f'Observed = {observed}')
    ax.axvline(np.mean(null_distribution), color='g', linestyle='--',
               label=f'Expected = {np.mean(null_distribution):.1f}')

    p_value = np.mean(null_distribution >= observed)
    ax.set_xlabel('Number of Coincidences')
    ax.set_ylabel('Probability Density')
    ax.set_title(f'Monte Carlo Null Distribution (p = {p_value:.4f})')
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


def generate_all_figures(
    events: List[Event],
    residuals: pd.DataFrame,
    coincidence_results: pd.DataFrame,
    null_distribution: np.ndarray,
    observed_coincidences: int,
    figures_dir: Path
) -> Dict[str, Path]:
    """
    Generate all figures for the analysis.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Event magnitude distribution
    paths['magnitude_dist'] = plot_event_magnitude_distribution(
        events, figures_dir / 'event_magnitude_distribution.png'
    )

    # Inter-event times
    paths['inter_event'] = plot_inter_event_times(
        events, figures_dir / 'inter_event_times.png'
    )

    # Coincidence vs window
    paths['coinc_window'] = plot_coincidence_vs_window(
        coincidence_results, figures_dir / 'coincidence_vs_window.png'
    )

    # Epsilon over time
    paths['epsilon_rolling'] = plot_epsilon_over_time(
        events, residuals, figures_dir / 'epsilon_rolling.png'
    )

    # Stream events
    paths['stream_events'] = plot_stream_events(
        events, residuals, figures_dir / 'stream_events.png'
    )

    # Null distribution
    if len(null_distribution) > 0:
        paths['null_dist'] = plot_null_distribution(
            null_distribution, observed_coincidences, figures_dir / 'null_distribution.png'
        )

    return paths
