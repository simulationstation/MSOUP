#!/usr/bin/env python3
"""
Timing Analysis Pipeline - Main Entry Script

Estimates rare-event leakage rate ε from distributed clock timing data
and tests for nonlocal coincidence patterns.

Target: ε ≈ 5.33e−5 per sample

Author: Claude Code
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_ingest import (
    download_igs_clock_files,
    parse_igs_clock_file,
    combine_clock_streams,
    compute_clock_residuals,
    get_stream_info
)
from src.event_detection import (
    detect_all_events,
    events_to_dataframe,
    compute_event_rate
)
from src.coincidence import (
    find_coincidences,
    monte_carlo_coincidence_test,
    analyze_coincidence_vs_window,
    compute_coincidence_significance,
    filter_known_disturbances,
    coincidences_to_dataframe
)
from src.visualization import generate_all_figures


# Configuration
CONFIG = {
    'seed': 42,
    'target_epsilon': 5.33e-5,

    # Data settings
    'sample_rate': '30S',  # IGS 30-second clocks
    'clock_type': 'AR',    # Receiver clocks (AR) vs satellite (AS)

    # Event detection thresholds (in sigma units)
    'phase_threshold': 5.0,
    'freq_threshold': 5.0,
    'min_segment_samples': 10,

    # Coincidence settings
    'default_window_seconds': 2.0,
    'min_streams_for_coincidence': 3,
    'coincidence_windows': [1.0, 2.0, 5.0, 30.0, 300.0],
    'n_monte_carlo': 10000,
}


def setup_directories(base_dir: Path) -> dict:
    """Create output directories."""
    dirs = {
        'data_ingest': base_dir / 'data_ingest',
        'events': base_dir / 'events',
        'figures': base_dir / 'figures',
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def generate_report_md(results: dict, output_path: Path):
    """Generate markdown report."""

    # Determine conclusion
    epsilon_hat = results.get('epsilon_hat', 0)
    epsilon_upper = results.get('epsilon_upper', 0)
    target = CONFIG['target_epsilon']
    p_value = results.get('coincidence_p_value', 1.0)

    if epsilon_hat > 0 and p_value < 0.01:
        conclusion = "DETECTED"
        conclusion_text = f"Significant coincident event rate detected at ε = {epsilon_hat:.2e}"
    elif epsilon_upper < target:
        conclusion = "NOT DETECTED"
        conclusion_text = f"Upper bound ε < {epsilon_upper:.2e} excludes target rate {target:.2e}"
    else:
        conclusion = "INCONCLUSIVE"
        conclusion_text = f"Data insufficient to confirm or exclude target rate {target:.2e}"

    report = f"""# Timing Analysis Report

## Rare-Event Leakage Rate Estimation

Generated: {datetime.now().isoformat()}

---

## Summary

**Conclusion: {conclusion}**

{conclusion_text}

---

## Data Sources

### Primary Dataset: IGS GNSS Clock Products
- Source: International GNSS Service (IGS)
- URL: https://igs.bkg.bund.de/root_ftp/IGS/products/
- Product: Final clock solutions (30-second sampling)
- Period: {results.get('date_range', 'N/A')}

### Coverage
- Number of clock streams: {results.get('n_streams', 'N/A')}
- Total samples: {results.get('n_samples', 'N/A'):,}
- Duration: {results.get('duration_hours', 'N/A'):.1f} hours

---

## Event Detection

### Method
- **Phase-step detection**: Piecewise-constant changepoint detection using PELT algorithm
  (ruptures library) with L2 cost function
- **Frequency-step detection**: Local slope estimation with changepoint in derivative
- **Threshold**: {CONFIG['phase_threshold']}σ (robust MAD scaling)
- **Minimum segment**: {CONFIG['min_segment_samples']} samples between changepoints

### Results
- Total events detected: {results.get('n_events', 0)}
- Phase steps: {results.get('n_phase_steps', 0)}
- Frequency steps: {results.get('n_freq_steps', 0)}
- Streams with events: {results.get('n_streams_with_events', 0)}

### Event Rate (ε)
- **ε̂ = {results.get('epsilon_hat', 0):.6e}** events per sample
- 95% CI: [{results.get('epsilon_lower', 0):.6e}, {results.get('epsilon_upper', 0):.6e}]
- Target ε: {target:.2e}

Ratio to target: {results.get('epsilon_hat', 0) / target:.2f}x

---

## Coincidence Analysis

### Method
- **Coincidence window**: {CONFIG['default_window_seconds']} seconds (also tested: {CONFIG['coincidence_windows']})
- **Minimum streams**: {CONFIG['min_streams_for_coincidence']} independent streams required
- **Null model**: Independent Poisson processes with empirical per-stream rates
- **Statistical test**: Monte Carlo permutation test ({CONFIG['n_monte_carlo']} iterations)
  - Permute event times within day while preserving daily rates

### Results
- Observed coincidences: {results.get('observed_coincidences', 0)}
- Expected (null): {results.get('expected_coincidences', 0):.2f}
- Excess: {results.get('excess_coincidences', 0):.2f}
- **p-value: {results.get('coincidence_p_value', 1.0):.4f}**

### Coincidence vs Window Size
| Window (s) | Observed | Expected | Excess | p-value |
|------------|----------|----------|--------|---------|
"""

    # Add window analysis results
    window_results = results.get('window_analysis', [])
    for row in window_results:
        report += f"| {row['window_seconds']:.1f} | {row['observed']} | {row['expected']:.2f} | {row['excess']:.2f} | {row['p_value']:.4f} |\n"

    report += f"""
---

## False Positive Control

### Exclusions Applied
- Leap second periods: ±1 day around known leap seconds excluded
- Same-site events: Events from clocks at same site not counted as independent

### Validation
- Inter-event time distribution tested against exponential (Poisson assumption)
- Per-stream event rates checked for uniformity

---

## Figures

1. `event_magnitude_distribution.png` - Distribution of event magnitudes
2. `inter_event_times.png` - Inter-event time histogram vs exponential
3. `coincidence_vs_window.png` - Coincidence rate vs window size
4. `epsilon_rolling.png` - Rolling event rate over time
5. `stream_events.png` - Time series with marked events
6. `null_distribution.png` - Monte Carlo null distribution

---

## Conclusion

**{conclusion}**

{conclusion_text}

### Interpretation

"""

    if conclusion == "DETECTED":
        report += """The analysis found statistically significant coincident timing events across
multiple independent clock streams. This pattern exceeds what would be expected from
independent random processes at the p < 0.01 level.
"""
    elif conclusion == "NOT DETECTED":
        report += f"""The estimated event rate ε = {results.get('epsilon_hat', 0):.2e} with upper
bound {results.get('epsilon_upper', 0):.2e} is below the target rate of {target:.2e}.
No significant coincidence pattern was detected.
"""
    else:
        report += f"""The analysis could not conclusively determine whether the target event rate
is present. The 95% confidence interval [{results.get('epsilon_lower', 0):.2e},
{results.get('epsilon_upper', 0):.2e}] includes the target value {target:.2e}.
Additional data would be needed for a definitive conclusion.
"""

    report += """
---

## Data Availability

All analysis inputs and outputs are saved in the pipeline directory:
- `data_ingest/`: Raw and parsed clock files
- `events/`: Detected event lists
- `figures/`: Generated plots
- `REPORT.json`: Machine-readable results

---

*Generated by Timing Analysis Pipeline*
"""

    with open(output_path, 'w') as f:
        f.write(report)

    return conclusion


def generate_report_json(results: dict, output_path: Path):
    """Generate JSON report."""
    # Convert any numpy types to Python native types
    def convert(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    json_results = convert(results)

    with open(output_path, 'w') as f:
        json.dump(json_results, f, indent=2)


def run_analysis(
    start_date: datetime,
    end_date: datetime,
    base_dir: Path,
    config: dict = None
) -> dict:
    """
    Run complete analysis pipeline.

    Args:
        start_date: Start date for data download
        end_date: End date for data download
        base_dir: Base directory for outputs
        config: Optional config override

    Returns:
        Results dictionary
    """
    if config is None:
        config = CONFIG

    np.random.seed(config['seed'])

    print("="*60)
    print("TIMING ANALYSIS PIPELINE")
    print(f"Target ε: {config['target_epsilon']:.2e}")
    print(f"Date range: {start_date.date()} to {end_date.date()}")
    print("="*60)

    # Setup directories
    base_dir = Path(base_dir)
    dirs = setup_directories(base_dir)

    results = {
        'analysis_timestamp': datetime.now().isoformat(),
        'config': config,
        'target_epsilon': config['target_epsilon'],
    }

    # =========================================================================
    # STEP 1: Data Ingestion
    # =========================================================================
    print("\n[1/5] Downloading and parsing clock data...")

    clock_files = download_igs_clock_files(
        start_date, end_date,
        dirs['data_ingest'],
        sample_rate=config['sample_rate']
    )

    if not clock_files:
        print("ERROR: No clock files downloaded. Check network access.")
        results['error'] = "No data downloaded"
        return results

    print(f"  Downloaded {len(clock_files)} files")

    # Parse all files
    all_clocks = {}
    for f in clock_files:
        clocks = parse_igs_clock_file(f)
        for name, df in clocks.items():
            if name in all_clocks:
                all_clocks[name] = pd.concat([all_clocks[name], df])
            else:
                all_clocks[name] = df

    # Sort and deduplicate
    for name in all_clocks:
        df = all_clocks[name]
        df = df.drop_duplicates(subset=['epoch']).sort_values('epoch').reset_index(drop=True)
        all_clocks[name] = df

    # Get stream info
    stream_info = get_stream_info(all_clocks)
    stream_info.to_csv(dirs['data_ingest'] / 'stream_info.csv', index=False)

    print(f"  Parsed {len(all_clocks)} unique clock streams")

    # Combine into matrix
    combined = combine_clock_streams(all_clocks, clock_type=config['clock_type'])
    print(f"  Combined matrix: {combined.shape[0]} epochs x {combined.shape[1]} streams")

    if combined.empty:
        print("ERROR: No receiver clocks found in data.")
        results['error'] = "No receiver clocks"
        return results

    # Compute residuals
    residuals = compute_clock_residuals(combined)
    residuals.to_csv(dirs['data_ingest'] / 'residuals.csv')

    # Store metadata
    results['date_range'] = f"{start_date.date()} to {end_date.date()}"
    results['n_streams'] = len(residuals.columns)
    results['n_samples'] = residuals.notna().sum().sum()
    duration = (residuals.index.max() - residuals.index.min()).total_seconds()
    results['duration_hours'] = duration / 3600

    # =========================================================================
    # STEP 2: Event Detection
    # =========================================================================
    print("\n[2/5] Detecting events...")

    events = detect_all_events(
        residuals,
        phase_threshold=config['phase_threshold'],
        freq_threshold=config['freq_threshold'],
        min_segment=config['min_segment_samples']
    )

    print(f"  Detected {len(events)} events")

    # Filter known disturbances
    events_filtered = filter_known_disturbances(events)
    print(f"  After filtering: {len(events_filtered)} events")

    # Save events
    events_df = events_to_dataframe(events_filtered)
    events_df.to_csv(dirs['events'] / 'all_events.csv', index=False)

    # Compute statistics
    n_phase = sum(1 for e in events_filtered if e.event_type == 'phase_step')
    n_freq = sum(1 for e in events_filtered if e.event_type == 'freq_step')
    streams_with_events = len(set(e.stream for e in events_filtered))

    results['n_events'] = len(events_filtered)
    results['n_phase_steps'] = n_phase
    results['n_freq_steps'] = n_freq
    results['n_streams_with_events'] = streams_with_events

    # =========================================================================
    # STEP 3: Compute Event Rate (ε)
    # =========================================================================
    print("\n[3/5] Computing event rate...")

    epsilon_hat, epsilon_lower, epsilon_upper = compute_event_rate(
        events_filtered, results['n_samples']
    )

    results['epsilon_hat'] = epsilon_hat
    results['epsilon_lower'] = epsilon_lower
    results['epsilon_upper'] = epsilon_upper

    print(f"  ε̂ = {epsilon_hat:.6e}")
    print(f"  95% CI: [{epsilon_lower:.6e}, {epsilon_upper:.6e}]")
    print(f"  Target: {config['target_epsilon']:.2e}")

    # =========================================================================
    # STEP 4: Coincidence Analysis
    # =========================================================================
    print("\n[4/5] Running coincidence analysis...")

    # Find coincidences at default window
    coincidences = find_coincidences(
        events_filtered,
        window_seconds=config['default_window_seconds'],
        min_streams=config['min_streams_for_coincidence']
    )

    print(f"  Found {len(coincidences)} coincident event groups (W={config['default_window_seconds']}s)")

    # Save coincidences
    coinc_df = coincidences_to_dataframe(coincidences)
    coinc_df.to_csv(dirs['events'] / 'coincidences.csv', index=False)

    # Monte Carlo test
    print(f"  Running Monte Carlo ({config['n_monte_carlo']} iterations)...")
    observed, expected, p_value, null_dist = monte_carlo_coincidence_test(
        events_filtered,
        window_seconds=config['default_window_seconds'],
        min_streams=config['min_streams_for_coincidence'],
        n_permutations=config['n_monte_carlo'],
        seed=config['seed']
    )

    results['observed_coincidences'] = observed
    results['expected_coincidences'] = expected
    results['excess_coincidences'] = observed - expected
    results['coincidence_p_value'] = p_value

    print(f"  Observed: {observed}, Expected: {expected:.2f}")
    print(f"  p-value: {p_value:.4f}")

    # Analyze across windows
    print("  Analyzing window sizes...")
    window_results = analyze_coincidence_vs_window(
        events_filtered,
        windows=config['coincidence_windows'],
        min_streams=config['min_streams_for_coincidence'],
        n_permutations=min(1000, config['n_monte_carlo']),  # Faster for sweep
        seed=config['seed']
    )
    window_results.to_csv(dirs['events'] / 'window_analysis.csv', index=False)
    results['window_analysis'] = window_results.to_dict('records')

    # =========================================================================
    # STEP 5: Generate Figures and Report
    # =========================================================================
    print("\n[5/5] Generating figures and report...")

    figure_paths = generate_all_figures(
        events_filtered,
        residuals,
        window_results,
        null_dist,
        observed,
        dirs['figures']
    )

    print(f"  Generated {len(figure_paths)} figures")

    # Generate reports
    conclusion = generate_report_md(results, base_dir / 'REPORT.md')
    generate_report_json(results, base_dir / 'REPORT.json')

    print("\n" + "="*60)
    print(f"ANALYSIS COMPLETE")
    print(f"Conclusion: {conclusion}")
    print(f"Reports: {base_dir}/REPORT.md, REPORT.json")
    print("="*60)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Timing Analysis Pipeline for Rare Event Detection'
    )
    parser.add_argument(
        '--start-date', type=str, default=None,
        help='Start date (YYYY-MM-DD). Default: 7 days ago'
    )
    parser.add_argument(
        '--end-date', type=str, default=None,
        help='End date (YYYY-MM-DD). Default: yesterday'
    )
    parser.add_argument(
        '--days', type=int, default=7,
        help='Number of days to analyze (if dates not specified)'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Output directory. Default: current directory'
    )

    args = parser.parse_args()

    # Set dates
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    else:
        end_date = datetime.now() - timedelta(days=1)

    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    else:
        start_date = end_date - timedelta(days=args.days - 1)

    # Set output directory
    if args.output_dir:
        base_dir = Path(args.output_dir)
    else:
        base_dir = Path(__file__).parent

    # Run analysis
    results = run_analysis(start_date, end_date, base_dir)

    return 0 if 'error' not in results else 1


if __name__ == '__main__':
    sys.exit(main())
