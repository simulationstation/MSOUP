"""CLI for mverse-channel simulations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import typer

from mverse_channel.config import RunConfig, SimulationConfig, SweepConfig, load_config
from mverse_channel.inference.fitting import fit_baseline, fit_extended
from mverse_channel.inference.model_selection import aic, bic
from mverse_channel.reporting.figures import (
    plot_anomaly_scores,
    plot_coherence,
    plot_detection_curve,
)
from mverse_channel.reporting.report_md import generate_report
from mverse_channel.sim.generate_data import generate_and_save, generate_time_series
from mverse_channel.sim.run_sim import simulate_protocol
from mverse_channel.sim.sweeps import power_sweep
from mverse_channel.rng import get_rng

app = typer.Typer(add_completion=False)


@app.command()
def simulate(config: Path, out: Path) -> None:
    """Run a single simulation and save outputs."""
    run_config = load_config(config, RunConfig)
    generate_and_save(run_config.simulation, out)


@app.command()
def fit(data: Path, out: Path) -> None:
    """Fit baseline and extended models to saved metrics."""
    out.mkdir(parents=True, exist_ok=True)
    metrics = json.loads((data.parent / "metrics.json").read_text())
    sim_config = SimulationConfig.model_validate(json.loads((data.parent / "config.json").read_text()))
    run_config = RunConfig(simulation=sim_config)
    rng = get_rng(sim_config.seed)
    baseline_fit = fit_baseline(metrics, sim_config, run_config.inference, rng)
    extended_fit = fit_extended(metrics, sim_config, run_config.inference, rng)
    n_samples = int(np.load(data).get("n_steps", np.array([100]))[0])
    fit_result = {
        "baseline": {
            "params": baseline_fit.params,
            "loglike": baseline_fit.loglike,
            "aic": aic(baseline_fit.loglike, len(baseline_fit.params)),
            "bic": bic(baseline_fit.loglike, len(baseline_fit.params), n_samples),
        },
        "extended": {
            "params": extended_fit.params,
            "loglike": extended_fit.loglike,
            "aic": aic(extended_fit.loglike, len(extended_fit.params)),
            "bic": bic(extended_fit.loglike, len(extended_fit.params), n_samples),
        },
    }
    (out / "fit.json").write_text(json.dumps(fit_result, indent=2))


@app.command()
def report(data_dir: Path, out: Path) -> None:
    """Generate a markdown report with figures."""
    config_json = (data_dir / "config.json").read_text()
    metrics = json.loads((data_dir / "metrics.json").read_text())
    sim_config = SimulationConfig.model_validate_json(config_json)
    series = dict(np.load(data_dir / "data.npz"))
    out.parent.mkdir(parents=True, exist_ok=True)
    figures_dir = data_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_coherence(series, sim_config.measurement.sample_rate, figures_dir / "coherence.png")
    plot_anomaly_scores({"run": metrics.get("anomaly_score", 0.0)}, figures_dir / "anomaly.png")
    generate_report(
        config_json,
        metrics,
        {"coherence": "figures/coherence.png", "anomaly": "figures/anomaly.png"},
        out,
    )


@app.command()
def sweep(config: Path, out: Path) -> None:
    """Run a power sweep."""
    sweep_config = load_config(config, SweepConfig)
    results = power_sweep(sweep_config, out)
    plot_detection_curve(results, out / "detection_curve.png")


@app.command("end-to-end")
def end_to_end(preset: str = "demo") -> None:
    """Run an end-to-end demo pipeline."""
    if preset != "demo":
        raise typer.BadParameter("Only the demo preset is supported.")
    run_config = RunConfig()
    run_config.simulation.n_levels = 4
    run_config.simulation.duration = 3.0
    run_config.simulation.dt = 0.1
    run_config.simulation.hidden_channel.enabled = True
    run_config.simulation.hidden_channel.epsilon_max = 0.15
    out_dir = Path("outputs/mverse_demo")
    metrics = generate_and_save(run_config.simulation, out_dir)
    series = generate_time_series(run_config.simulation, get_rng(run_config.simulation.seed))
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_coherence(series, run_config.simulation.measurement.sample_rate, figures_dir / "coherence.png")
    plot_anomaly_scores({"demo": metrics.get("anomaly_score", 0.0)}, figures_dir / "anomaly.png")
    generate_report(
        run_config.simulation.model_dump_json(indent=2),
        metrics,
        {"coherence": "figures/coherence.png", "anomaly": "figures/anomaly.png"},
        out_dir / "report.md",
    )


@app.command()
def protocol(out: Path) -> None:
    """Run the experiment design protocol matrix."""
    run_config = RunConfig()
    simulate_protocol(run_config.simulation, run_config.protocol, out)


def run() -> None:
    app()


if __name__ == "__main__":
    run()
