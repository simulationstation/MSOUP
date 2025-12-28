"""Generate figures for a saved run."""

from pathlib import Path

import numpy as np

from mverse_channel.reporting.figures import plot_coherence, plot_anomaly_scores


def main() -> None:
    data_dir = Path("outputs/mverse_demo")
    series = dict(np.load(data_dir / "data.npz"))
    figures_dir = data_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plot_coherence(series, 50.0, figures_dir / "coherence.png")
    plot_anomaly_scores({"demo": 0.0}, figures_dir / "anomaly.png")


if __name__ == "__main__":
    main()
