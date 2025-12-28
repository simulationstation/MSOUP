# Mverse Channel Report

## Configuration
```json
{
  "seed": 123,
  "n_levels": 4,
  "duration": 3.0,
  "dt": 0.1,
  "omega_a": 5.0,
  "omega_b": 5.6,
  "coupling_j": 0.08,
  "kappa_a": 0.06,
  "kappa_b": 0.08,
  "nth_a": 0.05,
  "nth_b": 0.05,
  "drive_amp_a": 0.1,
  "drive_amp_b": 0.07,
  "drive_freq": 5.0,
  "drive_phase": 0.0,
  "kerr_a": 0.0,
  "kerr_b": 0.0,
  "coherence_mode": "q_factor",
  "q_ref": 50.0,
  "kappa_ref": 0.2,
  "modulation": {
    "enabled": true,
    "delta_omega": 0.05,
    "omega_mod": 0.8
  },
  "topology": {
    "preset": "tree",
    "n_nodes": 4,
    "n_edges": 3,
    "cycle_count": 0,
    "max_cycles": 6
  },
  "measurement": {
    "sample_rate": 50.0,
    "amp_noise": 0.2,
    "shared_crosstalk": 0.0,
    "digitize_bits": null,
    "bandpass_low": null,
    "bandpass_high": null
  },
  "hidden_channel": {
    "enabled": true,
    "phenomenology": "additive",
    "epsilon0": 0.0,
    "epsilon_max": 0.15,
    "beta": 0.4,
    "gamma": 0.4,
    "eta": 0.4,
    "rho": 0.8,
    "tau_x": 1.5,
    "coherence_threshold": 0.5,
    "topology_threshold": 0.3,
    "boundary_threshold": 0.2,
    "threshold_width": 0.1
  }
}
```

## Metrics Summary
- **coherence_peak**: 1.000
- **coherence_mag**: 1.000
- **cross_corr_peak**: 0.497
- **non_gaussian**: -0.257
- **mardia_kurtosis**: 20.804
- **mardia_skewness**: 4.389
- **memory**: 1.150
- **covariance**: [[0.042823463777437204, 0.0007325168818689924, -0.00880856956855791, 0.0007572866669656944], [0.0007325168818689924, 0.02289681659576998, -0.0006897980560329875, -0.0023099776953748193], [-0.00880856956855791, -0.0006897980560329875, 0.048719179874612537, -0.0022073610732914043], [0.0007572866669656944, -0.0023099776953748193, -0.0022073610732914043, 0.025575415712129144]]
- **anomaly_score**: 0.723
- **freqs**: 16.000

## Figures
![coherence](figures/coherence.png)
![anomaly](figures/anomaly.png)
