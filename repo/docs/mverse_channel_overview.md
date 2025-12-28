# Mverse Channel Pipeline

This document summarizes the falsifiable simulation + analysis pipeline for the
coherence/topology/boundary-condition gated extra correlation channel hypothesis.

## Scope
- Baseline open quantum system simulation with measurement chain.
- Extended model with gated hidden-sector correlation channel.
- Synthetic experiment design with nulls and treatment conditions.
- Model fitting, model selection, and power analysis.
- Reproducible artifacts (data, metrics, figures, report).

## Key Modules
- `mverse_channel.physics`: baseline and extended dynamics, modulation proxies, topology proxies.
- `mverse_channel.metrics`: coherence, cross-correlation, non-Gaussianity, memory tests.
- `mverse_channel.inference`: likelihoods, fitting, AIC/BIC, posterior predictive checks.
- `mverse_channel.sim`: run simulations, protocol matrices, and sweeps.
- `mverse_channel.reporting`: figure and report generation.
