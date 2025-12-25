# MSoup/Mverse Neutrality Report

## Definitions
- Δm: rank(A_+) − rank(A_-) from GF(2) constraint matrices.
- ΔS = −Δm ln 2 from exact solution counting.
- μ/T: inferred from Δ(F/T) per added independent constraint.
- Coarse-graining: projection of constraints onto blocks or random projections.

## Entropy sanity checks
- n=128 Δm=5 ΔS=-3.4657 (expected -3.4657)
- n=256 Δm=5 ΔS=-3.4657 (expected -3.4657)

## Dual-variable sweep
See figure: `figures/mu_over_t.png`.

## Coarse-graining
See figure: `figures/delta_m_coarsegrain.png`.

## Sensitivity summary
- μ/T within 0.10 of ln2 for 0.00% of sweep points.
- CG1 Δm(ℓ) stays O(1) for 100.00% of instances.
- CG2 Δm(ℓ) stays O(1) for 100.00% of instances.

RESULT: NO-GO
