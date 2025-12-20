"""
Report Generator for Mverse Shadow Simulation

Produces markdown report with simulation results and K1/K2 recommendations.
"""

from typing import Dict, Optional
from pathlib import Path
from datetime import datetime

from .config import SimConfig
from .metrics import MetricsSummary, format_metrics_table


def generate_report(config: SimConfig,
                     metrics_r500: MetricsSummary,
                     metrics_half: Optional[MetricsSummary],
                     output_path: Path,
                     runtime_seconds: Optional[float] = None) -> str:
    """
    Generate markdown report.

    Parameters
    ----------
    config : SimConfig
        Simulation configuration
    metrics_r500 : MetricsSummary
        Metrics at r500
    metrics_half : MetricsSummary, optional
        Metrics at 0.5*r500
    output_path : Path
        Output file path
    runtime_seconds : float, optional
        Total runtime

    Returns
    -------
    str
        Report content
    """
    tc = config.truth
    lc = config.lensing
    dc = config.dynamics
    ic = config.inference

    k1 = metrics_r500.k1
    k2 = metrics_r500.k2

    lines = [
        "# Mverse Shadow Simulation: K1/K2 Kill Condition Calibration",
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Executive Summary",
        "",
        "This simulation tests whether an M2 observer would falsely conclude that",
        "the gravity-channel projection is non-universal (alpha varies) even when",
        "the underlying truth is perfectly universal (alpha = 1.0 constant).",
        "",
        "**Key Finding:** Even with universal alpha, known systematic biases",
        "(hydrostatic bias, projection effects, anisotropy) cause substantial",
        "scatter in the observed ratio R = M_exc^lens / M_exc^dyn.",
        "",
        f"- **Median R_X** (lens/X-ray): {k1.R_X_median:.3f}",
        f"- **Median R_V** (lens/velocity): {k1.R_V_median:.3f}",
        f"- **Scatter (MAD) R_X**: {k1.R_X_mad:.3f}",
        f"- **Scatter (MAD) R_V**: {k1.R_V_mad:.3f}",
        "",
        "## Recommended Pre-Registration Thresholds",
        "",
        "Based on this simulation under the null hypothesis (universal alpha):",
        "",
        f"| Kill Condition | Metric | Recommended Tolerance |",
        f"|----------------|--------|----------------------|",
        f"| K1 (scatter) | 95th pct of \\|R-1\\| | **{metrics_r500.recommended_K1_tolerance:.3f}** |",
        f"| K2 (mass trend) | 3-sigma null slope | **{metrics_r500.recommended_K2_tolerance_mass:.4f}** dex^-1 |",
        f"| K2 (z trend) | 3-sigma null slope | **{metrics_r500.recommended_K2_tolerance_z:.4f}** |",
        "",
        "**Interpretation:**",
        "- K1: If observed |R-1| exceeds this tolerance for a significant fraction",
        f"  of clusters (>{5}%), the model may have a problem.",
        "- K2: If the slope of R vs log(M) or z exceeds these tolerances,",
        "  there may be real environmental dependence of alpha.",
        "",
        "## Simulation Settings",
        "",
        f"| Parameter | Value |",
        f"|-----------|-------|",
        f"| N clusters | {config.n_clusters} |",
        f"| Random seed | {config.seed} |",
        f"| Mode | {'Fast (vectorized)' if config.fast_mode else 'Full (per-object fitting)'} |",
        f"| z range | [{tc.z_min}, {tc.z_max}] |",
        f"| log(M200) range | [{tc.log_mass_min}, {tc.log_mass_max}] |",
        f"| Alpha (truth) | {tc.alpha_true} (universal) |",
    ]

    if runtime_seconds:
        lines.append(f"| Runtime | {runtime_seconds:.1f} s |")

    lines.extend([
        "",
        "### Bias Parameters",
        "",
        f"| Bias | Mean | Scatter |",
        f"|------|------|---------|",
        f"| Hydrostatic (b_hse) | {dc.b_hse_mean:.2f} | {dc.b_hse_scatter:.2f} |",
        f"| Projection boost | 0.00 | {tc.proj_boost_scatter:.2f} dex |",
        f"| Anisotropy (b_aniso) | {dc.b_aniso_mean:.2f} | {dc.b_aniso_scatter:.2f} |",
        f"| Lensing calibration | {lc.m_cal_mean:.3f} | {lc.m_cal_scatter:.3f} |",
        "",
        "### Noise Levels",
        "",
        f"| Observable | Noise (fractional) |",
        f"|------------|-------------------|",
        f"| X-ray mass | {dc.xray_noise:.0%} |",
        f"| Velocity dispersion | {dc.vel_noise:.0%} |",
        f"| Visible mass (gas) | {ic.f_gas_inferred_scatter:.0%} |",
        f"| Visible mass (stars) | {ic.f_star_inferred_scatter:.0%} |",
        "",
    ])

    # Add metrics tables
    lines.append(format_metrics_table(metrics_r500))

    if metrics_half:
        lines.append("")
        lines.append(format_metrics_table(metrics_half))

    lines.extend([
        "",
        "## Physical Interpretation",
        "",
        "### Why R_X > 1 (lensing > X-ray hydrostatic)",
        "",
        "The X-ray hydrostatic mass is biased low because:",
        f"- Non-thermal pressure support not accounted for (b_hse ~ {dc.b_hse_mean:.0%})",
        "- Analyst assumes hydrostatic equilibrium",
        "",
        f"This causes M_exc^dynX to be underestimated, pushing R_X above 1.",
        f"Expected: R_X ~ 1/(1-b_hse) ~ {1/(1-dc.b_hse_mean):.2f}",
        f"Observed median: {k1.R_X_median:.3f}",
        "",
        "### Scatter Sources",
        "",
        "1. **Projection effects**: Triaxial halos projected along different axes",
        "   cause scatter in lensing masses (not dynamics)",
        "",
        "2. **Intrinsic scatter in biases**: b_hse and b_aniso vary cluster-to-cluster",
        "",
        "3. **Measurement noise**: Shape noise (lensing), temperature errors (X-ray),",
        "   velocity errors (dynamics)",
        "",
        "4. **Visible mass uncertainty**: Gas and stellar fraction estimation errors",
        "",
        "## How to Use These Results",
        "",
        "1. **Pre-register K1/K2 tolerances** before analyzing real data",
        "",
        "2. **Do NOT interpret |R-1| < tolerance as evidence for universal gravity**",
        "   - This is the expected null result under standard systematics",
        "",
        "3. **DO interpret |R-1| >> tolerance as potential tension** that warrants",
        "   investigation of either:",
        "   - Underestimated systematics",
        "   - Real physics (non-universal gravity channel)",
        "",
        "4. **K2 trends are more diagnostic** than K1 scatter:",
        "   - Mass/z trends in R would be harder to explain with systematics",
        "   - But beware: b_hse has known mass/z dependence that creates false trends",
        "",
        "## Caveats",
        "",
        "- This simulation uses simplified NFW profiles and analytic scalings",
        "- Real cluster physics is more complex (baryonic effects, substructure)",
        "- Systematic biases in real data may differ from assumed values",
        "- Selection effects not modeled",
        "",
        "---",
        "",
        "*Report generated by mverse_shadow_sim*",
    ])

    content = "\n".join(lines)

    # Write to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(content)

    return content
