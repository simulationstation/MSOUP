"""
Strong lensing substructure constraints for Msoup pipeline.

This module encodes constraints from:
- Vegetti et al. papers on gravitational imaging
- Compiled subhalo mass function constraints
- Half-mode mass limits from flux-ratio anomalies

Key references:
- Vegetti 2024 Space Science Reviews: "Strong gravitational lensing as a probe of dark matter"
- Gilman et al. 2020: Warm dark matter constraints from flux ratios
- Hsueh et al. 2020: SHARP analysis of substructure

Constraint types:
1. HARD: Direct detections and upper limits from real data
2. FORECAST: Projected sensitivities (Roman, Rubin) - for consistency checks only

The primary constraint is on the subhalo mass function normalization and cutoff mass.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable
from scipy.stats import norm


@dataclass
class SubhaloMFConstraint:
    """
    Constraint on the subhalo mass function.

    The subhalo mass function is often parameterized as:
        dn/dM = A * (M / M_pivot)^alpha * f_suppress(M, M_hm)

    where f_suppress encodes the suppression from warm/fuzzy DM or other effects.

    For WDM/suppression models:
        f_suppress(M, M_hm) ~ [1 + (M_hm / M)^beta]^(-gamma)

    Key observable: M_hm (half-mode mass) where the HMF is suppressed by 50%.
    """
    name: str
    constraint_type: str       # "HARD" or "FORECAST"
    z_lens: float              # Lens redshift
    z_source: float            # Source redshift
    M_hm_limit: float          # Half-mode mass limit in M_sun
    M_hm_uncertainty: float    # Uncertainty (1-sigma)
    is_upper_limit: bool       # True if this is an upper limit
    reference: str             # Paper reference
    notes: str = ""

    def log_likelihood(self, M_hm_model: float) -> float:
        """
        Compute log-likelihood of model given this constraint.

        For upper limits: likelihood = 1 if M_hm_model < limit, else penalized
        For measurements: Gaussian likelihood
        """
        if self.is_upper_limit:
            # Upper limit: no penalty if below limit, steep penalty above
            if M_hm_model <= self.M_hm_limit:
                return 0.0
            else:
                # Penalty for exceeding upper limit
                excess = (M_hm_model - self.M_hm_limit) / self.M_hm_uncertainty
                return -0.5 * excess**2
        else:
            # Gaussian measurement
            return norm.logpdf(M_hm_model, self.M_hm_limit, self.M_hm_uncertainty)


@dataclass
class FluxRatioConstraint:
    """
    Constraint from quadruply-lensed quasar flux ratios.

    Flux ratio anomalies are sensitive to substructure at 10^6-10^9 M_sun.
    """
    name: str
    constraint_type: str
    z_lens: float
    M_sub_min: float           # Minimum detectable subhalo mass
    M_sub_max: float           # Maximum probed mass
    f_sub_limit: float         # Substructure mass fraction limit
    f_sub_uncertainty: float
    reference: str


@dataclass
class LensingConstraints:
    """
    Collection of strong lensing constraints.

    Provides likelihood evaluation for Msoup model parameters.
    """
    subhalo_mf_constraints: List[SubhaloMFConstraint] = field(default_factory=list)
    flux_ratio_constraints: List[FluxRatioConstraint] = field(default_factory=list)

    @classmethod
    def load_default(cls) -> "LensingConstraints":
        """
        Load default constraints from literature.

        These are conservative constraints compiled from:
        - Gilman et al. 2020 (flux ratios)
        - Vegetti et al. 2018 (gravitational imaging)
        - Hsueh et al. 2020 (SHARP)
        - Enzi et al. 2021 (combined analysis)
        """
        constraints = cls()

        # ====================================================================
        # HARD CONSTRAINTS (from real data)
        # ====================================================================

        # Gilman et al. 2020 - Warm DM constraint from 11 quasar lenses
        # M_hm < 10^7.6 M_sun at 2-sigma (conservative)
        # This translates to WDM mass > 5.2 keV
        constraints.subhalo_mf_constraints.append(SubhaloMFConstraint(
            name="Gilman2020_FluxRatios",
            constraint_type="HARD",
            z_lens=0.5,  # Typical lens redshift
            z_source=2.0,
            M_hm_limit=10**7.6,  # ~4e7 M_sun
            M_hm_uncertainty=10**7.6 * 0.5,  # Factor ~2 uncertainty
            is_upper_limit=True,
            reference="Gilman et al. 2020, MNRAS 491, 6077",
            notes="Combined analysis of 11 quadruply lensed quasars"
        ))

        # Hsueh et al. 2020 - SHARP program
        # Consistent with CDM, weak upper limit on M_hm
        constraints.subhalo_mf_constraints.append(SubhaloMFConstraint(
            name="Hsueh2020_SHARP",
            constraint_type="HARD",
            z_lens=0.6,
            z_source=1.5,
            M_hm_limit=10**8.0,  # 10^8 M_sun
            M_hm_uncertainty=10**8.0 * 0.7,
            is_upper_limit=True,
            reference="Hsueh et al. 2020, MNRAS 492, 3047",
            notes="SHARP gravitational imaging analysis"
        ))

        # Vegetti et al. 2018 - Direct subhalo detection
        # Detection at ~10^9 M_sun implies CDM-like abundance at that scale
        # This sets a LOWER bound on where suppression can kick in
        constraints.subhalo_mf_constraints.append(SubhaloMFConstraint(
            name="Vegetti2018_Detection",
            constraint_type="HARD",
            z_lens=0.88,
            z_source=2.06,
            M_hm_limit=10**9.0,  # Detections at this scale
            M_hm_uncertainty=10**9.0 * 0.5,
            is_upper_limit=False,  # This is a detection, not upper limit
            reference="Vegetti et al. 2018, MNRAS 481, 3661",
            notes="Direct detection of 10^9 M_sun subhalo"
        ))

        # Enzi et al. 2021 - Combined constraint
        # Most stringent combined analysis
        constraints.subhalo_mf_constraints.append(SubhaloMFConstraint(
            name="Enzi2021_Combined",
            constraint_type="HARD",
            z_lens=0.5,
            z_source=2.0,
            M_hm_limit=10**7.8,
            M_hm_uncertainty=10**7.8 * 0.4,
            is_upper_limit=True,
            reference="Enzi et al. 2021, MNRAS 506, 5848",
            notes="Combined flux ratio + imaging analysis"
        ))

        # ====================================================================
        # FORECAST CONSTRAINTS (from projections)
        # ====================================================================

        # Roman strong lensing forecast
        # Projected to probe down to 10^6 M_sun
        constraints.subhalo_mf_constraints.append(SubhaloMFConstraint(
            name="Roman_Forecast",
            constraint_type="FORECAST",
            z_lens=0.5,
            z_source=2.0,
            M_hm_limit=10**6.5,
            M_hm_uncertainty=10**6.5 * 0.3,
            is_upper_limit=True,
            reference="Oguri & Marshall 2010; Roman projections",
            notes="Forecast - not actual constraint"
        ))

        # Rubin LSST forecast
        constraints.subhalo_mf_constraints.append(SubhaloMFConstraint(
            name="Rubin_Forecast",
            constraint_type="FORECAST",
            z_lens=0.4,
            z_source=1.5,
            M_hm_limit=10**7.0,
            M_hm_uncertainty=10**7.0 * 0.4,
            is_upper_limit=True,
            reference="LSST Science Collaboration projections",
            notes="Forecast - not actual constraint"
        ))

        return constraints

    def get_hard_constraints(self) -> List[SubhaloMFConstraint]:
        """Return only HARD (real data) constraints."""
        return [c for c in self.subhalo_mf_constraints
                if c.constraint_type == "HARD"]

    def get_forecast_constraints(self) -> List[SubhaloMFConstraint]:
        """Return only FORECAST constraints."""
        return [c for c in self.subhalo_mf_constraints
                if c.constraint_type == "FORECAST"]

    def log_likelihood_hard(self, M_hm_model: float) -> float:
        """
        Compute log-likelihood from all HARD constraints.

        Parameters:
            M_hm_model: Model prediction for half-mode mass in M_sun

        Returns:
            Total log-likelihood
        """
        hard = self.get_hard_constraints()
        if not hard:
            return 0.0

        log_lik = 0.0
        for constraint in hard:
            log_lik += constraint.log_likelihood(M_hm_model)

        return log_lik

    def log_likelihood_all(self, M_hm_model: float) -> float:
        """
        Compute log-likelihood from all constraints.
        """
        log_lik = 0.0
        for constraint in self.subhalo_mf_constraints:
            log_lik += constraint.log_likelihood(M_hm_model)
        return log_lik

    def summary(self) -> str:
        """Return summary of constraints."""
        lines = ["Strong Lensing Substructure Constraints", "=" * 50]

        hard = self.get_hard_constraints()
        lines.append(f"\nHARD constraints ({len(hard)}):")
        for c in hard:
            limit_type = "upper limit" if c.is_upper_limit else "detection"
            lines.append(f"  - {c.name}: M_hm {'<' if c.is_upper_limit else '~'} "
                        f"{c.M_hm_limit:.2e} M_sun ({limit_type})")
            lines.append(f"    z_lens={c.z_lens:.2f}, Ref: {c.reference}")

        forecast = self.get_forecast_constraints()
        lines.append(f"\nFORECAST constraints ({len(forecast)}):")
        for c in forecast:
            lines.append(f"  - {c.name}: M_hm < {c.M_hm_limit:.2e} M_sun")

        return "\n".join(lines)

    def is_consistent(self, M_hm_model: float,
                      use_forecasts: bool = False) -> Tuple[bool, str]:
        """
        Check if model is consistent with constraints.

        Parameters:
            M_hm_model: Model prediction for half-mode mass
            use_forecasts: Include forecast constraints

        Returns:
            (is_consistent, message)
        """
        constraints = (self.subhalo_mf_constraints if use_forecasts
                      else self.get_hard_constraints())

        violations = []
        for c in constraints:
            if c.is_upper_limit:
                # For upper limits, model must be below
                if M_hm_model > c.M_hm_limit + 2 * c.M_hm_uncertainty:
                    violations.append(
                        f"{c.name}: M_hm={M_hm_model:.2e} > limit={c.M_hm_limit:.2e}"
                    )
            else:
                # For detections, model must not suppress too much
                if M_hm_model > c.M_hm_limit + 2 * c.M_hm_uncertainty:
                    violations.append(
                        f"{c.name}: M_hm={M_hm_model:.2e} >> detection scale {c.M_hm_limit:.2e}"
                    )

        if violations:
            return False, "VIOLATIONS:\n" + "\n".join(violations)
        else:
            return True, "Model consistent with all constraints"


def convert_wdm_mass_to_M_hm(m_wdm_keV: float) -> float:
    """
    Convert WDM particle mass to half-mode mass.

    The half-mode mass for thermal WDM is approximately:
        M_hm ~ 10^10 * (m_wdm / keV)^(-3.3) M_sun

    Reference: Schneider et al. 2012

    Parameters:
        m_wdm_keV: WDM particle mass in keV

    Returns:
        M_hm in M_sun
    """
    return 1e10 * (m_wdm_keV)**(-3.33)


def convert_M_hm_to_wdm_mass(M_hm: float) -> float:
    """
    Convert half-mode mass to equivalent WDM particle mass.

    Parameters:
        M_hm: Half-mode mass in M_sun

    Returns:
        m_wdm in keV
    """
    return (M_hm / 1e10)**(-1.0/3.33)


def subhalo_mass_function_ratio(M_sub: np.ndarray,
                                M_hm: float,
                                alpha: float = -1.9,
                                beta: float = 1.0,
                                gamma: float = 2.7) -> np.ndarray:
    """
    Ratio of suppressed to CDM subhalo mass function.

    dn/dM (suppressed) / dn/dM (CDM) = [1 + (M_hm / M)^beta]^(-gamma)

    This is a common parameterization for WDM-like suppression.

    Parameters:
        M_sub: Subhalo masses in M_sun
        M_hm: Half-mode mass in M_sun
        alpha: CDM slope (not used in ratio, but for reference)
        beta: Suppression steepness
        gamma: Suppression amplitude

    Returns:
        Ratio array (1 for CDM, <1 for suppression)
    """
    M_sub = np.asarray(M_sub)
    if M_hm <= 0:
        return np.ones_like(M_sub)

    ratio = (1 + (M_hm / M_sub)**beta)**(-gamma)
    return ratio


def lensing_signal_prediction(M_hm: float,
                              z_lens: float = 0.5,
                              f_sub_cdm: float = 0.01) -> Dict:
    """
    Predict lensing observables for a given M_hm.

    Parameters:
        M_hm: Half-mode mass in M_sun
        z_lens: Lens redshift
        f_sub_cdm: CDM substructure mass fraction

    Returns:
        Dictionary with predicted observables
    """
    # Mass range probed by lensing
    M_min = 1e6  # Detection threshold
    M_max = 1e10  # Upper bound

    M_grid = np.logspace(np.log10(M_min), np.log10(M_max), 100)

    # Suppression ratio
    ratio = subhalo_mass_function_ratio(M_grid, M_hm)

    # Effective substructure fraction
    # Integrate suppression over the mass function
    # Approximate: use geometric mean of ratio
    avg_suppression = np.exp(np.mean(np.log(ratio)))
    f_sub_suppressed = f_sub_cdm * avg_suppression

    # Number of detectable subhalos
    # Rough scaling: N_sub ∝ integral of dn/dM
    N_sub_cdm = 1.0  # Normalized
    N_sub_suppressed = N_sub_cdm * avg_suppression

    # Flux ratio anomaly signal
    # Simplified: larger suppression -> smaller anomaly
    flux_anomaly_rms = 0.1 * avg_suppression**(0.5)  # Rough scaling

    return {
        "M_hm": M_hm,
        "z_lens": z_lens,
        "f_sub_suppressed": f_sub_suppressed,
        "f_sub_ratio": avg_suppression,
        "N_sub_ratio": N_sub_suppressed,
        "flux_anomaly_rms": flux_anomaly_rms,
        "detectable": M_hm < 1e8,  # Rough detectability threshold
    }
