"""
Strong lensing substructure constraints for Msoup pipeline.

This module implements PER-PAPER conservative checks rather than a combined
likelihood. Each paper's constraint is evaluated independently with explicit
mapping assumptions documented.

Key references:
- Gilman et al. 2020, MNRAS 491, 6077: Flux ratio anomalies
- Hsueh et al. 2020, MNRAS 492, 3047: SHARP gravitational imaging
- Vegetti et al. 2018, MNRAS 481, 3661: Direct subhalo detection
- Enzi et al. 2021, MNRAS 506, 5848: Combined WDM analysis

IMPORTANT: This module does NOT combine heterogeneous constraints into a single
likelihood. Each paper is treated as an independent consistency check.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum


class ConstraintStatus(Enum):
    """Status of a constraint check."""
    PASSED = "passed"
    VIOLATED = "violated"
    NOT_COMPARABLE = "not_comparable"  # When mapping is uncertain


@dataclass
class PaperConstraint:
    """
    Single constraint from a published paper.

    Each constraint is evaluated independently with explicit documentation
    of what the paper actually measures and how we map it to our model.
    """
    name: str
    reference: str
    constraint_type: str  # "upper_limit", "detection", "measurement"

    # What the paper actually reports
    reported_quantity: str  # e.g., "m_WDM", "M_hm", "f_sub"
    reported_value: float
    reported_uncertainty: float
    reported_units: str
    confidence_level: float  # e.g., 0.95 for 95% CL

    # Mapping to our model
    mapping_used: str  # Description of how we convert to M_hm
    mapping_assumptions: List[str]
    M_hm_equivalent: Optional[float]  # M_hm in M_sun, or None if not comparable

    # Redshift info
    z_lens: float
    z_source: float

    notes: str = ""

    def evaluate(self, model_M_hm: float) -> Tuple[ConstraintStatus, Dict]:
        """
        Evaluate this constraint against a model prediction.

        Parameters:
            model_M_hm: Model's half-mode mass in M_sun

        Returns:
            (status, details_dict)
        """
        if self.M_hm_equivalent is None:
            return ConstraintStatus.NOT_COMPARABLE, {
                "reason": "No valid M_hm mapping available",
                "notes": self.notes
            }

        if self.constraint_type == "upper_limit":
            # Model must not exceed the upper limit
            satisfied = model_M_hm <= self.M_hm_equivalent
            sigma_distance = (model_M_hm - self.M_hm_equivalent) / max(self.reported_uncertainty, 1e10)

            return (
                ConstraintStatus.PASSED if satisfied else ConstraintStatus.VIOLATED,
                {
                    "limit": self.M_hm_equivalent,
                    "model": model_M_hm,
                    "satisfied": satisfied,
                    "sigma_distance": sigma_distance,
                    "comparison": f"M_hm_model ({model_M_hm:.2e}) {'<=' if satisfied else '>'} limit ({self.M_hm_equivalent:.2e})"
                }
            )

        elif self.constraint_type == "detection":
            # Detection at scale M means suppression scale must be below M
            # (otherwise we'd suppress the subhalos that were detected)
            satisfied = model_M_hm <= self.M_hm_equivalent

            return (
                ConstraintStatus.PASSED if satisfied else ConstraintStatus.VIOLATED,
                {
                    "detection_scale": self.M_hm_equivalent,
                    "model": model_M_hm,
                    "satisfied": satisfied,
                    "comparison": f"Detection at {self.M_hm_equivalent:.2e} requires M_hm < this scale"
                }
            )

        else:
            return ConstraintStatus.NOT_COMPARABLE, {
                "reason": f"Unknown constraint type: {self.constraint_type}"
            }


def wdm_mass_to_M_hm(m_wdm_keV: float) -> float:
    """
    Convert WDM particle mass to half-mode mass.

    Uses the standard thermal relic mapping from Schneider et al. 2012:
        M_hm = 10^10 * (m_WDM / keV)^(-3.33) M_sun

    This is the ONLY WDM-to-M_hm mapping used in this module.
    All papers reporting m_WDM constraints are converted using this formula.
    """
    if m_wdm_keV <= 0:
        return np.inf
    return 1e10 * (m_wdm_keV)**(-3.33)


def M_hm_to_wdm_mass(M_hm: float) -> float:
    """
    Convert half-mode mass to equivalent WDM particle mass.

    Inverse of wdm_mass_to_M_hm.
    """
    if M_hm <= 0:
        return np.inf
    return (M_hm / 1e10)**(-1.0/3.33)


@dataclass
class LensingConstraints:
    """
    Collection of per-paper lensing constraints.

    Each constraint is evaluated independently. NO combined likelihood.
    """
    constraints: List[PaperConstraint] = field(default_factory=list)

    @classmethod
    def load_default(cls) -> "LensingConstraints":
        """
        Load constraints from literature with explicit mapping documentation.

        IMPORTANT: Each constraint documents:
        1. What the paper actually reports
        2. How we map it to M_hm (if at all)
        3. What assumptions are involved
        """
        lc = cls()

        # ================================================================
        # Gilman et al. 2020, MNRAS 491, 6077
        # "Warm dark matter constraints from strong gravitational lensing"
        # ================================================================
        # They report: m_WDM > 5.2 keV (95% CL) from 11 quasar lenses
        m_wdm_gilman = 5.2  # keV, 95% CL lower limit
        M_hm_gilman = wdm_mass_to_M_hm(m_wdm_gilman)

        lc.constraints.append(PaperConstraint(
            name="Gilman2020_FluxRatios",
            reference="Gilman et al. 2020, MNRAS 491, 6077",
            constraint_type="upper_limit",
            reported_quantity="m_WDM",
            reported_value=5.2,
            reported_uncertainty=0.8,  # Approximate from their posteriors
            reported_units="keV",
            confidence_level=0.95,
            mapping_used="Schneider et al. 2012 thermal relic relation",
            mapping_assumptions=[
                "Thermal relic WDM",
                "Standard transfer function shape",
                "M_hm = 10^10 * (m_WDM/keV)^(-3.33) M_sun"
            ],
            M_hm_equivalent=M_hm_gilman,  # ~4e7 M_sun
            z_lens=0.5,  # typical
            z_source=2.0,
            notes="Combined analysis of 11 quadruply lensed quasars"
        ))

        # ================================================================
        # Hsueh et al. 2020, MNRAS 492, 3047
        # "SHARP - VII. New constraints on warm dark matter"
        # ================================================================
        # They report: m_WDM > 4.0 keV (95% CL) from SHARP imaging
        m_wdm_hsueh = 4.0
        M_hm_hsueh = wdm_mass_to_M_hm(m_wdm_hsueh)

        lc.constraints.append(PaperConstraint(
            name="Hsueh2020_SHARP",
            reference="Hsueh et al. 2020, MNRAS 492, 3047",
            constraint_type="upper_limit",
            reported_quantity="m_WDM",
            reported_value=4.0,
            reported_uncertainty=1.0,
            reported_units="keV",
            confidence_level=0.95,
            mapping_used="Schneider et al. 2012 thermal relic relation",
            mapping_assumptions=[
                "Thermal relic WDM",
                "Standard transfer function shape",
                "M_hm = 10^10 * (m_WDM/keV)^(-3.33) M_sun"
            ],
            M_hm_equivalent=M_hm_hsueh,  # ~1e8 M_sun
            z_lens=0.6,
            z_source=1.5,
            notes="SHARP gravitational imaging analysis"
        ))

        # ================================================================
        # Vegetti et al. 2018, MNRAS 481, 3661
        # "Constraining sterile neutrino cosmologies with strong lensing"
        # ================================================================
        # They DETECT a subhalo at ~10^9 M_sun
        # This implies CDM-like abundance at this scale, setting an upper bound

        lc.constraints.append(PaperConstraint(
            name="Vegetti2018_Detection",
            reference="Vegetti et al. 2018, MNRAS 481, 3661",
            constraint_type="detection",
            reported_quantity="M_sub",
            reported_value=1e9,
            reported_uncertainty=5e8,
            reported_units="M_sun",
            confidence_level=0.95,
            mapping_used="Direct mass detection",
            mapping_assumptions=[
                "Subhalo mass from Einstein ring perturbation",
                "If M_hm > M_sub, this detection would be suppressed",
                "Conservative: require M_hm < M_detection"
            ],
            M_hm_equivalent=1e9,  # Detection scale
            z_lens=0.88,
            z_source=2.06,
            notes="Direct detection - model must not suppress this scale"
        ))

        # ================================================================
        # Enzi et al. 2021, MNRAS 506, 5848
        # "Combined constraints on warm dark matter"
        # ================================================================
        # Combined analysis: m_WDM > 6.3 keV (95% CL)
        # This is the MOST STRINGENT published constraint
        m_wdm_enzi = 6.3
        M_hm_enzi = wdm_mass_to_M_hm(m_wdm_enzi)

        lc.constraints.append(PaperConstraint(
            name="Enzi2021_Combined",
            reference="Enzi et al. 2021, MNRAS 506, 5848",
            constraint_type="upper_limit",
            reported_quantity="m_WDM",
            reported_value=6.3,
            reported_uncertainty=0.5,
            reported_units="keV",
            confidence_level=0.95,
            mapping_used="Schneider et al. 2012 thermal relic relation",
            mapping_assumptions=[
                "Thermal relic WDM",
                "Combined flux ratio + imaging analysis",
                "M_hm = 10^10 * (m_WDM/keV)^(-3.33) M_sun"
            ],
            M_hm_equivalent=M_hm_enzi,  # ~2e7 M_sun
            z_lens=0.5,
            z_source=2.0,
            notes="Most stringent combined constraint"
        ))

        return lc

    def get_constraint_table(self) -> List[Dict]:
        """
        Generate a table of all constraints with mapping info.

        Returns list of dicts suitable for CSV export.
        """
        table = []
        for c in self.constraints:
            table.append({
                "paper": c.name,
                "reference": c.reference,
                "reported_quantity": c.reported_quantity,
                "reported_value": c.reported_value,
                "reported_units": c.reported_units,
                "confidence_level": c.confidence_level,
                "mapping_used": c.mapping_used,
                "M_hm_equivalent_Msun": c.M_hm_equivalent,
                "m_WDM_equivalent_keV": M_hm_to_wdm_mass(c.M_hm_equivalent) if c.M_hm_equivalent else None,
                "constraint_type": c.constraint_type,
                "z_lens": c.z_lens,
                "notes": c.notes
            })
        return table

    def evaluate_all(self, model_M_hm: float) -> Dict:
        """
        Evaluate all constraints against a model prediction.

        Parameters:
            model_M_hm: Model's half-mode mass in M_sun

        Returns:
            Dictionary with per-paper results and summary
        """
        results = []
        n_passed = 0
        n_violated = 0
        n_not_comparable = 0

        for c in self.constraints:
            status, details = c.evaluate(model_M_hm)

            result = {
                "paper": c.name,
                "reference": c.reference,
                "status": status.value,
                "constraint_type": c.constraint_type,
                "M_hm_limit": c.M_hm_equivalent,
                "model_M_hm": model_M_hm,
                **details
            }
            results.append(result)

            if status == ConstraintStatus.PASSED:
                n_passed += 1
            elif status == ConstraintStatus.VIOLATED:
                n_violated += 1
            else:
                n_not_comparable += 1

        return {
            "model_M_hm": model_M_hm,
            "n_constraints": len(self.constraints),
            "n_passed": n_passed,
            "n_violated": n_violated,
            "n_not_comparable": n_not_comparable,
            "overall_consistent": n_violated == 0,
            "constraints": results
        }

    def evaluate_constraints(self, M_hm_model: float,
                            mode: str = "consistency",
                            use_forecasts: bool = False) -> Dict:
        """
        Backward-compatible method matching old interface.

        Parameters:
            M_hm_model: Model's half-mode mass in M_sun
            mode: "consistency" (default) - just check bounds
            use_forecasts: Ignored (no forecasts in this implementation)

        Returns:
            Dictionary with per-constraint results
        """
        eval_result = self.evaluate_all(M_hm_model)

        # Reformat to match old interface
        constraints_formatted = []
        for r in eval_result["constraints"]:
            constraints_formatted.append({
                "name": r["paper"],
                "satisfied": r["status"] == "passed",
                "type": r["constraint_type"],
                "reference": r["reference"],
                "details": r
            })

        return {
            "mode": mode,
            "overall_consistent": eval_result["overall_consistent"],
            "constraints": constraints_formatted,
            "message": f"{eval_result['n_passed']}/{eval_result['n_constraints']} constraints passed"
        }

    def is_consistent(self, M_hm_model: float, **kwargs) -> Tuple[bool, str]:
        """Check if model is consistent with all constraints."""
        result = self.evaluate_constraints(M_hm_model, **kwargs)
        return result["overall_consistent"], result["message"]

    def summary(self) -> str:
        """Return human-readable summary of constraints."""
        lines = [
            "Strong Lensing Constraints (Per-Paper)",
            "=" * 50,
            "",
            "MAPPING NOTE: WDM mass to M_hm uses:",
            "  M_hm = 10^10 * (m_WDM/keV)^(-3.33) M_sun",
            "  (Schneider et al. 2012 thermal relic relation)",
            ""
        ]

        for c in self.constraints:
            lines.append(f"[{c.name}]")
            lines.append(f"  Reference: {c.reference}")
            lines.append(f"  Reports: {c.reported_quantity} = {c.reported_value} {c.reported_units}")
            lines.append(f"  Type: {c.constraint_type} at {c.confidence_level*100:.0f}% CL")
            if c.M_hm_equivalent:
                lines.append(f"  M_hm equivalent: {c.M_hm_equivalent:.2e} M_sun")
            lines.append(f"  Mapping: {c.mapping_used}")
            lines.append("")

        return "\n".join(lines)


def get_most_constraining_M_hm() -> float:
    """
    Return the most constraining M_hm upper limit from the literature.

    Currently: Enzi et al. 2021 with m_WDM > 6.3 keV -> M_hm < 2e7 M_sun
    """
    return wdm_mass_to_M_hm(6.3)  # ~2e7 M_sun
