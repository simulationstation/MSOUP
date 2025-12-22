from __future__ import annotations

import numpy as np

from .constants import SPEED_OF_LIGHT_KM_S
from .distances import angular_diameter_distance, comoving_distance, luminosity_distance
from .config import DEFAULT_H_EARLY, DEFAULT_OMEGA_L0, DEFAULT_OMEGA_M0


def distance_modulus(z: np.ndarray, delta_m: float, nuisance_M: float = 0.0, **kwargs) -> np.ndarray:
    """Compute distance modulus mu."""
    dl = luminosity_distance(z, delta_m=delta_m, **kwargs)  # km/s units cancel via c/H
    mu = 5.0 * np.log10(np.maximum(dl, 1e-9)) + 25.0 + nuisance_M
    return mu


def bao_dv(z: np.ndarray, delta_m: float, **kwargs) -> np.ndarray:
    """
    Compute BAO volume distance D_V in Mpc.

    D_V(z) = [ (1+z)^2 * D_A(z)^2 * c*z / H(z) ]^(1/3)
    """
    z = np.asarray(z, dtype=float)
    da = angular_diameter_distance(z, delta_m=delta_m, **kwargs)
    from .model import h_eff

    h_eff_vals = h_eff(
        z,
        delta_m=delta_m,
        h_early=kwargs.get("h_early"),
        omega_m0=kwargs.get("omega_m0"),
        omega_L0=kwargs.get("omega_L0"),
    )
    c_km_s = kwargs.get("c_km_s", SPEED_OF_LIGHT_KM_S)
    dv = ((1 + z) ** 2 * da ** 2 * (c_km_s * z / h_eff_vals)) ** (1.0 / 3.0)
    return dv


def bao_dm(z: np.ndarray, delta_m: float, **kwargs) -> np.ndarray:
    """
    Compute BAO comoving distance D_M in Mpc.

    D_M(z) = (1+z) * D_A(z)
    """
    z = np.asarray(z, dtype=float)
    da = angular_diameter_distance(z, delta_m=delta_m, **kwargs)
    return (1 + z) * da


def bao_dh(z: np.ndarray, delta_m: float, **kwargs) -> np.ndarray:
    """
    Compute BAO Hubble distance D_H in Mpc.

    D_H(z) = c / H(z)
    where c is in km/s and H(z) is in km/s/Mpc, so D_H is in Mpc.
    """
    z = np.asarray(z, dtype=float)
    from .model import h_eff

    h_eff_vals = h_eff(
        z,
        delta_m=delta_m,
        h_early=kwargs.get("h_early"),
        omega_m0=kwargs.get("omega_m0"),
        omega_L0=kwargs.get("omega_L0"),
    )
    c_km_s = kwargs.get("c_km_s", SPEED_OF_LIGHT_KM_S)
    return c_km_s / h_eff_vals


def canonicalize_bao_observable(observable: str) -> str:
    """Normalize BAO observable strings and validate dimensionless ratios."""
    obs_upper = observable.upper().replace(" ", "")
    if obs_upper in ("DV/RD", "DV_RD", "DVRD"):
        return "DV/rd"
    if obs_upper in ("DM/RD", "DM_RD", "DMRD"):
        return "DM/rd"
    if obs_upper in ("DH/RD", "DH_RD", "DHRD"):
        return "DH/rd"
    raise ValueError(
        f"Unsupported BAO observable '{observable}'. Expected DV/rd, DM/rd, or DH/rd."
    )


def resolve_rd_fid_scaling(
    observable: str,
    rd_mpc: float,
    rd_fid_mpc: float | None = None,
    rd_scaling: str | None = None,
    paper_tag: str | None = None,
    rd_fid_conventions: dict | None = None,
) -> tuple[float, str | None]:
    """
    Determine multiplicative factor to map a predicted DV/rd, DM/rd, or DH/rd
    to the convention used in a publication when an rd_fid is provided.

    Args:
        observable: Canonical observable name (DV/rd, DM/rd, DH/rd)
        rd_mpc: Sound horizon used for predictions
        rd_fid_mpc: Optional fiducial sound horizon from publication
        rd_scaling: Optional explicit hint, one of:
            - "multiply_by_rd_over_rdfid": predicted*(rd/rd_fid)
            - "multiply_by_rdfid_over_rd": predicted*(rd_fid/rd)
        paper_tag: Optional publication tag for lookup
        rd_fid_conventions: Optional mapping {paper_tag: rd_scaling}

    Returns:
        (scaling_factor, applied_hint or None)
    """
    if rd_fid_mpc is None:
        return 1.0, None

    hint = rd_scaling
    if hint is None and rd_fid_conventions and paper_tag:
        hint = rd_fid_conventions.get(paper_tag)

    if hint is None:
        return 1.0, None

    hint_lower = hint.lower()
    if hint_lower == "multiply_by_rd_over_rdfid":
        return rd_mpc / rd_fid_mpc, hint_lower
    if hint_lower == "multiply_by_rdfid_over_rd":
        return rd_fid_mpc / rd_mpc, hint_lower

    raise ValueError(
        f"Unknown rd_scaling hint '{hint}'. Use multiply_by_rd_over_rdfid or multiply_by_rdfid_over_rd."
    )


def bao_predict(
    z: float,
    observable: str,
    delta_m: float,
    rd_mpc: float,
    sanity_check: bool = False,
    rd_fid_mpc: float | None = None,
    rd_scaling: str | None = None,
    rd_fid_conventions: dict | None = None,
    paper_tag: str | None = None,
    **kwargs,
) -> float:
    """
    Compute BAO observable prediction for a single redshift.

    Args:
        z: Redshift
        observable: One of 'DV/rd', 'DM/rd', 'DH/rd' (case-insensitive)
        delta_m: MSOUP delta_m parameter
        rd_mpc: Sound horizon at drag epoch in Mpc (fixed, not fitted)
        sanity_check: If True, perform and log sanity assertions
        **kwargs: Passed to distance functions (h_early, omega_m0, omega_L0, c_km_s)

    Returns:
        Predicted observable value (dimensionless ratio)

    Raises:
        ValueError: If observable type is not recognized or sanity check fails
    """
    from .model import h_eff

    canonical_obs = canonicalize_bao_observable(observable)
    z_arr = np.array([z])
    c_km_s = kwargs.get("c_km_s", SPEED_OF_LIGHT_KM_S)

    # Compute base quantities
    h_early = kwargs.get("h_early", DEFAULT_H_EARLY)
    omega_m0 = kwargs.get("omega_m0", DEFAULT_OMEGA_M0)
    omega_L0 = kwargs.get("omega_L0", DEFAULT_OMEGA_L0)

    da = angular_diameter_distance(z_arr, delta_m=delta_m, h_early=h_early, omega_m0=omega_m0, omega_L0=omega_L0, c_km_s=c_km_s)[0]
    dm = (1 + z) * da  # D_M = (1+z) * D_A
    h_z = h_eff(z_arr, delta_m=delta_m, h_early=h_early, omega_m0=omega_m0, omega_L0=omega_L0)[0]
    dh = c_km_s / h_z  # D_H = c / H(z) in Mpc

    if sanity_check:
        # Sanity assertions
        if da <= 0:
            raise ValueError(f"Sanity check failed: D_A(z={z}) = {da} <= 0")
        if dm <= 0:
            raise ValueError(f"Sanity check failed: D_M(z={z}) = {dm} <= 0")
        if dh <= 0:
            raise ValueError(f"Sanity check failed: D_H(z={z}) = {dh} <= 0")

        # Check D_M = (1+z) * D_A identity
        dm_check = (1 + z) * da
        if abs(dm - dm_check) > 1e-10:
            raise ValueError(f"Sanity check failed: D_M(z={z}) = {dm} != (1+z)*D_A = {dm_check}")

        # Check H(z) is in plausible range (40 < H(z) < 400 for 0 <= z <= 3)
        if not (40 < h_z < 400):
            raise ValueError(f"Sanity check failed: H(z={z}) = {h_z} not in [40, 400] km/s/Mpc")

        # Check ratios are positive and dimensionless (order-of-magnitude reasonable)
        if rd_mpc <= 0:
            raise ValueError(f"Sanity check failed: rd_mpc = {rd_mpc} <= 0")

    if canonical_obs == "DV/rd":
        dv = bao_dv(z_arr, delta_m=delta_m, h_early=h_early, omega_m0=omega_m0, omega_L0=omega_L0, c_km_s=c_km_s)[0]
        result = dv / rd_mpc
    elif canonical_obs == "DM/rd":
        result = dm / rd_mpc
    elif canonical_obs == "DH/rd":
        result = dh / rd_mpc
    else:
        raise ValueError(f"Unknown BAO observable: '{observable}'. Supported: DV/rd, DM/rd, DH/rd")

    scaling, applied_hint = resolve_rd_fid_scaling(
        canonical_obs, rd_mpc, rd_fid_mpc=rd_fid_mpc,
        rd_scaling=rd_scaling, paper_tag=paper_tag,
        rd_fid_conventions=rd_fid_conventions,
    )
    result *= scaling

    if sanity_check:
        # Final check: result should be positive and finite
        if not (np.isfinite(result) and result > 0):
            raise ValueError(f"Sanity check failed: {observable}(z={z}) = {result} is not finite positive")

    return result


def lens_time_delay_scaling(z_lens: float, z_source: float, delta_m: float, **kwargs) -> float:
    """Simplified lensing geometry scaling using angular diameter distances."""
    da_l = angular_diameter_distance(np.array([z_lens]), delta_m=delta_m, **kwargs)[0]
    da_s = angular_diameter_distance(np.array([z_source]), delta_m=delta_m, **kwargs)[0]
    da_ls = comoving_distance(np.array([z_source]), delta_m=delta_m, **kwargs)[0] - comoving_distance(np.array([z_lens]), delta_m=delta_m, **kwargs)[0]
    if da_l <= 0 or da_s <= 0 or da_ls <= 0:
        raise ValueError("Distances must be positive")
    return da_l * da_ls / da_s
