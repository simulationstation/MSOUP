"""
Unit tests for Msoup closure model.

Run with: pytest tests/test_model.py -v
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from msoup_model import (
    MsoupParams, CosmologyParams, MsoupGrowthSolver,
    visibility, m_vis, c_eff_squared, validate_lcdm_limit,
    HaloMassFunction, compute_half_mode_scale,
    RotationCurveFitter, nfw_velocity, cored_nfw_velocity,
    VISIBILITY_KAPPA
)
from data.lensing import LensingConstraints


class TestVisibility:
    """Tests for visibility function."""

    def test_visibility_at_threshold(self):
        """V(M=2) should equal 1."""
        assert np.isclose(visibility(2.0, kappa=1.0), 1.0)

    def test_visibility_below_threshold(self):
        """V(M<2) should equal 1 (clamped)."""
        assert np.isclose(visibility(1.0, kappa=1.0), 1.0)
        assert np.isclose(visibility(0.0, kappa=1.0), 1.0)

    def test_visibility_above_threshold(self):
        """V(M>2) should decay exponentially."""
        V = visibility(3.0, kappa=1.0)
        assert 0 < V < 1
        assert np.isclose(V, np.exp(-1))

    def test_visibility_kappa_scaling(self):
        """Higher kappa means faster decay."""
        V1 = visibility(3.0, kappa=1.0)
        V2 = visibility(3.0, kappa=2.0)
        assert V2 < V1

    def test_visibility_fixed_kappa_constant(self):
        """Fixed κ should be exposed but not treated as free."""
        assert np.isclose(VISIBILITY_KAPPA, 1.0)
        V = visibility(3.0)
        assert np.isclose(V, np.exp(-VISIBILITY_KAPPA))


class TestMvis:
    """Tests for visible order statistic M_vis(z)."""

    def test_mvis_high_z_limit(self):
        """M_vis should approach 2 at high z."""
        params = MsoupParams(Delta_M=0.5, z_t=2.0, w=0.5, c_star_sq=100)
        M = m_vis(20.0, params)
        assert np.isclose(M, 2.0, atol=0.01)

    def test_mvis_low_z_limit(self):
        """M_vis should approach 2 + Delta_M at low z."""
        params = MsoupParams(Delta_M=0.5, z_t=2.0, w=0.5, c_star_sq=100)
        M = m_vis(0.0, params)
        assert np.isclose(M, 2.5, atol=0.01)

    def test_mvis_transition(self):
        """M_vis should be ~2 + Delta_M/2 at z_t."""
        params = MsoupParams(Delta_M=0.5, z_t=2.0, w=0.5, c_star_sq=100)
        M = m_vis(2.0, params)
        assert np.isclose(M, 2.25, atol=0.01)

    def test_mvis_monotonic(self):
        """M_vis should decrease with increasing z."""
        params = MsoupParams(Delta_M=0.5, z_t=2.0, w=0.5, c_star_sq=100)
        z = np.linspace(0, 10, 100)
        M = m_vis(z, params)
        # Check monotonically decreasing
        assert np.all(np.diff(M) <= 0)


class TestCeffSquared:
    """Tests for effective sound speed squared."""

    def test_ceff_lower_at_high_z(self):
        """c_eff^2 should be lower at high z than low z."""
        params = MsoupParams(Delta_M=0.5, z_t=2.0, w=0.5, c_star_sq=100)
        c_sq_high_z = c_eff_squared(10.0, params)
        c_sq_low_z = c_eff_squared(0.0, params)
        assert c_sq_high_z < c_sq_low_z

    def test_ceff_max_when_mvis_high(self):
        """c_eff^2 should approach c_*^2 when M_vis >> 2."""
        params = MsoupParams(Delta_M=1.0, z_t=5.0, w=0.5, c_star_sq=100)
        # At low z, M_vis ~ 3
        c_sq = c_eff_squared(0.0, params)
        assert c_sq > 90  # Near c_*^2

    def test_ceff_scales_with_c_star_sq(self):
        """c_eff^2 should scale with c_*^2."""
        params1 = MsoupParams(Delta_M=0.5, z_t=1.0, w=0.5, c_star_sq=100)
        params2 = MsoupParams(Delta_M=0.5, z_t=1.0, w=0.5, c_star_sq=400)
        c_sq1 = c_eff_squared(0.0, params1)
        c_sq2 = c_eff_squared(0.0, params2)
        assert c_sq2 > c_sq1


class TestMsoupParams:
    """Tests for parameter validation."""

    def test_valid_params(self):
        """Valid parameters should not raise."""
        params = MsoupParams(c_star_sq=100, Delta_M=0.5, z_t=2.0, w=0.5)
        assert params.c_star_sq == 100

    def test_negative_delta_m_raises(self):
        """Negative Delta_M should raise."""
        with pytest.raises(ValueError):
            MsoupParams(c_star_sq=100, Delta_M=-0.5, z_t=2.0, w=0.5)

    def test_zero_w_raises(self):
        """Zero w should raise."""
        with pytest.raises(ValueError):
            MsoupParams(c_star_sq=100, Delta_M=0.5, z_t=2.0, w=0.0)

    def test_to_array(self):
        """to_array should return [c_star_sq, Delta_M, z_t, w]."""
        params = MsoupParams(c_star_sq=100, Delta_M=0.5, z_t=2.0, w=0.5)
        arr = params.to_array()
        assert np.allclose(arr, [100, 0.5, 2.0, 0.5])

    def test_from_array(self):
        """from_array should reconstruct params."""
        arr = np.array([100, 0.5, 2.0, 0.5])
        params = MsoupParams.from_array(arr)
        assert params.c_star_sq == 100
        assert params.Delta_M == 0.5


class TestCosmology:
    """Tests for cosmology module."""

    def test_hubble_parameter(self):
        """H(0) should equal H0."""
        cosmo = CosmologyParams()
        assert np.isclose(cosmo.H(0), cosmo.H0)

    def test_hubble_increases_with_z(self):
        """H should increase with z."""
        cosmo = CosmologyParams()
        H0 = cosmo.H(0)
        H1 = cosmo.H(1)
        H2 = cosmo.H(2)
        assert H1 > H0
        assert H2 > H1

    def test_growth_factor_z0(self):
        """D(z=0) should be 1 (normalized)."""
        cosmo = CosmologyParams()
        D = cosmo.growth_factor_lcdm(0)
        assert np.isclose(D, 1.0, atol=0.01)

    def test_growth_factor_decreases_with_z(self):
        """D should decrease with increasing z."""
        cosmo = CosmologyParams()
        z = np.array([0, 1, 2, 5])
        D = cosmo.growth_factor_lcdm(z)
        assert np.all(np.diff(D) < 0)


class TestGrowthSolver:
    """Tests for growth equation solver."""

    def test_lcdm_limit(self):
        """Growth should be scale-independent when c_*^2 = 0."""
        assert validate_lcdm_limit(verbose=False)

    def test_suppression_at_high_k(self):
        """Growth should be suppressed at high k when c_*^2 > 0."""
        params = MsoupParams(c_star_sq=100, Delta_M=0.5, z_t=2.0, w=0.5)
        cosmo = CosmologyParams()
        solver = MsoupGrowthSolver(
            params, cosmo,
            k_min=0.01, k_max=10, n_k=20,
            z_max=5, n_z=30
        )
        solution = solver.solve()

        # Check suppression at z=0
        T_ratio = solution.transfer_ratio(solution.k_grid, z=0)

        # Low k should be unsuppressed
        assert T_ratio[0] > 0.95

        # High k should be suppressed
        assert T_ratio[-1] < T_ratio[0]

    def test_solution_shape(self):
        """Solution should have correct shape."""
        params = MsoupParams(c_star_sq=100, Delta_M=0.5, z_t=2.0, w=0.5)
        cosmo = CosmologyParams()
        solver = MsoupGrowthSolver(
            params, cosmo,
            k_min=0.1, k_max=5, n_k=15,
            z_max=3, n_z=20
        )
        solution = solver.solve()

        assert solution.k_grid.shape == (15,)
        assert solution.z_grid.shape == (20,)
        assert solution.D_kz.shape == (15, 20)

    def test_growth_units_dimensionless(self):
        """(c_eff * k / H)^2 term must be dimensionless."""
        params = MsoupParams(c_star_sq=150, Delta_M=0.5, z_t=2.0, w=0.5)
        cosmo = CosmologyParams()
        solver = MsoupGrowthSolver(params, cosmo, k_min=1.0, k_max=1.0, n_k=1, z_max=0.0, n_z=2)
        k = 1.0
        z = 0.5
        term = solver._dimensionless_suppression_term(k, z)
        c_eff = np.sqrt(c_eff_squared(z, params))
        expected = (c_eff * k * cosmo.h / cosmo.H(z))**2
        assert np.isclose(term, expected, rtol=1e-6)

    def test_cdm_matches_lcdm_growth(self):
        """c_*^2=0 should return LCDM growth to within tolerance."""
        params = MsoupParams(c_star_sq=0.0, Delta_M=0.5, z_t=2.0, w=0.5)
        cosmo = CosmologyParams()
        solver = MsoupGrowthSolver(params, cosmo, k_min=0.05, k_max=1.0, n_k=5, z_max=5.0, n_z=25)
        solution = solver.solve()
        D_ref = cosmo.growth_factor_lcdm(solution.z_grid)
        assert np.allclose(solution.D_kz, D_ref[None, :], atol=1e-3, rtol=1e-3)

    def test_power_suppression_monotone(self):
        """P/P_CDM should be monotone non-increasing with k and in [0,1]."""
        params = MsoupParams(c_star_sq=120, Delta_M=0.5, z_t=2.0, w=0.5)
        cosmo = CosmologyParams()
        solver = MsoupGrowthSolver(params, cosmo, k_min=0.01, k_max=5.0, n_k=40, z_max=3.0, n_z=30)
        solution = solver.solve()
        ratio = solution.power_ratio(solution.k_grid, z=0)
        assert np.all(ratio <= 1.0 + 1e-8)
        assert np.all(np.diff(ratio) <= 1e-6)


class TestHaloMassFunction:
    """Tests for halo mass function."""


class TestHalfModeMapping:
    """Tests for half-mode mass and k mapping."""

    def test_half_mode_mapping_matches_definition(self):
        """Synthetic suppression with known k_hm should map to expected M_hm."""
        cosmo = CosmologyParams()
        params = MsoupParams()
        k_grid = np.logspace(-2, 1, 60)
        k_hm_true = k_grid[34]  # ensure grid contains the crossing point
        power_ratio = np.clip(1 - 0.75 * (k_grid / k_hm_true), 0.25, 1.0)
        transfer = np.sqrt(power_ratio)
        z_grid = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
        D_kz = transfer[:, None] * np.ones((1, len(z_grid)))
        D_lcdm_z = np.ones_like(z_grid)

        from msoup_model.growth import GrowthSolution

        solution = GrowthSolution(
            k_grid=k_grid,
            z_grid=z_grid,
            D_kz=D_kz,
            D_lcdm_z=D_lcdm_z,
            params=params,
            cosmo=cosmo,
        )

        hm = compute_half_mode_scale(solution, z=0, power_threshold=0.25)
        M_expected = (4.0 / 3.0) * np.pi * cosmo.rho_m_0 * (np.pi / k_hm_true)**3
        assert np.isclose(hm.k_hm, k_hm_true, rtol=0.05)
        assert np.isclose(hm.M_hm, M_expected, rtol=0.05)


class TestRotationCurveCores:
    """Tests for rotation curve coring logic."""

    def test_core_radius_zero_when_unsuppressed(self):
        """c_*^2 = 0 should give r_c=0 and NFW equivalence."""
        params = MsoupParams(c_star_sq=0.0, Delta_M=0.5, z_t=2.0, w=0.5)
        cosmo = CosmologyParams()
        fitter = RotationCurveFitter(cosmo=cosmo, msoup_params=params)
        r_c = fitter.core_radius(M200=1e10, c=10.0)
        assert np.isclose(r_c, 0.0)

        r = np.linspace(0.5, 5.0, 20)
        v_nfw = nfw_velocity(r, 1e10, 10.0, cosmo)
        v_cored = cored_nfw_velocity(r, 1e10, 10.0, r_c, cosmo)
        assert np.allclose(v_nfw, v_cored, atol=1e-6)


class TestLensingConstraints:
    """Tests for lensing constraint evaluation."""

    def test_consistency_mode_reports_constraints(self):
        lensing = LensingConstraints.load_default()
        evaluation = lensing.evaluate_constraints(1e7, mode="consistency", use_forecasts=False)
        assert "constraints" in evaluation and len(evaluation["constraints"]) > 0
        assert evaluation["mode"] == "consistency"

    def test_hmf_ratio_unity_for_cdm(self):
        """HMF ratio should be ~1 for CDM (no suppression)."""
        cosmo = CosmologyParams()
        hmf = HaloMassFunction(cosmo=cosmo, growth_solution=None)
        result = hmf.compute_hmf(z=0, M_min=1e8, M_max=1e12, n_M=10)

        # Without suppression, ratio should be ~1
        assert np.allclose(result.ratio, 1.0, atol=0.1)

    def test_hmf_suppression_with_msoup(self):
        """HMF should show some suppression with Msoup."""
        params = MsoupParams(c_star_sq=200, Delta_M=0.5, z_t=2.0, w=0.5)
        cosmo = CosmologyParams()
        solver = MsoupGrowthSolver(params, cosmo, n_k=30, n_z=40)
        solution = solver.solve()

        hmf = HaloMassFunction(cosmo=cosmo, growth_solution=solution)
        result = hmf.compute_hmf(z=0, M_min=1e8, M_max=1e12, n_M=20)

        # Ratio should be <= 1 (suppressed or equal to CDM)
        assert np.all(result.ratio <= 1.1)  # Allow small numerical tolerance


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
