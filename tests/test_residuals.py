"""Tests for msoup_purist_closure.residuals module."""
import numpy as np
import pytest

from msoup_purist_closure.residuals import (
    ProbeResidualResult,
    compute_bao_residuals,
    compute_chi2_covariance,
    compute_chi2_diagonal,
    compute_sn_residuals,
    compute_td_residuals,
    marginalize_offset_covariance,
    marginalize_offset_diagonal,
)


class TestChi2Diagonal:
    """Tests for diagonal chi-square computation."""

    def test_perfect_fit_zero_chi2(self):
        resid = np.zeros(10)
        sigma = np.ones(10)
        chi2, dof = compute_chi2_diagonal(resid, sigma)
        assert chi2 == 0.0
        assert dof == 10

    def test_one_sigma_deviation(self):
        # Each point is 1 sigma off -> chi2 = N
        resid = np.ones(10)
        sigma = np.ones(10)
        chi2, dof = compute_chi2_diagonal(resid, sigma)
        assert chi2 == pytest.approx(10.0)
        assert dof == 10

    def test_handles_nan(self):
        resid = np.array([1.0, np.nan, 1.0])
        sigma = np.array([1.0, 1.0, 1.0])
        chi2, dof = compute_chi2_diagonal(resid, sigma)
        assert chi2 == pytest.approx(2.0)
        assert dof == 2

    def test_handles_zero_sigma(self):
        resid = np.array([1.0, 1.0, 1.0])
        sigma = np.array([1.0, 0.0, 1.0])
        chi2, dof = compute_chi2_diagonal(resid, sigma)
        assert chi2 == pytest.approx(2.0)
        assert dof == 2


class TestChi2Covariance:
    """Tests for full covariance chi-square computation."""

    def test_diagonal_cov_matches_diagonal(self):
        resid = np.array([1.0, 2.0, 3.0])
        sigma = np.array([1.0, 2.0, 3.0])
        C = np.diag(sigma**2)
        chi2_cov, method = compute_chi2_covariance(resid, C)
        chi2_diag, _ = compute_chi2_diagonal(resid, sigma)
        assert chi2_cov == pytest.approx(chi2_diag, rel=1e-6)
        assert method == "full_cov"

    def test_correlated_case(self):
        # Simple 2x2 positive definite matrix
        C = np.array([[1.0, 0.5], [0.5, 1.0]])
        resid = np.array([1.0, 1.0])
        chi2, method = compute_chi2_covariance(resid, C)
        # C^-1 for [[1, 0.5], [0.5, 1]] is [[4/3, -2/3], [-2/3, 4/3]]
        # chi2 = r.T @ C^-1 @ r = [1,1] @ [[4/3, -2/3], [-2/3, 4/3]] @ [1,1] = 4/3
        expected = 4.0 / 3.0
        assert chi2 == pytest.approx(expected, rel=1e-6)
        assert method == "full_cov"

    def test_size_mismatch_returns_fallback(self):
        resid = np.array([1.0, 2.0])
        C = np.eye(3)  # wrong size
        chi2, method = compute_chi2_covariance(resid, C)
        assert np.isnan(chi2)
        assert method == "diag_fallback"


class TestMarginalization:
    """Tests for SN offset marginalization."""

    def test_diagonal_marginalization_constant_offset(self):
        # Residuals with constant offset
        offset = 0.5
        base_resid = np.zeros(10)
        resid = base_resid + offset
        sigma = np.ones(10)
        adjusted, b_hat = marginalize_offset_diagonal(resid, sigma)
        assert b_hat == pytest.approx(offset)
        assert np.allclose(adjusted, 0.0, atol=1e-10)

    def test_diagonal_marginalization_weights(self):
        # Different weights should affect b_hat
        resid = np.array([1.0, 1.0, 2.0])
        sigma = np.array([1.0, 1.0, 10.0])  # third point has low weight
        adjusted, b_hat = marginalize_offset_diagonal(resid, sigma)
        # b_hat should be closer to 1.0 than 2.0 due to weights
        assert b_hat < 1.5

    def test_covariance_marginalization_matches_diagonal_for_diag_cov(self):
        resid = np.array([1.5, 1.5, 1.5])
        sigma = np.ones(3)
        C = np.diag(sigma**2)
        adj_diag, b_diag = marginalize_offset_diagonal(resid, sigma)
        adj_cov, b_cov = marginalize_offset_covariance(resid, C)
        assert b_cov == pytest.approx(b_diag, rel=1e-6)
        assert np.allclose(adj_cov, adj_diag, rtol=1e-6)


class TestSNResiduals:
    """Tests for SN residual computation."""

    def test_sn_residuals_basic(self):
        z_obs = np.array([0.1, 0.5, 1.0])
        mu_obs = np.array([40.0, 43.0, 44.5])
        mu_pred = np.array([40.0, 43.0, 44.5])  # perfect fit
        sigma = np.ones(3)

        result = compute_sn_residuals(z_obs, mu_obs, mu_pred, sigma)
        assert result.status == "OK"
        assert result.n == 3
        assert result.chi2 == pytest.approx(0.0, abs=1e-6)
        assert result.method == "diagonal"
        assert result.marginalized_offset is True

    def test_sn_residuals_with_offset(self):
        z_obs = np.array([0.1, 0.5, 1.0])
        mu_obs = np.array([40.5, 43.5, 45.0])  # all offset by 0.5
        mu_pred = np.array([40.0, 43.0, 44.5])
        sigma = np.ones(3)

        result = compute_sn_residuals(z_obs, mu_obs, mu_pred, sigma, marginalize_offset=True)
        assert result.b_hat == pytest.approx(0.5)
        # After marginalization, residuals should be ~0
        assert result.chi2 == pytest.approx(0.0, abs=1e-6)

    def test_sn_residuals_with_covariance(self):
        z_obs = np.array([0.1, 0.5, 1.0])
        mu_obs = np.array([40.0, 43.0, 44.5])
        mu_pred = np.array([40.0, 43.0, 44.5])
        sigma = np.ones(3)
        C = np.eye(3)

        result = compute_sn_residuals(z_obs, mu_obs, mu_pred, sigma, C=C, marginalize_offset=True)
        assert result.status == "OK"
        assert result.method == "full_cov"

    def test_sn_residuals_dof_reduction(self):
        # DOF should be N-1 when marginalization is used
        z_obs = np.array([0.1, 0.5, 1.0])
        mu_obs = np.array([40.1, 43.1, 44.6])
        mu_pred = np.array([40.0, 43.0, 44.5])
        sigma = np.ones(3)

        result = compute_sn_residuals(z_obs, mu_obs, mu_pred, sigma, marginalize_offset=True)
        assert result.dof == 2  # N=3, minus 1 for marginalized offset


class TestBAOResiduals:
    """Tests for BAO residual computation."""

    def test_bao_residuals_basic(self):
        z_obs = np.array([0.5, 1.0, 1.5])
        obs_values = np.array([10.0, 15.0, 18.0])
        pred_values = np.array([10.0, 15.0, 18.0])
        sigma = np.ones(3)

        result = compute_bao_residuals(z_obs, obs_values, pred_values, sigma, "DV_over_rd")
        assert result.status == "OK"
        assert result.n == 3
        assert result.chi2 == pytest.approx(0.0)
        assert result.marginalized_offset is False
        assert result.obs_column == "DV_over_rd"

    def test_bao_residuals_no_marginalization(self):
        # BAO should never marginalize offset
        z_obs = np.array([0.5, 1.0])
        obs_values = np.array([10.5, 15.5])  # offset by 0.5
        pred_values = np.array([10.0, 15.0])
        sigma = np.ones(2)

        result = compute_bao_residuals(z_obs, obs_values, pred_values, sigma, "DV_over_rd")
        assert result.marginalized_offset is False
        # chi2 should be 0.5 (0.5^2 / 1^2 * 2 = 0.5)
        assert result.chi2 == pytest.approx(0.5)


class TestTDResiduals:
    """Tests for time-delay residual computation."""

    def test_td_residuals_basic(self):
        z_lens = np.array([0.3, 0.5, 0.7])
        ddt_obs = np.array([2000.0, 3000.0, 4000.0])
        ddt_pred = np.array([2000.0, 3000.0, 4000.0])
        sigma = np.array([100.0, 150.0, 200.0])

        result = compute_td_residuals(z_lens, ddt_obs, ddt_pred, sigma)
        assert result.status == "OK"
        assert result.n == 3
        assert result.chi2 == pytest.approx(0.0)
        assert result.marginalized_offset is False
        assert result.obs_column == "D_dt"

    def test_td_residuals_chi2_calculation(self):
        z_lens = np.array([0.3, 0.5])
        ddt_obs = np.array([2100.0, 3150.0])  # 100 and 150 off
        ddt_pred = np.array([2000.0, 3000.0])
        sigma = np.array([100.0, 150.0])

        result = compute_td_residuals(z_lens, ddt_obs, ddt_pred, sigma)
        # chi2 = (100/100)^2 + (150/150)^2 = 1 + 1 = 2
        assert result.chi2 == pytest.approx(2.0)
        assert result.dof == 2


class TestProbeResidualResult:
    """Tests for the result dataclass."""

    def test_default_values(self):
        result = ProbeResidualResult(status="OK")
        assert result.n == 0
        assert np.isnan(result.chi2)
        assert result.dof == 0
        assert result.method == "none"
        assert result.marginalized_offset is False

    def test_not_testable_status(self):
        result = ProbeResidualResult(status="NOT_TESTABLE", reason="Missing column: mu_obs")
        assert result.status == "NOT_TESTABLE"
        assert result.reason == "Missing column: mu_obs"
