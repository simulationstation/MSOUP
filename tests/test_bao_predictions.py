"""Tests for BAO observable predictions (DV/rd, DM/rd, DH/rd)."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from msoup_purist_closure.observables import bao_dv, bao_dm, bao_dh, bao_predict
from msoup_purist_closure.residuals import compute_bao_residuals


class TestBaoPredict:
    """Tests for bao_predict function supporting all BAO observables."""

    # Cosmology kwargs for standard LCDM
    cosmo_kwargs = {
        "h_early": 67.0,
        "omega_m0": 0.315,
        "omega_L0": 0.685,
        "c_km_s": 299792.458,
    }
    rd_mpc = 147.09  # Planck 2018 sound horizon

    def test_bao_predict_dv_rd_returns_finite(self):
        """DV/rd prediction returns finite positive value."""
        result = bao_predict(0.5, "DV/rd", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        assert np.isfinite(result)
        assert result > 0

    def test_bao_predict_dm_rd_returns_finite(self):
        """DM/rd prediction returns finite positive value."""
        result = bao_predict(0.5, "DM/rd", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        assert np.isfinite(result)
        assert result > 0

    def test_bao_predict_dh_rd_returns_finite(self):
        """DH/rd prediction returns finite positive value."""
        result = bao_predict(0.5, "DH/rd", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        assert np.isfinite(result)
        assert result > 0

    def test_bao_predict_case_insensitive(self):
        """Observable names are case-insensitive."""
        result1 = bao_predict(0.5, "dv/rd", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        result2 = bao_predict(0.5, "DV/RD", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        result3 = bao_predict(0.5, "Dv/Rd", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        assert result1 == pytest.approx(result2)
        assert result2 == pytest.approx(result3)

    def test_bao_predict_unknown_observable_raises(self):
        """Unknown observable type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown BAO observable"):
            bao_predict(0.5, "unknown_obs", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)

    def test_bao_predict_all_observables_at_multiple_redshifts(self):
        """All observables return finite values at multiple redshifts."""
        for z in [0.1, 0.5, 1.0, 2.0]:
            for obs in ["DV/rd", "DM/rd", "DH/rd"]:
                result = bao_predict(z, obs, delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
                assert np.isfinite(result), f"Failed for {obs} at z={z}"
                assert result > 0, f"Failed for {obs} at z={z}"

    def test_dh_increases_with_z_at_high_z(self):
        """DH/rd should increase at high z as H(z) increases (c/H decreases but rd is constant)."""
        # Actually DH = c/H, so as H increases, DH decreases
        dh_low = bao_predict(0.1, "DH/rd", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        dh_high = bao_predict(2.0, "DH/rd", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        assert dh_low > dh_high  # DH decreases with z

    def test_dm_increases_with_z(self):
        """DM/rd should increase monotonically with z."""
        dm_low = bao_predict(0.1, "DM/rd", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        dm_mid = bao_predict(0.5, "DM/rd", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        dm_high = bao_predict(2.0, "DM/rd", delta_m=0.0, rd_mpc=self.rd_mpc, **self.cosmo_kwargs)
        assert dm_low < dm_mid < dm_high


class TestBaoUnderlyingFunctions:
    """Tests for underlying bao_dv, bao_dm, bao_dh functions."""

    cosmo_kwargs = {
        "h_early": 67.0,
        "omega_m0": 0.315,
        "omega_L0": 0.685,
        "c_km_s": 299792.458,
    }

    def test_bao_dv_returns_array(self):
        """bao_dv returns array of same shape as input."""
        z = np.array([0.1, 0.5, 1.0])
        result = bao_dv(z, delta_m=0.0, **self.cosmo_kwargs)
        assert result.shape == z.shape
        assert np.all(np.isfinite(result))

    def test_bao_dm_returns_array(self):
        """bao_dm returns array of same shape as input."""
        z = np.array([0.1, 0.5, 1.0])
        result = bao_dm(z, delta_m=0.0, **self.cosmo_kwargs)
        assert result.shape == z.shape
        assert np.all(np.isfinite(result))

    def test_bao_dh_returns_array(self):
        """bao_dh returns array of same shape as input."""
        z = np.array([0.1, 0.5, 1.0])
        result = bao_dh(z, delta_m=0.0, **self.cosmo_kwargs)
        assert result.shape == z.shape
        assert np.all(np.isfinite(result))


class TestMixedBaoResiduals:
    """Tests for BAO chi-square with mixed observables."""

    def test_mixed_observable_chi2_finite(self):
        """Mixed DV/rd, DM/rd, DH/rd data produces finite chi2."""
        # Simulate data with all three observable types
        z_obs = np.array([0.15, 0.38, 0.38, 0.51, 0.51])
        obs_values = np.array([4.47, 10.23, 25.0, 13.36, 22.33])
        pred_values = np.array([4.5, 10.3, 24.8, 13.4, 22.4])  # close predictions
        sigma = np.array([0.17, 0.17, 0.76, 0.21, 0.58])

        result = compute_bao_residuals(z_obs, obs_values, pred_values, sigma, "BAO_mixed")
        assert result.status == "OK"
        assert np.isfinite(result.chi2)
        assert result.n == 5
        assert result.dof == 5

    def test_chi2_includes_all_rows(self):
        """Chi-square computation includes all N rows (no silent skipping)."""
        # Create data with 10 points
        n = 10
        z_obs = np.linspace(0.2, 2.0, n)
        obs_values = np.ones(n) * 10.0
        pred_values = np.ones(n) * 10.0  # perfect match
        sigma = np.ones(n) * 0.1

        result = compute_bao_residuals(z_obs, obs_values, pred_values, sigma, "test")
        assert result.n == n
        assert result.dof == n
        assert result.chi2 == pytest.approx(0.0)


class TestBaoDataValidation:
    """Tests for BAO data loading and validation."""

    def test_sigma_only_csv_not_testable(self):
        """A BAO CSV with only sigma (no value) should fail validation."""
        from msoup_purist_closure.run import load_bao_observed_data
        from msoup_purist_closure.config import ProbeConfig

        # Create a temporary CSV without 'value' column
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("z,sigma,probe,name\n")
            f.write("0.38,0.009,sdss_bao,BOSS_LowZ\n")
            f.write("0.51,0.008,sdss_bao,BOSS_CMASS\n")
            temp_path = f.name

        try:
            probe = ProbeConfig(name="test_bao", path=temp_path, type="bao")
            df, status = load_bao_observed_data(probe)

            # Should fail because 'value' column is missing
            assert df is None
            assert "missing column" in status.lower() or "value" in status.lower()
        finally:
            Path(temp_path).unlink()

    def test_valid_bao_csv_loads_correctly(self):
        """A valid BAO CSV with observable, value, sigma loads correctly."""
        from msoup_purist_closure.run import load_bao_observed_data
        from msoup_purist_closure.config import ProbeConfig

        # Create a valid temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("z,observable,value,sigma,tracer,paper_tag\n")
            f.write("0.15,DV/rd,4.47,0.17,MGS,Ross2015\n")
            f.write("0.38,DM/rd,10.23,0.17,BOSS,Alam2017\n")
            f.write("0.38,DH/rd,25.0,0.76,BOSS,Alam2017\n")
            temp_path = f.name

        try:
            probe = ProbeConfig(name="test_bao", path=temp_path, type="bao")
            df, status = load_bao_observed_data(probe)

            assert df is not None
            assert status == "OK"
            assert len(df) == 3
            assert 'observable' in df.columns
            assert 'value' in df.columns
            assert 'sigma' in df.columns
        finally:
            Path(temp_path).unlink()
