"""M4/M5: driver bookkeeping and conservation (numerics.md Section 8.5; verification_studies.md Section 10)."""

import pytest


def test_cutoff_lands_on_step():
    pytest.skip("M4: not implemented")


def test_cutoff_bookkeeping():
    """Thrust and qp zero after t_cut; mp, ms, mu, a constant in coast; cutoff state in both segments."""
    pytest.skip("M4: not implemented")


def test_helium_balance():
    """Relative drift of ms + mu <= 1e-10 (FE, RK4, AB3), <= 1e-8 (BE)."""
    pytest.skip("M4: not implemented")


def test_coupling_isolation():
    """Different Cd / gamma0, same gas inputs -> identical pressure and propellant histories."""
    pytest.skip("M4: not implemented")


def test_hermite_apogee_on_ballistic():
    pytest.skip("M4: not implemented")


def test_ab3_restarts_at_cutoff():
    pytest.skip("M5: not implemented")


def test_be_reports_residual_and_counts():
    pytest.skip("M5: not implemented")
