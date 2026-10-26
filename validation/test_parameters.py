"""M1: parameter builder and validator (model.md, Section 3.6 derived table)."""

import pytest


def test_derived_table_matches_spec():
    """rho_p 1161.6, Vu0 0.12060, Vtank 2.3262, K_F 0.071, q_p_ref 246.47, ms0 39.47, Vs 0.6866,
    q_r_req 2.788, C_reg 5.900e-6, a_fix 0.2311, M0 12503.1."""
    pytest.skip("M1: not implemented")


def test_missing_input_raises():
    pytest.skip("M1: not implemented")


def test_specified_mass_mode_keeps_ms0_and_c_reg():
    pytest.skip("M1: not implemented")


def test_a0_by_scenario():
    """a0 = 0 regulated, a_fix fixed orifice."""
    pytest.skip("M1: not implemented")


def test_tcut_dt_mismatch_raises():
    pytest.skip("M1: not implemented")
