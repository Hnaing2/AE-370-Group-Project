"""M2: constitutive model and RHS (model.md Sections 5-6; verification_studies.md Section 10)."""

import pytest


def test_reference_force_and_flow_at_pset():
    """F_eng 568,000 N, T_ax ~533,758 N, qp ~246.47 kg/s at Pu = Pset (Section 5.1)."""
    pytest.skip("M2: not implemented")


def test_force_flow_balance():
    """F_eng/(Isp g0) = qp, T_ax/F_eng = cos(delta), M_dot = -qp."""
    pytest.skip("M2: not implemented")


def test_flow_function_branches():
    """Phi = 1 for r <= r_*, continuous value and slope at r_*, -> 0 as r -> 1, 0 for r >= 1."""
    pytest.skip("M2: not implemented")


def test_ullage_pressure_derivative():
    """dPu/dt = (R Tu/Vu) qr - (Pu/Vu)(qp/rho_p) matches the chain rule through the RHS."""
    pytest.skip("M2: not implemented")


def test_coast_freezes_gas_and_valve_states():
    pytest.skip("M2: not implemented")


def test_fixed_orifice_valve_rate_zero():
    pytest.skip("M2: not implemented")


def test_rhs_is_pure():
    """RHS does not modify its input and is deterministic."""
    pytest.skip("M2: not implemented")


def test_invalid_state_raises():
    """V <= 0, Vu <= 0, nonpositive masses, or a outside [0, 1] raise."""
    pytest.skip("M2: not implemented")
