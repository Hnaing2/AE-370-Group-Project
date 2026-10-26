"""M6: independent physical benchmarks and Jacobian checks (verification_studies.md Section 10; numerics.md Section 9)."""

import pytest


def test_ballistic_trajectory():
    """No thrust/drag, constant g: closed-form x, h, V, gamma; checks the gamma equation."""
    pytest.skip("M6: not implemented")


def test_vertical_constant_force_burn():
    """gamma = pi/2, fixed Pu, no drag, constant g: V = V0 + Isp g0 cos(delta) ln(M0/M) - g0 t."""
    pytest.skip("M6: not implemented")


def test_constant_valve_command():
    """a = a_bar + (a0 - a_bar) exp(-t/tau_v)."""
    pytest.skip("M6: not implemented")


def test_passive_isothermal_ullage():
    """qr = 0: Pu Vu = mu0 R Tu."""
    pytest.skip("M6: not implemented")


def test_fd_jacobian_matches_analytic_choked():
    """Several smooth states, saturated and unsaturated command branches, more than one step size."""
    pytest.skip("M6: not implemented")


def test_fixed_orifice_jacobian_last_row_zero():
    pytest.skip("M6: not implemented")
