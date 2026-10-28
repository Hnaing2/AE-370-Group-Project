"""Constitutive model and RHS."""

import dataclasses

import numpy as np
import pytest

from src.parameters import FIXED_ORIFICE_CASE, REFERENCE_CASE, build_initial_state, build_params
from src.physics import (
    IA, IGAMMA, IH, IMP, IMS, IMU, IV, IX,
    PhysicalModelInvalid,
    critical_pressure_ratio,
    derived_quantities,
    flow_function,
    jacobian,
    rhs,
    standard_atmosphere_1976,
)


@pytest.fixture(scope="module")
def ref():
    return build_params(REFERENCE_CASE)


@pytest.fixture(scope="module")
def fixed():
    return build_params(FIXED_ORIFICE_CASE)


def state_at_pset(p, a=0.5):
    """Reference state with mu chosen so that Pu = Pset exactly, valve partly open."""
    u = build_initial_state(p)
    Vu = p.Vtank - u[IMP] / p.rho_p
    u[IMU] = p.Pset * Vu / (p.R_he * p.Tu)
    u[IA] = a
    return u


def test_reference_force_and_flow_at_pset(ref):
    """F_eng 568,000 N, T_ax ~533,758 N, qp ~246.47 kg/s at Pu = Pset."""
    q = derived_quantities(state_at_pset(ref), ref, "powered")
    assert q["Pu"] == pytest.approx(ref.Pset, rel=1e-14)
    assert q["F_eng"] == pytest.approx(568000.0, rel=1e-12)
    assert round(q["T_ax"]) == 533758
    assert round(q["qp"], 2) == 246.47
    assert round(q["qp"] * ref.t_cut) == 1849


def test_force_flow_balance(ref):
    """F_eng/(Isp g0) = qp, T_ax/F_eng = cos(delta), M_dot = -qp."""
    u = state_at_pset(ref)
    q = derived_quantities(u, ref, "powered")
    du = rhs(0.0, u, ref, "powered")
    assert q["F_eng"] / (ref.Isp * ref.g0) == pytest.approx(q["qp"], rel=1e-14)
    assert q["T_ax"] / q["F_eng"] == pytest.approx(np.cos(ref.delta), rel=1e-14)
    assert du[IMP] + du[IMS] + du[IMU] == pytest.approx(-q["qp"], rel=1e-14)
    assert du[IMS] == -du[IMU]  # helium is conserved exactly


def test_flow_function_branches(ref):
    """Phi = 1 for r <= r_*, continuous value and slope at r_*, -> 0 as r -> 1, 0 for r >= 1."""
    g = ref.gamma_he
    rs = critical_pressure_ratio(g)
    assert rs == pytest.approx(0.4871, abs=1e-4)
    for r in (0.0, 0.237, 0.5 * rs, rs):
        assert flow_function(r, g) == 1.0
    eps = 1e-6
    assert flow_function(rs + eps, g) == pytest.approx(1.0, abs=1e-9)
    slope_right = (flow_function(rs + 2 * eps, g) - flow_function(rs + eps, g)) / eps
    assert abs(slope_right) < 1e-4  # left slope is 0; Phi has its maximum at r_*
    assert flow_function(0.8, g) == pytest.approx(0.787, abs=1e-3)
    assert flow_function(1.0 - 1e-10, g) < 1e-3
    for r in (1.0, 1.5):
        assert flow_function(r, g) == 0.0
    with pytest.raises(ValueError):
        flow_function(-0.1, g)


def test_ullage_pressure_derivative(ref):
    """dPu/dt = (R Tu/Vu) qr - (Pu/Vu)(qp/rho_p) matches the chain rule through the RHS."""
    u = state_at_pset(ref)
    q = derived_quantities(u, ref, "powered")
    du = rhs(0.0, u, ref, "powered")
    expected = ref.R_he * ref.Tu / q["Vu"] * q["qr"] - q["Pu"] / q["Vu"] * q["qp"] / ref.rho_p

    eps = 1e-7
    dPu = 0.0
    for i in (IMP, IMU):
        h = eps * u[i]
        up, um = u.copy(), u.copy()
        up[i] += h
        um[i] -= h
        dPu_di = (derived_quantities(up, ref, "powered")["Pu"]
                  - derived_quantities(um, ref, "powered")["Pu"]) / (2 * h)
        dPu += dPu_di * du[i]
    assert dPu == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("case", ["ref", "fixed"])
def test_coast_freezes_gas_and_valve_states(case, request):
    p = request.getfixturevalue(case)
    u = state_at_pset(p)
    q = derived_quantities(u, p, "coast")
    assert q["F_eng"] == q["T_ax"] == q["qp"] == q["qr"] == 0.0
    du = rhs(10.0, u, p, "coast")
    assert np.all(du[IMP:] == 0.0)
    V, gamma = u[IV], u[IGAMMA]
    assert du[IX] == pytest.approx(V * np.cos(gamma))
    assert du[IH] == pytest.approx(V * np.sin(gamma))
    assert du[IV] == pytest.approx(-q["D"] / q["M"] - q["g"] * np.sin(gamma))


def test_fixed_orifice_valve_rate_zero(fixed):
    u = build_initial_state(fixed)
    assert u[IA] == fixed.a_fix
    for Pu_frac in (0.5, 1.0, 1.1):  # command would be saturated or unsaturated if regulated
        v = u.copy()
        Vu = fixed.Vtank - v[IMP] / fixed.rho_p
        v[IMU] = Pu_frac * fixed.Pset * Vu / (fixed.R_he * fixed.Tu)
        assert rhs(0.0, v, fixed, "powered")[IA] == 0.0
    assert np.all(jacobian(0.0, u, fixed, "powered")[IA] == 0.0)


def test_regulator_command(ref):
    """a_cmd = clip(K_p (Pset - Pu), 0, 1) and a_dot = (a_cmd - a)/tau_v."""
    u0 = build_initial_state(ref)  # Pu0 > Pset, valve closed
    q0 = derived_quantities(u0, ref, "powered")
    assert q0["a_cmd"] == 0.0 and q0["qr"] == 0.0 and q0["choked"]
    assert rhs(0.0, u0, ref, "powered")[IA] == 0.0

    u = state_at_pset(ref, a=0.3)
    Vu = ref.Vtank - u[IMP] / ref.rho_p
    u[IMU] = 0.98 * ref.Pset * Vu / (ref.R_he * ref.Tu)  # 2% droop
    q = derived_quantities(u, ref, "powered")
    assert q["a_cmd"] == pytest.approx(ref.kappa * 0.02, rel=1e-9)
    assert rhs(0.0, u, ref, "powered")[IA] == pytest.approx((q["a_cmd"] - 0.3) / ref.tau_v, rel=1e-12)

    u[IMU] *= 0.5  # deep droop saturates the command
    assert derived_quantities(u, ref, "powered")["a_cmd"] == 1.0


def test_pressure_subsystem_independent_of_trajectory(ref):
    """Changing Cd, h, V, or gamma must not change the pressure/propellant rates."""
    u = state_at_pset(ref)
    base = rhs(0.0, u, ref, "powered")[IMP:]
    other = dataclasses.replace(ref, Cd=0.8)
    v = u.copy()
    v[IH], v[IV], v[IGAMMA] = 30000.0, 700.0, 0.9
    assert np.array_equal(rhs(0.0, v, other, "powered")[IMP:], base)


def test_rhs_is_pure(ref):
    """RHS does not modify its input and is deterministic."""
    u = state_at_pset(ref)
    u_copy = u.copy()
    f1 = rhs(1.0, u, ref, "powered")
    f2 = rhs(1.0, u, ref, "powered")
    assert np.array_equal(u, u_copy)
    assert np.array_equal(f1, f2)
    assert f1.shape == (8,) and f1.dtype == np.float64
    assert np.array_equal(rhs(5.0, u, ref, "powered"), f1)  # autonomous


def test_invalid_state_raises(ref):
    """V <= 0, Vu <= 0, nonpositive masses, or a outside [0, 1] raise."""
    u = state_at_pset(ref)
    bad = []
    for i, val in ((IV, 0.0), (IV, -1.0), (IMP, 0.0), (IMS, -1.0), (IMU, 0.0),
                   (IA, -1e-6), (IA, 1.0 + 1e-6), (IH, np.nan), (IV, np.inf)):
        v = u.copy()
        v[i] = val
        bad.append(v)
    v = u.copy()
    v[IMP] = ref.Vtank * ref.rho_p * 1.001  # ullage volume <= 0
    bad.append(v)
    for v in bad:
        for phase in ("powered", "coast"):
            with pytest.raises(PhysicalModelInvalid):
                rhs(0.0, v, ref, phase)
    with pytest.raises(ValueError):
        rhs(0.0, u, ref, "boost")
    # roundoff at the valve bounds is tolerated, not clipped
    for a in (-1e-13, 1.0 + 1e-13):
        v = u.copy()
        v[IA] = a
        assert np.all(np.isfinite(rhs(0.0, v, ref, "powered")))


def test_fd_jacobian_shape_and_structure(ref):
    """Central FD Jacobian: zero x column, helium rows opposite, last row -1/tau_v when saturated."""
    u0 = build_initial_state(ref)  # a0 = 0 on the bound: one-sided difference in a
    J = jacobian(0.0, u0, ref, "powered")
    assert J.shape == (8, 8)
    assert np.all(J[:, IX] == 0.0)
    assert np.array_equal(J[IMS], -J[IMU])
    assert J[IA, IA] == pytest.approx(-1.0 / ref.tau_v, rel=1e-8)
    assert J[IA, IMP] == 0.0 and J[IA, IMU] == 0.0  # command saturated at 0
    assert J[IMS, IA] < 0.0  # opening the valve drains the supply

    Jc = jacobian(10.0, u0, ref, "coast")
    assert np.all(Jc[IMP:] == 0.0)
    assert np.all(Jc[:, IX] == 0.0)


def test_fd_jacobian_matches_rhs_increment(ref):
    """J du predicts rhs(u + du) - rhs(u) for a small, smooth, unsaturated perturbation."""
    u = state_at_pset(ref, a=0.4)
    Vu = ref.Vtank - u[IMP] / ref.rho_p
    u[IMU] = 0.97 * ref.Pset * Vu / (ref.R_he * ref.Tu)
    J = jacobian(0.0, u, ref, "powered")
    scales = np.array([ref.h0, ref.h0, ref.V0, 1.0, ref.mp0, ref.ms0, ref.mu0, 1.0])
    rng = np.random.default_rng(0)
    du = 1e-5 * scales * rng.standard_normal(8)
    df = rhs(0.0, u + du, ref, "powered") - rhs(0.0, u, ref, "powered")
    assert np.allclose(J @ du, df, rtol=1e-3, atol=1e-12 * np.abs(rhs(0.0, u, ref, "powered")).max())


@pytest.mark.parametrize("h, rho, T", [
    (0.0, 1.2250, 288.150),
    (20000.0, 8.8910e-2, 216.650),
    (30000.0, 1.8410e-2, 226.509),
    (40000.0, 3.9957e-3, 250.350),
    (50000.0, 1.0269e-3, 270.650),
])
def test_standard_atmosphere_1976_table(h, rho, T):
    """Published 1976 US Standard Atmosphere values at geometric altitude (used only for Mach)."""
    rho_calc, T_calc = standard_atmosphere_1976(h)
    assert rho_calc == pytest.approx(rho, rel=1e-4)
    assert T_calc == pytest.approx(T, abs=1e-3)
