"""Independent physical benchmarks and Jacobian checks.

Each benchmark is a test-only Params made with dataclasses.replace from the reference case, run through the
production RHS and simulate driver, and compared with a closed form coded here independently of the RHS.
"""

import dataclasses
import functools
import math

import numpy as np
import pytest

from src.integrators import amplification_factor
from src.parameters import FIXED_ORIFICE_CASE, REFERENCE_CASE, STUDY_BASELINE, build_initial_state, build_params
from src.physics import (
    IA, IGAMMA, IH, IMP, IMS, IMU, IV, IX,
    analytic_jacobian_choked,
    derived_quantities,
    jacobian,
    state_scales,
)
from src.simulation import derived_history, simulate

FLAT_EARTH_R_E = 1e30  # m; R_E / (R_E + h) is exactly 1.0, so g = g0
T_POWERED_ONLY = 8.0   # s; t_end just past cutoff
EXPECTED_ORDER = {"fe": 1, "rk4": 4, "ab3": 3, "be": 1}


@functools.lru_cache(maxsize=None)
def reference_params():
    return build_params(REFERENCE_CASE)


def ballistic_params():
    """No thrust (K_F = 0, so qp = 0, Pu stays above Pset, and the valve stays closed: M is constant),
    no drag (Cd = 0), constant g. Both phases are then the same ballistic flight."""
    return dataclasses.replace(reference_params(), K_F=0.0, Cd=0.0, R_E=FLAT_EARTH_R_E)


def fixed_pressure_params(Pu, **changes):
    """Test-only fixed-Pu scenario: infinite propellant density, so draining propellant leaves Vu = Vtank,
    and mu0 set so that mu0 R Tu / Vtank = Pu. While qr = 0, Pu is constant to the last bit."""
    p = reference_params()
    return dataclasses.replace(p, rho_p=math.inf, mu0=Pu * p.Vtank / (p.R_he * p.Tu), **changes)


def vertical_burn_params():
    """gamma0 = pi/2, no drag, constant g, fixed Pu = Pu0 > Pset (valve stays closed, qr = 0)."""
    p = reference_params()
    return fixed_pressure_params(p.Pu0, Cd=0.0, R_E=FLAT_EARTH_R_E, gamma0=math.pi / 2)


def ballistic_exact(p, t):
    """Columns x, h, V, gamma of drag-free flight in constant gravity."""
    vx, vz0 = p.V0 * math.cos(p.gamma0), p.V0 * math.sin(p.gamma0)
    vz = vz0 - p.g0 * t
    return np.column_stack([p.x0 + vx * t, p.h0 + vz0 * t - 0.5 * p.g0 * t * t, np.hypot(vx, vz), np.arctan2(vz, vx)])


def ballistic_apogee(p):
    """(t, x, h) at apogee."""
    vx, vz0 = p.V0 * math.cos(p.gamma0), p.V0 * math.sin(p.gamma0)
    t = vz0 / p.g0
    return t, p.x0 + vx * t, p.h0 + vz0 * vz0 / (2.0 * p.g0)


def vertical_burn_exact(p, u0, t):
    """Columns h, V, mp for a vertical burn at constant qp, no drag, constant g."""
    M0 = p.m_dry + u0[IMP] + u0[IMS] + u0[IMU]
    Pu = u0[IMU] * p.R_he * p.Tu / p.Vtank
    qp = p.K_F * Pu / (p.Isp * p.g0)
    c = p.Isp * p.g0 * math.cos(p.delta)
    M = M0 - qp * t
    V = p.V0 + c * np.log(M0 / M) - p.g0 * t
    h = p.h0 + p.V0 * t + c * (t - M / qp * np.log(M0 / M)) - 0.5 * p.g0 * t * t
    return np.column_stack([h, V, u0[IMP] - qp * t])


def test_ballistic_trajectory():
    """No thrust/drag, constant g: closed-form x, h, V, gamma at every stored state of both segments, and the
    Hermite apogee; checks the gamma equation."""
    p = ballistic_params()
    result = simulate("rk4", p, {"dt": 0.05})
    assert result.status == "completed"
    for seg in (result.powered, result.coast):
        err = np.abs(seg.U[:, :IMP] - ballistic_exact(p, seg.t))
        assert np.all(err.max(axis=0) <= [1e-8, 1e-8, 1e-10, 1e-12]), err.max(axis=0)
        assert np.all(seg.U[:, IMP:] == build_initial_state(p)[IMP:])  # nothing burns, nothing flows
    t_apo, x_apo, h_apo = ballistic_apogee(p)
    apogee = result.events["apogee"]
    assert apogee["t"] == pytest.approx(t_apo, abs=1e-10)
    assert apogee["u"][IX] == pytest.approx(x_apo, abs=1e-8)
    assert apogee["u"][IH] == pytest.approx(h_apo, abs=1e-8)
    assert abs(apogee["u"][IGAMMA]) <= 1e-12


def test_vertical_constant_force_burn():
    """gamma = pi/2, fixed Pu, no drag, constant g: V = V0 + Isp g0 cos(delta) ln(M0/M) - g0 t, the altitude
    closed form, and mp = mp0 - qp t; checks mass, force projection, and gravity. Coast: V falls at g0."""
    p = vertical_burn_params()
    u0 = build_initial_state(p)
    result = simulate("rk4", p, {"dt": 2.5e-3, "t_end": T_POWERED_ONLY})
    assert result.status == "no_apogee"
    powered, coast = result.powered, result.coast

    hist = derived_history(result)
    Pu = hist["Pu"][hist["powered"]]
    assert np.all(Pu == Pu[0]) and Pu[0] > p.Pset          # fixed Pu, valve held closed
    assert np.all(powered.U[:, IA] == 0.0) and np.all(powered.U[:, IMU] == u0[IMU])
    assert np.all(np.abs(powered.U[:, IGAMMA] - math.pi / 2) <= 1e-15)
    assert np.all(np.abs(powered.U[:, IX]) <= 1e-9)

    err = np.abs(np.column_stack([powered.U[:, IH], powered.U[:, IV], powered.U[:, IMP]])
                 - vertical_burn_exact(p, u0, powered.t))
    assert np.all(err.max(axis=0) <= [1e-8, 1e-10, 1e-9]), err.max(axis=0)

    u_cut, t_cut = powered.U[-1], powered.t[-1]
    tc = coast.t - t_cut
    np.testing.assert_allclose(coast.U[:, IV], u_cut[IV] - p.g0 * tc, rtol=0, atol=1e-10)
    np.testing.assert_allclose(coast.U[:, IH], u_cut[IH] + u_cut[IV] * tc - 0.5 * p.g0 * tc * tc, rtol=0, atol=1e-9)


ORDER_STEPS = {  # dt pairs that divide t_cut, in each method's asymptotic range and above round-off
    "ballistic": {"fe": (0.1, 0.05), "rk4": (0.5, 0.25), "ab3": (0.1, 0.05), "be": (0.1, 0.05)},
    "vertical_burn": {"fe": (0.01, 0.005), "rk4": (0.375, 0.1875), "ab3": (0.01, 0.005), "be": (0.01, 0.005)},
}


def benchmark_errors(name, method, dt):
    """Max errors against the closed form: ballistic x/h over all stored states and the apogee height;
    vertical burn h and V over the powered states."""
    if name == "ballistic":
        p = ballistic_params()
        result = simulate(method, p, {"dt": dt})
        assert result.status == "completed"
        pos = max(np.max(np.abs(seg.U[:, IX:IV] - ballistic_exact(p, seg.t)[:, :2])) for seg in (result.powered, result.coast))
        return {"x, h": pos, "apogee h": abs(result.events["apogee"]["u"][IH] - ballistic_apogee(p)[2])}
    p = vertical_burn_params()
    result = simulate(method, p, {"dt": dt, "t_end": T_POWERED_ONLY})
    assert result.status == "no_apogee"
    seg = result.powered
    exact = vertical_burn_exact(p, build_initial_state(p), seg.t)
    return {"h": np.max(np.abs(seg.U[:, IH] - exact[:, 0])), "V": np.max(np.abs(seg.U[:, IV] - exact[:, 1]))}


@pytest.mark.parametrize("name", list(ORDER_STEPS))
@pytest.mark.parametrize("method", list(EXPECTED_ORDER))
def test_driver_order_on_benchmarks(name, method):
    """Through the full driver (AB3 start/restart, BE Newton solve, Hermite apogee), each method converges at
    its theoretical order on the smooth benchmarks."""
    coarse, fine = (benchmark_errors(name, method, dt) for dt in ORDER_STEPS[name][method])
    for key in coarse:
        order = math.log2(coarse[key] / fine[key])
        assert order == pytest.approx(EXPECTED_ORDER[method], abs=0.1), (key, coarse[key], fine[key])


@pytest.mark.parametrize("method", ["fe", "rk4", "be"])
def test_constant_valve_command(method):
    """Command frozen at a_bar (fixed Pu below Pset, C_reg = 0 so qr = 0): one-step methods give exactly
    a_n = a_bar + (a0 - a_bar) R(-dt/tau_v)^n, and RK4 tracks a = a_bar + (a0 - a_bar) exp(-t/tau_v).
    A short burn (t_cut = 0.5 s, about 25 tau_v) keeps the fine-dt run cheap."""
    p = fixed_pressure_params(0.97 * reference_params().Pset, C_reg=0.0, t_cut=0.5)
    a_bar = p.kappa * 0.03
    dt = 1.25e-3
    result = simulate(method, p, {"dt": dt, "t_end": 0.6})
    assert result.status == "no_apogee"
    powered = result.powered
    hist = derived_history(result)
    np.testing.assert_allclose(hist["a_cmd"][hist["powered"]], a_bar, rtol=1e-12)
    assert np.all(hist["qr"] == 0.0)

    n = np.arange(powered.t.size)
    R = amplification_factor(method, -dt / p.tau_v).real
    np.testing.assert_allclose(powered.U[:, IA], a_bar + (p.a0 - a_bar) * R ** n, rtol=0, atol=1e-10)
    if method == "rk4":
        np.testing.assert_allclose(powered.U[:, IA], a_bar + (p.a0 - a_bar) * np.exp(-powered.t / p.tau_v),
                                   rtol=0, atol=1e-7)
    assert np.all(result.coast.U[:, IA] == powered.U[-1, IA])  # held at its cutoff value


def test_passive_isothermal_ullage():
    """qr = 0 (C_reg = 0): mu stays mu0, so Pu Vu = mu0 R Tu at every state, and the blowdown follows
    Vu^2 = Vu0^2 + 2 B mu0 R Tu t / rho_p with B = K_F / (Isp g0)."""
    p = dataclasses.replace(reference_params(), C_reg=0.0)
    result = simulate("rk4", p, {"dt": 2.5e-3, "t_end": T_POWERED_ONLY})
    assert result.status == "no_apogee"
    powered = result.powered
    assert np.all(powered.U[:, IMU] == p.mu0) and np.all(powered.U[:, IMS] == p.ms0)

    hist = derived_history(result)
    on = hist["powered"]
    mRT = p.mu0 * p.R_he * p.Tu
    np.testing.assert_allclose(hist["Pu"][on] * hist["Vu"][on], mRT, rtol=1e-15)

    B = p.K_F / (p.Isp * p.g0)
    Vu = np.sqrt(p.Vu0 ** 2 + 2.0 * B * mRT * powered.t / p.rho_p)
    np.testing.assert_allclose(hist["Pu"][on], mRT / Vu, rtol=1e-11)
    np.testing.assert_allclose(powered.U[:, IMP], p.rho_p * (p.Vtank - Vu), rtol=1e-11)
    assert hist["Pu"][on][-1] < 0.25 * p.Pset  # a real blowdown, far below the regulated pressure


def jacobian_state(p, Pu_frac, a, h=25000.0, V=650.0, gamma=0.9, mp_frac=0.6, ms_frac=0.8):
    """A smooth mid-flight state with Pu = Pu_frac * Pset."""
    u = build_initial_state(p)
    u[IH], u[IV], u[IGAMMA] = h, V, gamma
    u[IMP], u[IMS], u[IA] = mp_frac * p.mp0, ms_frac * p.ms0, a
    u[IMU] = Pu_frac * p.Pset * (p.Vtank - u[IMP] / p.rho_p) / (p.R_he * p.Tu)
    return u


def scaled(J, p):
    S = state_scales(p)
    return J * S[np.newaxis, :] / S[:, np.newaxis]


JACOBIAN_STATES = {  # name: (Pu / Pset, a, expected command branch)
    "initial": (None, None, "closed"),        # u0: Pu0 > Pset, valve on its a = 0 bound
    "modulating": (0.995, 0.4, "modulating"),  # K_p (Pset - Pu) = 0.1 (kappa 20) or 0.5 (kappa 100)
    "deep droop": (0.5, 0.9, "open"),
    "overpressure": (1.02, 0.3, "closed"),
}


@pytest.mark.parametrize("case", [{}, {"tau_v": 0.005, "kappa": 100.0}], ids=["nominal", "tau5_kappa100"])
@pytest.mark.parametrize("state", list(JACOBIAN_STATES))
@pytest.mark.parametrize("rel_step", [1e-5, 1e-6, 1e-7])
def test_fd_jacobian_matches_analytic_choked(case, state, rel_step):
    """Several smooth states, saturated and unsaturated command branches, more than one step size."""
    p = build_params({**STUDY_BASELINE, **case})
    Pu_frac, a, branch = JACOBIAN_STATES[state]
    u = build_initial_state(p) if Pu_frac is None else jacobian_state(p, Pu_frac, a)
    assert derived_quantities(u, p, "powered")["choked"]
    Ja = scaled(analytic_jacobian_choked(0.0, u, p), p)
    Jf = scaled(jacobian(0.0, u, p, "powered", rel_step=rel_step), p)
    np.testing.assert_allclose(Jf, Ja, rtol=1e-5, atol=1e-8 * np.max(np.abs(Ja)))
    feedback = Ja[IA, [IMP, IMU]]
    assert np.all(feedback < 0.0) if branch == "modulating" else np.all(feedback == 0.0)
    assert Ja[IA, IA] == -1.0 / p.tau_v


def test_analytic_jacobian_rejects_subcritical():
    p = reference_params()
    u = jacobian_state(p, 0.97, 0.5, ms_frac=0.5)  # Ps falls to ~0.31 Ps0: r ~ 0.7 > r_*
    assert not derived_quantities(u, p, "powered")["choked"]
    with pytest.raises(ValueError):
        analytic_jacobian_choked(0.0, u, p)


def test_fixed_orifice_jacobian_last_row_zero():
    """Fixed orifice: a_dot = 0, so the whole last row is zero in both Jacobians; not the regulated row."""
    fixed = build_params(FIXED_ORIFICE_CASE)
    regulated = reference_params()
    for u in (build_initial_state(fixed), jacobian_state(fixed, 0.9, fixed.a_fix)):
        Jf = jacobian(0.0, u, fixed, "powered")
        Ja = analytic_jacobian_choked(0.0, u, fixed)
        assert np.all(Jf[IA] == 0.0) and np.all(Ja[IA] == 0.0)
        np.testing.assert_allclose(scaled(Jf, fixed), scaled(Ja, fixed), rtol=1e-5,
                                   atol=1e-8 * np.max(np.abs(scaled(Ja, fixed))))
        assert Jf[IMS, IA] < 0.0  # the valve opening still sets the flow
        assert jacobian(0.0, u, regulated, "powered")[IA, IA] == pytest.approx(-1.0 / regulated.tau_v, rel=1e-8)
