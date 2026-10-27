"""M4/M5: driver bookkeeping and conservation (numerics.md Section 8.5; verification_studies.md Section 10)."""

import functools

import numpy as np
import pytest

from src.parameters import FIXED_ORIFICE_CASE, REFERENCE_CASE, build_params
from src.physics import IA, IGAMMA, IH, IMP, IMS, IMU, IV, IX
from src.simulation import (
    Segment,
    compute_metrics,
    derived_history,
    find_apogee,
    hermite_interpolate,
    sample_at_time,
    simulate,
)

T_POWERED_ONLY = 8.0  # s; t_end just past cutoff for checks that only need the powered segment
HELIUM_LIMIT = {"fe": 1e-10, "rk4": 1e-10, "ab3": 1e-10, "be": 1e-8}  # Section 11.1


@functools.lru_cache(maxsize=None)
def run(method="rk4", dt=2.5e-3, t_end=None, **changes):
    params = build_params({**REFERENCE_CASE, **changes})
    return simulate(method, params, {"dt": dt, "t_end": t_end})


@pytest.mark.parametrize("dt", [2.5e-3, 1.25e-3, 6.25e-4])
def test_cutoff_lands_on_step(dt):
    result = run(dt=dt, t_end=T_POWERED_ONLY)
    n_cut = result.diagnostics["n_cut"]
    p = result.params
    assert n_cut * dt == pytest.approx(p.t_cut, rel=1e-14)
    assert np.array_equal(result.powered.t, np.arange(n_cut + 1) * dt)  # t_n = n dt, not accumulated
    assert np.array_equal(result.coast.t, (n_cut + np.arange(result.coast.t.size)) * dt)
    assert result.powered.t[-1] == result.coast.t[0] == result.diagnostics["t_cut"]


def test_dt_must_divide_t_cut():
    with pytest.raises(ValueError):
        run(dt=7e-4, t_end=T_POWERED_ONLY)


def test_cutoff_bookkeeping():
    """Thrust and qp zero after t_cut; mp, ms, mu, a constant in coast; cutoff state in both segments."""
    result = run()
    assert result.status == "completed"
    powered, coast = result.powered, result.coast
    assert np.array_equal(powered.U[-1], coast.U[0])
    assert powered.F[-1][IMP] < 0.0 and powered.F[-1][IMU] > 0.0  # powered-side derivative at t_cut
    assert np.all(coast.F[:, IMP:] == 0.0)                         # coast-side derivatives
    assert np.all(coast.U[:, IMP:] == coast.U[0, IMP:])

    hist = derived_history(result)
    after = ~hist["powered"]
    for key in ("F_eng", "T_ax", "qp", "qr"):
        assert np.all(hist[key][after] == 0.0), key
    assert np.all(hist["T_ax"][hist["powered"]] > 0.0)


def test_fixed_orifice_valve_constant():
    params = build_params(FIXED_ORIFICE_CASE)
    result = simulate("rk4", params, {"dt": 2.5e-3, "t_end": T_POWERED_ONLY})
    for seg in (result.powered, result.coast):
        assert np.all(seg.U[:, IA] == params.a_fix)


@pytest.mark.parametrize("method", ["fe", "rk4", "ab3", "be"])
def test_helium_balance(method):
    """Relative drift of ms + mu <= 1e-10 (FE, RK4, AB3), <= 1e-8 (BE)."""
    if method in ("ab3", "be"):
        pytest.skip("M5: not implemented")
    result = run(method=method, t_end=T_POWERED_ONLY)
    assert result.status == "no_apogee"  # stopped at t_end, after cutoff
    assert result.diagnostics["helium_drift"] <= HELIUM_LIMIT[method]
    helium = result.powered.U[:, IMS] + result.powered.U[:, IMU]
    assert np.max(np.abs(helium - helium[0])) / helium[0] == result.diagnostics["helium_drift"]


def test_coupling_isolation():
    """Different Cd / gamma0, same gas inputs -> identical pressure and propellant histories."""
    nominal = run(t_end=T_POWERED_ONLY)
    for changes in ({"Cd": 0.4, "gamma0": 0.93}, {"Cd": 0.8, "gamma0": 1.27}):
        other = run(t_end=T_POWERED_ONLY, **changes)
        assert not np.array_equal(other.powered.U[:, :IMP], nominal.powered.U[:, :IMP])
        for seg, ref in ((other.powered, nominal.powered), (other.coast, nominal.coast)):
            assert np.array_equal(seg.U[:, IMP:], ref.U[:, IMP:])
        assert np.array_equal(derived_history(other)["Pu"], derived_history(nominal)["Pu"])


def test_rhs_call_counts():
    """RK4 reuses each stored F as k1 (4 calls per step); FE takes 1. Each segment adds its final F."""
    for method, per_step in (("rk4", 4), ("fe", 1)):
        result = run(method=method, t_end=T_POWERED_ONLY)
        n_steps = result.powered.t.size - 1 + result.coast.t.size - 1
        assert result.counters["rhs_calls"] == per_step * n_steps + 2


def test_sample_at_time_never_crosses_cutoff():
    result = run(t_end=T_POWERED_ONLY)
    t_cut = result.diagnostics["t_cut"]
    u_cut = result.powered.U[-1]
    assert np.array_equal(sample_at_time(result, t_cut, phase="powered"), u_cut)
    assert np.array_equal(sample_at_time(result, t_cut, phase="coast"), u_cut)
    assert np.array_equal(sample_at_time(result, result.powered.t[7]), result.powered.U[7])
    with pytest.raises(ValueError):
        sample_at_time(result, t_cut + 1e-3, phase="powered")
    with pytest.raises(ValueError):
        sample_at_time(result, t_cut - 1e-3, phase="coast")


def test_hermite_interpolant_exact_for_cubics():
    c = np.array([[1.0, 2.0, -3.0, 0.5], [-4.0, 0.0, 1.5, -2.0]])  # two cubic components
    u = lambda t: c @ np.array([1.0, t, t * t, t ** 3])
    du = lambda t: c @ np.array([0.0, 1.0, 2.0 * t, 3.0 * t * t])
    t0, t1 = 0.3, 1.7
    for tq in np.linspace(t0, t1, 9):
        interp = hermite_interpolate(t0, t1, u(t0), u(t1), du(t0), du(t1), tq)
        assert np.allclose(interp, u(tq), rtol=0.0, atol=1e-13)


def test_hermite_apogee_on_ballistic():
    """No thrust/drag, constant g: closed-form states at a coarse 1 s step; h(t) is quadratic, so the
    Hermite apogee is exact to round-off."""
    g, h0, V0, gamma0 = 9.80665, 19000.0, 536.0, 1.10
    vx, vz0 = V0 * np.cos(gamma0), V0 * np.sin(gamma0)
    t = np.arange(0.0, 60.0, 1.0)
    vz = vz0 - g * t
    V = np.hypot(vx, vz)
    U = np.ones((t.size, 8))
    U[:, IX], U[:, IH], U[:, IV], U[:, IGAMMA] = vx * t, h0 + vz0 * t - 0.5 * g * t * t, V, np.arctan2(vz, vx)
    F = np.zeros((t.size, 8))
    F[:, IX], F[:, IH], F[:, IV], F[:, IGAMMA] = vx, vz, -g * vz / V, -g * vx / V ** 2

    apogee = find_apogee(Segment("coast", t, U, F))
    t_apo = vz0 / g
    assert apogee["step"] == int(t_apo)
    assert apogee["t"] == pytest.approx(t_apo, rel=1e-13)
    assert apogee["u"][IH] == pytest.approx(h0 + vz0 ** 2 / (2.0 * g), rel=1e-13)
    assert apogee["u"][IX] == pytest.approx(vx * t_apo, rel=1e-13)


def test_invalid_state_ends_run():
    """FE far outside its stability limit drives the valve opening out of [0, 1]; never clipped."""
    result = run(method="fe", dt=0.075)
    assert result.status == "physical_model_invalid"
    assert result.coast is None
    assert result.diagnostics["invalid"]["phase"] == "powered"
    assert "valve opening" in result.diagnostics["invalid"]["message"]
    metrics = compute_metrics(result)
    assert np.isnan(metrics["apogee_h"]) and np.isnan(metrics["Pu_cut"])


def test_ab3_restarts_at_cutoff():
    pytest.skip("M5: not implemented")


def test_be_reports_residual_and_counts():
    pytest.skip("M5: not implemented")
