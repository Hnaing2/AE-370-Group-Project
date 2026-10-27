"""Driver bookkeeping and conservation."""

import functools

import numpy as np
import pytest

from src.integrators import step_ab3, step_rk4
from src.parameters import FIXED_ORIFICE_CASE, REFERENCE_CASE, build_params
from src.physics import IA, IGAMMA, IH, IMP, IMS, IMU, IV, IX, rhs, state_scales
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
HELIUM_LIMIT = {"fe": 1e-10, "rk4": 1e-10, "ab3": 1e-10, "be": 1e-8}


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
    """RK4 reuses each stored F as k1 (4 calls per step); FE and AB3 take 1. Each segment adds its
    initial F, and AB3 adds 3 more for each of its two RK4 steps per segment (start, restart)."""
    for method, per_step, extra in (("rk4", 4, 0), ("fe", 1, 0), ("ab3", 1, 2 * 2 * 3)):
        result = run(method=method, t_end=T_POWERED_ONLY)
        n_steps = result.powered.t.size - 1 + result.coast.t.size - 1
        assert result.counters["rhs_calls"] == per_step * n_steps + 2 + extra


def _count_rhs_calls(monkeypatch) -> list:
    """Patch rhs so every true evaluation (including one that raises) appends to the returned list."""
    import src.physics
    import src.simulation

    calls = []
    real_rhs = src.physics.rhs

    def counting_rhs(*args, **kwargs):
        calls.append(None)
        return real_rhs(*args, **kwargs)

    monkeypatch.setattr(src.physics, "rhs", counting_rhs)
    monkeypatch.setattr(src.simulation, "rhs", counting_rhs)
    return calls


@pytest.mark.parametrize("method", ["fe", "rk4", "ab3", "be"])
def test_every_rhs_call_counted(method, monkeypatch):
    """counters["rhs_calls"] equals the true number of RHS evaluations, BE finite differences included."""
    import src.physics
    import src.simulation

    calls = _count_rhs_calls(monkeypatch)
    jac_calls = []
    real_jacobian = src.physics.jacobian
    monkeypatch.setattr(src.simulation, "jacobian", lambda *a, **k: jac_calls.append(None) or real_jacobian(*a, **k))
    result = simulate(method, build_params(REFERENCE_CASE), {"dt": 5e-3, "t_end": T_POWERED_ONLY})
    assert result.status == "no_apogee"
    assert result.counters["rhs_calls"] == len(calls)
    assert result.counters["jac_calls"] == len(jac_calls)
    if method == "be":
        assert len(calls) > 16 * result.counters["jac_calls"]  # 16 per central-difference Jacobian


@pytest.mark.parametrize("method", ["rk4", "ab3", "fe"])
def test_failed_explicit_run_counts_every_rhs_call(method, monkeypatch):
    """A stage that leaves the physical domain raises inside the step; its calls still count. RK4 at
    15 ms with tau_v = 5 ms, kappa = 5 rejects a negative stage valve opening in the first steps."""
    calls = _count_rhs_calls(monkeypatch)
    params = build_params({**REFERENCE_CASE, "tau_v": 0.005, "kappa": 5.0})
    result = simulate(method, params, {"dt": 0.015, "t_end": T_POWERED_ONLY})
    assert result.status == "physical_model_invalid"
    assert result.counters["rhs_calls"] == len(calls)


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
    """Each segment starts with two RK4 steps on its own derivative history, then AB3:
    the coast segment never uses powered-side derivatives."""
    result = run(method="ab3", t_end=T_POWERED_ONLY)
    assert result.status == "no_apogee"
    assert result.counters["ab3_restarts"] == 1
    dt = result.diagnostics["dt"]
    for seg in (result.powered, result.coast):
        f = lambda t, u: rhs(t, u, result.params, seg.phase)
        t, U, F = seg.t, seg.U, seg.F
        assert seg.t.size > 4
        for k in range(t.size):
            assert np.array_equal(F[k], f(t[k], U[k]))
        for k in (0, 1):
            assert np.array_equal(U[k + 1], step_rk4(f, t[k], U[k], dt, F[k])[0])
        for k in range(2, t.size - 1):
            assert np.array_equal(U[k + 1], step_ab3(U[k], (F[k], F[k - 1], F[k - 2]), dt))
    # The coast-side derivative at t_cut differs from the powered side; restarting matters.
    assert not np.array_equal(result.coast.F[0], result.powered.F[-1])


def test_be_reports_residual_and_counts():
    """Every accepted BE step satisfies its scaled residual, recomputed here from the stored U and F."""
    result = run(method="be", t_end=T_POWERED_ONLY)
    assert result.status == "no_apogee"
    counters, diag = result.counters, result.diagnostics
    n_steps = result.powered.t.size - 1 + result.coast.t.size - 1
    assert counters["be_failures"] == 0 and diag["solver_failure"] is None
    assert counters["jac_calls"] >= n_steps                 # at least one Newton matrix per step
    assert counters["nonlinear_iters"] >= 2 * n_steps       # a correction plus the round-off check
    assert 0.0 <= diag["be_max_residual"] <= 1e-10
    S = state_scales(result.params)
    dt = diag["dt"]
    for seg in (result.powered, result.coast):
        G = (seg.U[1:] - seg.U[:-1] - dt * seg.F[1:]) / S
        assert np.max(np.abs(G)) <= 1e-10
        for k in (0, seg.t.size // 2, seg.t.size - 1):
            assert np.array_equal(seg.F[k], rhs(seg.t[k], seg.U[k], result.params, seg.phase))


def test_be_tightened_solve_changes_nothing():
    """Tighter algebraic settings leave the outputs unchanged far below tolerance."""
    nominal = run(method="be", t_end=T_POWERED_ONLY)
    params = nominal.params
    tight = simulate("be", params, {"dt": 2.5e-3, "t_end": T_POWERED_ONLY, "be_residual_tol": 1e-13,
                                    "be_solver_options": {"step_tol": 1e-14}})
    assert tight.status == "no_apogee"
    assert tight.diagnostics["be_max_residual"] <= 1e-13
    Pu_change = np.max(np.abs(derived_history(tight)["Pu"] - derived_history(nominal)["Pu"]))
    assert Pu_change <= 1e-6 * 8e3  # 8 kPa pressure tolerance
    assert np.max(np.abs(tight.coast.U[-1, :IMP] - nominal.coast.U[-1, :IMP])
                  / state_scales(params)[:IMP]) <= 1e-10


def test_be_rejected_step_ends_run():
    """A step without convergence ends the run as solver_failed; accepted states are kept."""
    params = build_params(REFERENCE_CASE)
    result = simulate("be", params, {"dt": 2.5e-3, "t_end": T_POWERED_ONLY,
                                     "be_solver_options": {"max_iter": 1}})
    assert result.status == "solver_failed"
    assert result.coast is None and result.powered.t.size == 1
    failure = result.diagnostics["solver_failure"]
    assert failure["phase"] == "powered" and failure["t_last_accepted"] == 0.0
    assert "no convergence" in failure["message"] and failure["residual"] > 0.0
    assert result.diagnostics["invalid"] is None
    assert result.counters["be_failures"] == 1
    metrics = compute_metrics(result)
    assert metrics["status"] == "solver_failed" and np.isnan(metrics["apogee_h"])
