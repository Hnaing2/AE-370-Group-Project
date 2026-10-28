"""Solver order on exact scalar problems."""

import math

import numpy as np
import pytest

from src.integrators import step_ab3, step_be, step_fe, step_rk4
from validation.scalar_convergence import EXPECTED_ORDER, PROBLEMS, convergence_table, integrate

T_END = 1.0
N_STEPS = (10, 20, 40, 80)


def observed_orders(method, f, u0, exact):
    errors = [abs(integrate(method, f, u0, T_END, n)[0] - exact) for n in N_STEPS]
    return [math.log2(e0 / e1) for e0, e1 in zip(errors, errors[1:])]


@pytest.mark.parametrize("method", list(EXPECTED_ORDER))
def test_decay_order(method):
    """u' = -u, u(0) = 1, exact exp(-t)."""
    orders = observed_orders(method, lambda t, u: -u, 1.0, math.exp(-T_END))
    assert orders[-1] == pytest.approx(EXPECTED_ORDER[method], abs=0.1)


@pytest.mark.parametrize("method", list(EXPECTED_ORDER))
def test_time_dependent_order(method):
    """u' = t - u, u(0) = 1, exact t - 1 + 2 exp(-t); catches wrong stage times or argument order."""
    orders = observed_orders(method, lambda t, u: t - u, 1.0, T_END - 1.0 + 2.0 * math.exp(-T_END))
    assert orders[-1] == pytest.approx(EXPECTED_ORDER[method], abs=0.1)


@pytest.mark.parametrize("problem", list(PROBLEMS))
@pytest.mark.parametrize("method", list(EXPECTED_ORDER))
def test_exported_convergence_table(problem, method):
    """The report's table (dt = 0.2 / 2^j) reaches each method's order at its finest pair."""
    table = convergence_table(problem, method)
    assert table["dt"][0] == 0.2 and len(table["observed_order"]) == len(table["dt"]) - 1
    assert table["observed_order"][-1] == pytest.approx(EXPECTED_ORDER[method], abs=0.1)
    assert all(e1 < e0 for e0, e1 in zip(table["error"], table["error"][1:]))


@pytest.mark.parametrize("method, calls_per_step", [("fe", 1), ("rk4", 4)])
def test_rhs_call_count(method, calls_per_step):
    seen = []

    def f(t, u):
        seen.append(t)
        return -u

    _, calls = integrate(method, f, 1.0, T_END, 10)
    assert calls == len(seen) == 10 * calls_per_step


def test_ab3_startup_and_call_count():
    """f(0, u0), two RK4 steps (3 new calls each), then one call per accepted state: n + 7 in all."""
    seen = []

    def f(t, u):
        seen.append(t)
        return -u

    _, calls = integrate("ab3", f, 1.0, T_END, 10)
    assert calls == len(seen) == 10 + 7
    rk4_start = [0.0, 0.05, 0.05, 0.1, 0.1, 0.15, 0.15, 0.2, 0.2]  # f(t_0); stages + f(t_1); stages + f(t_2)
    assert seen == pytest.approx(rk4_start + [0.1 * k for k in range(3, 11)], rel=1e-14)


def test_ab3_exact_for_quadratic_rate():
    """AB3 integrates the quadratic through (t_n, f_n), (t_{n-1}, f_{n-1}), (t_{n-2}, f_{n-2}) exactly,
    so u' = 3t^2 - 2t + 1 (u = t^3 - t^2 + t) is reproduced to round-off; catches coefficient order."""
    u = lambda t: t ** 3 - t ** 2 + t
    du = lambda t: 3.0 * t ** 2 - 2.0 * t + 1.0
    t_n, dt = 0.7, 0.1
    u_next = step_ab3(u(t_n), (du(t_n), du(t_n - dt), du(t_n - 2.0 * dt)), dt)
    assert u_next == pytest.approx(u(t_n + dt), rel=1e-14)


def test_ab3_vector_state():
    A = np.array([[-1.0, 2.0], [0.0, -3.0]])
    u = np.array([1.0, -2.0])
    u_copy = u.copy()
    f_hist = (A @ u, A @ u, A @ u)
    u_next = step_ab3(u, f_hist, 1e-3)
    np.testing.assert_array_equal(u, u_copy)
    np.testing.assert_allclose(u_next, u + 1e-3 * (A @ u), rtol=1e-14)  # 23 - 16 + 5 = 12


def test_be_scalar_decay_closed_form():
    """One BE step of u' = -u is u / (1 + dt); f is returned at the new state; calls are counted."""
    seen = []

    def f(t, u):
        seen.append(t)
        return -u

    res = step_be(f, 0.3, 2.0, 0.1)
    assert res.success and res.message == "converged"
    assert np.shape(res.u) == ()
    assert res.u == pytest.approx(2.0 / 1.1, rel=1e-14)
    assert res.f == -res.u
    assert res.residual <= 1e-10
    assert res.n_rhs == len(seen) and res.n_jac == 1  # one forward-difference Newton matrix
    assert seen == pytest.approx([0.4] * len(seen), rel=1e-15)  # f evaluated at t + dt only


def test_be_stiff_scaled_system():
    """Scaled solve with a supplied Jacobian on a nonlinear stiff system; scales spanning 1e-3..1e4."""
    S = np.array([1e4, 1e-3])

    def f(t, u):
        w = u / S
        return S * np.array([-w[0] ** 3 + w[1], -500.0 * w[1] + np.sin(w[0])])

    def jac(t, u):
        w = u / S
        J_w = np.array([[-3.0 * w[0] ** 2, 1.0], [np.cos(w[0]), -500.0]])
        return J_w * S[:, np.newaxis] / S[np.newaxis, :]

    u = S * np.array([0.8, 0.3])
    u_copy = u.copy()
    dt = 0.05
    res = step_be(f, 0.0, u, dt, jac=jac, scales=S)
    np.testing.assert_array_equal(u, u_copy)
    assert res.success, res.message
    assert res.n_jac >= 1
    np.testing.assert_allclose(res.f, f(dt, res.u), rtol=0, atol=0)
    G = (res.u - u - dt * f(dt, res.u)) / S
    assert np.max(np.abs(G)) == pytest.approx(res.residual, abs=1e-15)
    assert res.residual <= 1e-10
    assert res.n_iter >= 1
    # Same root from a forward-difference Newton matrix, and from scipy's hybr (cross-check).
    res_fd = step_be(f, 0.0, u, dt, scales=S)
    res_hybr = step_be(f, 0.0, u, dt, jac=jac, scales=S, solver_options={"method": "hybr"})
    for other in (res_fd, res_hybr):
        assert other.success, other.message
        np.testing.assert_allclose(other.u / S, res.u / S, rtol=0, atol=1e-10)
    assert res_fd.n_jac >= 1 and res_fd.n_rhs >= res.n_rhs + 2  # 2 columns of differences
    assert res_hybr.n_iter is None                              # hybr does not expose iterations


def test_be_linear_converges_in_one_correction():
    """On a linear problem the first Newton correction is exact; the second (round-off) one stops it."""
    res = step_be(lambda t, u: -u, 0.0, 1.0, 0.1, jac=lambda t, u: -1.0)
    assert res.success, res.message
    assert (res.n_iter, res.n_jac, res.n_rhs) == (2, 1, 3)


def test_be_reports_solver_failure():
    """u' = u^2 + 1 from u = 1 with dt = 1: z - 1 - (z^2 + 1) = 0 has no real root."""
    res = step_be(lambda t, u: u * u + 1.0, 0.0, 1.0, 1.0, jac=lambda t, u: 2.0 * u)
    assert not res.success
    assert res.residual > 1e-10
    assert res.message.startswith("solver failed")


def test_be_rejects_convergence_without_small_residual():
    """A loose step_tol stops the iteration early; the achieved residual still decides acceptance."""
    res = step_be(lambda t, u: -u ** 3, 0.0, 1.0, 0.5, jac=lambda t, u: -3.0 * u ** 2, residual_tol=1e-14,
                  solver_options={"step_tol": 1e-2})
    assert not res.success
    assert res.residual > 1e-14
    assert res.message.startswith("solver converged but")


def test_be_rejects_nonpositive_scales():
    with pytest.raises(ValueError):
        step_be(lambda t, u: -u, 0.0, np.ones(2), 0.1, scales=np.array([1.0, 0.0]))


def test_rk4_stage_times():
    seen = []

    def f(t, u):
        seen.append(t)
        return t - u

    step_rk4(f, 2.0, 1.0, 0.5)
    assert seen == [2.0, 2.25, 2.25, 2.5]


def test_rk4_reuses_supplied_k1():
    f = lambda t, u: t - u
    t, u, dt = 0.3, 0.7, 0.1
    u_fresh, calls_fresh = step_rk4(f, t, u, dt)
    u_reuse, calls_reuse = step_rk4(f, t, u, dt, f0=f(t, u))
    assert u_reuse == u_fresh
    assert (calls_fresh, calls_reuse) == (4, 3)


def test_fe_reuses_supplied_f0():
    f = lambda t, u: t - u
    u_next, calls = step_fe(f, 0.3, 0.7, 0.1, f0=f(0.3, 0.7))
    assert u_next == step_fe(f, 0.3, 0.7, 0.1)[0]
    assert calls == 0


@pytest.mark.parametrize("step", [step_fe, step_rk4])
def test_vector_state_not_mutated(step):
    """Steps work componentwise on float64 arrays and leave the input state untouched."""
    A = np.array([[-1.0, 2.0], [0.0, -3.0]])
    u = np.array([1.0, -2.0])
    u_copy = u.copy()
    u_next, _ = step(lambda t, v: A @ v, 0.0, u, 1e-3)
    np.testing.assert_array_equal(u, u_copy)
    assert u_next.shape == (2,) and u_next.dtype == np.float64
    np.testing.assert_allclose(u_next, u + 1e-3 * (A @ u), rtol=0, atol=1e-5)
