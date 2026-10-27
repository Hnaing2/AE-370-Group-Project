"""M3/M5: solver order on exact scalar problems (verification_studies.md Section 10)."""

import math

import numpy as np
import pytest

from src.integrators import step_fe, step_rk4

EXPECTED_ORDER = {"fe": 1, "rk4": 4, "ab3": 3, "be": 1}
SINGLE_STEPS = {"fe": step_fe, "rk4": step_rk4}
T_END = 1.0
N_STEPS = (10, 20, 40, 80)


def integrate(method, f, u0, t_end, n):
    """Take n fixed steps of dt = t_end / n with times n * dt; returns (u(t_end), rhs_calls)."""
    if method not in SINGLE_STEPS:
        pytest.skip(f"M5: {method} not implemented")
    step = SINGLE_STEPS[method]
    dt = t_end / n
    u, calls = u0, 0
    for k in range(n):
        u, c = step(f, k * dt, u, dt)
        calls += c
    return u, calls


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


@pytest.mark.parametrize("method, calls_per_step", [("fe", 1), ("rk4", 4)])
def test_rhs_call_count(method, calls_per_step):
    seen = []

    def f(t, u):
        seen.append(t)
        return -u

    _, calls = integrate(method, f, 1.0, T_END, 10)
    assert calls == len(seen) == 10 * calls_per_step


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
