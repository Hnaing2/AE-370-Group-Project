"""Stability functions and the frozen-Jacobian characterization."""

import functools
import math

import numpy as np
import pytest

from src.integrators import (
    ab3_characteristic_roots,
    amplification_factor,
    is_absolutely_stable,
    step_ab3,
    step_be,
    step_fe,
    step_rk4,
)
from src.parameters import STUDY_BASELINE, build_params
from src.simulation import simulate
from validation.stability import (
    NEUTRAL_TOL,
    characterize,
    classify_eigenvalues,
    fd_crosses_command_corner,
    max_stable_dt,
    summarize,
)

STEPS = {"fe": step_fe, "rk4": step_rk4}


@pytest.mark.parametrize("method", ["fe", "rk4"])
@pytest.mark.parametrize("z", [-0.3, -2.5 + 1.0j, 0.4j, 1.5])
def test_amplification_factor_matches_one_step(method, z):
    """One step of u' = lam u with lam dt = z multiplies u by R(z)."""
    lam, dt = z / 0.1, 0.1
    u1 = STEPS[method](lambda t, u: lam * u, 0.0, 1.0 + 0.0j, dt)[0]
    assert u1 == pytest.approx(amplification_factor(method, z), rel=1e-12)


@pytest.mark.parametrize("z", [-0.3, -2.5, 1.5])
def test_be_amplification_matches_one_step(z):
    """Same for BE (step_be works on real states)."""
    lam, dt = z / 0.1, 0.1
    u1 = step_be(lambda t, u: lam * u, 0.0, 1.0, dt, jac=lambda t, u: lam).u
    assert u1 == pytest.approx(amplification_factor("be", z).real, rel=1e-12)


@pytest.mark.parametrize("z", [-0.3, -0.5 + 0.4j, 0.6j])
def test_ab3_roots_match_recurrence(z):
    """AB3 on u' = lam u: the roots satisfy the characteristic polynomial, and the step-to-step ratio
    tends to the dominant root."""
    q = ab3_characteristic_roots(z)
    assert q.shape == (3,)
    poly = q ** 3 - (1.0 + 23.0 * z / 12.0) * q ** 2 + (16.0 * z / 12.0) * q - 5.0 * z / 12.0
    assert np.max(np.abs(poly)) <= 1e-14
    lam, dt = z, 1.0
    u = [1.0 + 0.0j, 0.9 + 0.1j, 0.7 - 0.2j]  # arbitrary start excites every root
    for _ in range(200):
        u.append(step_ab3(u[-1], (lam * u[-1], lam * u[-2], lam * u[-3]), dt))
    dominant = q[np.argmax(np.abs(q))]
    assert u[-1] / u[-2] == pytest.approx(dominant, rel=1e-8)


def test_ab3_roots_at_zero():
    assert sorted(np.abs(ab3_characteristic_roots(0.0))) == pytest.approx([0.0, 0.0, 1.0], abs=1e-15)
    for method in ("fe", "rk4", "ab3", "be"):
        assert is_absolutely_stable(method, 0.0)


@pytest.mark.parametrize("method, inside, outside", [
    ("fe", -2.0, -2.0 - 1e-9),
    ("rk4", -2.785, -2.786),
    ("ab3", -6.0 / 11.0, -6.0 / 11.0 - 1e-9),
])
def test_real_axis_endpoints(method, inside, outside):
    """Screening values: FE -2, RK4 -2.785, AB3 -6/11."""
    assert is_absolutely_stable(method, inside)
    assert not is_absolutely_stable(method, outside)
    assert is_absolutely_stable(method, 0.5 * inside)


def test_imaginary_axis_extents():
    """AB3 ~0.72, RK4 2 sqrt(2); FE has none; BE covers the whole axis."""
    assert is_absolutely_stable("ab3", 0.72j) and not is_absolutely_stable("ab3", 0.73j)
    assert is_absolutely_stable("rk4", 2.82j) and not is_absolutely_stable("rk4", 2.83j)
    assert not is_absolutely_stable("fe", 0.01j)
    assert all(is_absolutely_stable("be", y * 1j) for y in (0.01, 1.0, 100.0))


def test_be_stable_outside_unit_disk_at_one():
    assert is_absolutely_stable("be", 3.0) and not is_absolutely_stable("be", 0.5)


def test_margin():
    assert is_absolutely_stable("fe", -1.0, margin=0.1)       # |R| = 0
    assert not is_absolutely_stable("fe", -1.95, margin=0.1)  # |R| = 0.95
    assert is_absolutely_stable("ab3", -0.3, margin=0.1) != is_absolutely_stable("ab3", -0.53, margin=0.1)
    with pytest.raises(ValueError):
        amplification_factor("ab3", -0.1)


@pytest.mark.parametrize("lam, method, expected", [
    (-100.0, "fe", 0.02),
    (-30.0 - 40.0j, "fe", 60.0 / 2500.0),       # -2 Re(lam) / |lam|^2
    (-100.0, "ab3", 6.0 / 1100.0),
    (-1e-9 + 100.0j, "ab3", 0.72363 / 100.0),   # imaginary-axis extent
    (-100.0, "rk4", 2.7852936 / 100.0),
    (-1e-9 - 100.0j, "rk4", 2.0 * math.sqrt(2.0) / 100.0),
])
def test_max_stable_dt_values(lam, method, expected):
    assert max_stable_dt(method, lam) == pytest.approx(expected, rel=1e-5)


@pytest.mark.parametrize("lam", [-50.0, -25.85 - 82.25j, -5.85 + 85.6j, -100.85 - 370.3j])
@pytest.mark.parametrize("method", ["fe", "rk4", "ab3"])
def test_max_stable_dt_is_first_exit(lam, method):
    """Every dt up to dt_max is stable at the actual complex z; just beyond it is not."""
    dt_max = max_stable_dt(method, lam)
    assert all(is_absolutely_stable(method, lam * dt) for dt in np.linspace(0.0, dt_max * (1 - 1e-9), 200))
    assert not is_absolutely_stable(method, lam * dt_max * (1 + 1e-6))


def test_max_stable_dt_be_and_invalid():
    assert max_stable_dt("be", -1e6 + 1e6j) == math.inf
    with pytest.raises(ValueError):
        max_stable_dt("rk4", 0.02)


def test_classify_eigenvalues():
    eigs = np.array([-50.0, -1.0 + 2.0j, -1.0 - 2.0j, 0.024, 1e-14, 0.0, -7e-7])
    sets = classify_eigenvalues(eigs)
    assert sets["neutral"].size == 2 and np.all(np.abs(sets["neutral"]) <= NEUTRAL_TOL)
    assert sets["unstable"].tolist() == [0.024]
    assert sets["stable"].size == 4  # includes the slow supply-depletion mode, -7e-7


@pytest.mark.parametrize("method, dt_inside, dt_outside", [
    ("fe", 7.5e-3, 15e-3),
    ("rk4", 10e-3, 15e-3),
    ("ab3", 2.5e-3, 3.75e-3),
])
def test_valve_mode_limit_predicts_explicit_failure(method, dt_inside, dt_outside):
    """tau_v = 5 ms, kappa = 5: the valve mode lambda = -1/tau_v sets the frozen limit (FE 10 ms, RK4 13.9 ms,
    AB3 2.73 ms). A run inside it stays valid; one outside drives the valve opening out of [0, 1]."""
    params = build_params({**STUDY_BASELINE, "tau_v": 0.005, "kappa": 5.0})
    limit = max_stable_dt(method, -1.0 / params.tau_v)
    assert dt_inside < limit < dt_outside
    assert simulate(method, params, {"dt": dt_inside, "t_end": 8.0}).status == "no_apogee"
    outside = simulate(method, params, {"dt": dt_outside, "t_end": 8.0})
    assert outside.status == "physical_model_invalid"
    assert "valve opening" in outside.diagnostics["invalid"]["message"]


@functools.lru_cache(maxsize=None)
def nominal_characterization():
    params = build_params(STUDY_BASELINE)
    result = simulate("rk4", params, {"dt": 2.5e-3})
    assert result.status == "completed"
    return result, characterize(result, params)


def test_characterize_nominal_pilot():
    """Event-based samples, neutral directions reported apart, the valve mode at start, and limits
    consistent with the RK4 pilot that produced the states."""
    result, rows = nominal_characterization()
    p = result.params
    labels = [row["label"] for row in rows]
    for label in ("initialization", "first Pset crossing", "late burn", "just before cutoff", "coast start",
                  "near apogee"):
        assert label in labels
    assert [row["t"] for row in rows] == sorted(row["t"] for row in rows)

    for row in rows:
        assert len(row["eigenvalues"]) == 8
        assert row["neutral_count"] + len(row["stable"]) + len(row["unstable"]) == 8
        assert row["neutral_count"] >= (2 if row["phase"] == "powered" else 5)  # x; helium; frozen coast states
        assert row["dt_max"]["be"] == math.inf
        assert row["dt_max"]["rk4"] > result.diagnostics["dt"]
        for lam in row["unstable"]:
            assert 0.0 < lam[0] < 0.05  # slow trajectory divergence (time scale > 20 s), not a numerical mode
        if row["phase"] == "powered":
            assert not fd_crosses_command_corner(result.powered.U[np.searchsorted(result.powered.t, row["t"])], p)
        if row["analytic_jacobian_rel_diff"] is not None:
            assert row["analytic_jacobian_rel_diff"] < 1e-7

    init = rows[labels.index("initialization")]
    assert init["dt_max"]["fe"] == pytest.approx(2.0 * p.tau_v, rel=1e-6)  # valve mode -1/tau_v, closed valve
    assert init["command"] == "closed"
    crossing = rows[labels.index("first Pset crossing")]
    assert crossing["command"] == "modulating" and crossing["min_damping_ratio"] < 0.5

    summary = summarize(rows)
    for method in ("fe", "rk4", "ab3"):
        assert summary[method]["dt_max"] == min(row["dt_max"][method] for row in rows)
