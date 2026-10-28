"""Study machinery: history sampling and errors, output errors, E_score, timestep selection, and case sets."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

import validation.convergence as convergence
import validation.timestep_selection as timestep_selection
from results.sweeps import (
    CD_VALUES, GAMMA0_VALUES, KAPPA_VALUES, TAU_V_VALUES,
    cd_gamma0_grid_cases, load_selection, regulator_grid_cases,
)
from src.parameters import STUDY_BASELINE, build_params
from src.physics import IMU, IV, derived_quantities, rhs
from src.simulation import Segment, sample_at_time, simulate
from validation.atmosphere_check import rhs_us76
from validation.convergence import (
    REFERENCE_CHANGE_MAX, STUDY_CASES, TOLERANCES,
    add_observed_orders, be_algebraic_check, build_references, error_ratio, history_errors, json_safe,
    load_study_output, nominal_axial_impulse, output_errors, powered_pressure_thrust, sample_segment,
)
from validation.timestep_selection import (
    E_SCORE_MAX, STABILITY_DT_FACTOR,
    e_score, required_cases, select_timestep, verification_cases, verify_finite_grid,
)


@pytest.fixture(scope="module")
def params():
    return build_params(STUDY_BASELINE)


@pytest.fixture(scope="module")
def short_run(params):
    """RK4 at 2.5 ms through cutoff plus a short coast; enough for powered-history checks."""
    return simulate("rk4", params, {"dt": 2.5e-3, "t_end": 8.0})


def test_powered_pressure_thrust_matches_derived_quantities(params, short_run):
    U = short_run.powered.U[::250]
    Pu, T_ax = powered_pressure_thrust(U, params)
    for u, p, t in zip(U, Pu, T_ax):
        q = derived_quantities(u, params, "powered")
        assert p == pytest.approx(q["Pu"], rel=1e-14)
        assert t == pytest.approx(q["T_ax"], rel=1e-14)


def test_sample_segment_matches_sample_at_time(short_run):
    t = np.array([0.0, 1e-4, 0.0025, 1.2345, 3.0, 7.4999, 7.5])
    S = sample_segment(short_run.powered, t)
    for row, tq in zip(S, t):
        np.testing.assert_array_equal(row, sample_at_time(short_run, tq, "powered"))


def test_history_errors_zero_against_itself(params, short_run):
    out = history_errors(short_run.powered, short_run.powered, params)
    assert out["pressure_history"] == 0.0 and out["thrust_history"] == 0.0
    assert out["sampling_converged"]


def test_history_errors_constant_ullage_mass_offset(params, short_run):
    """Adding delta to mu everywhere shifts Pu by delta R Tu / Vu, largest where Vu is smallest (t = 0; the
    ullage grows as propellant drains)."""
    seg = short_run.powered
    delta = 1e-3
    U = seg.U.copy()
    U[:, IMU] += delta
    shifted = Segment(seg.phase, seg.t, U, seg.F)
    out = history_errors(shifted, seg, params)
    expected = delta * params.R_he * params.Tu / params.Vu0
    assert out["pressure_history"] == pytest.approx(expected, rel=1e-9)
    assert out["thrust_history"] == pytest.approx(expected * params.K_F * math.cos(params.delta), rel=1e-9)
    assert out["t_pressure_history"] == 0.0


def test_history_errors_nan_before_cutoff(params, short_run):
    seg = short_run.powered
    truncated = Segment(seg.phase, seg.t[:100], seg.U[:100], seg.F[:100])
    assert math.isnan(history_errors(truncated, seg, params)["pressure_history"])


def test_output_errors(params):
    ref = {k: 1.0 for k in ("apogee_h", "x_eval", "h_eval", "apogee_x", "apogee_t", "peak_specific_force_g",
                            "Pu_min", "Pu_max", "Pu_overshoot")}
    ref["I_ax"] = 4.0e6
    run = {**ref, "apogee_h": 4.0, "Pu_min": 1.0 + 300.0, "Pu_max": 1.0 - 500.0, "I_ax": 4.0e6 + 2.0e3}
    err = output_errors(run, ref, params)
    assert err["apogee_h"] == 3.0 and err["x_eval"] == 0.0
    assert err["pressure_extrema"] == 500.0
    assert err["axial_impulse_rel"] == pytest.approx(2.0e3 / nominal_axial_impulse(params))
    assert nominal_axial_impulse(params) == pytest.approx(533758.0 * 7.5, rel=1e-5)


def test_error_ratio_worst_and_missing():
    errors = {k: 0.1 * tol for k, tol in TOLERANCES.items()}
    errors["apogee_t"] = 0.3 * TOLERANCES["apogee_t"]
    assert error_ratio(errors) == (pytest.approx(0.3), "apogee_t")
    assert error_ratio({**errors, "x_eval": math.nan})[0] == math.inf
    assert error_ratio({**errors, "x_eval": None})[0] == math.inf


def _row(case, method, n_cut, ratio, rhs_calls, status="completed", helium=1e-15, be_failures=0, sampled=True,
         changes=None):
    return {"case": case, "changes": {} if changes is None else changes, "method": method, "n_cut": n_cut,
            "dt": 7.5 / n_cut, "status": status, "errors": {k: ratio * tol for k, tol in TOLERANCES.items()},
            "cost": {"rhs_calls": rhs_calls, "wall_time": 1e-5 * rhs_calls, "be_failures": be_failures},
            "helium_drift": helium, "history_sampling": {"sampling_converged": sampled}, "observed_order": None}


def _limits(dt_ms):
    return {m: {"dt_max": d * 1e-3, "case": "x"} for m, d in dt_ms.items()}


def test_e_score_is_max_over_rows():
    rows = [_row("a", "rk4", 3000, 0.1, 1), _row("b", "rk4", 3000, 0.4, 1)]
    assert e_score(rows) == pytest.approx(0.4)
    assert e_score(rows + [_row("c", "rk4", 3000, math.nan, 1)]) == math.inf


def test_select_timestep_cheapest_passing_with_checks():
    limits = _limits({"fe": 1.37, "rk4": 7.51, "ab3": 1.70, "be": math.inf})
    study = []
    for case in ("a", "b"):
        study += [
            _row(case, "fe", 6000, 0.2, 50_000),                   # inside FE limit, but past the 0.8 margin
            _row(case, "ab3", 3000, 0.1, 25_000, status="physical_model_invalid"),
            _row(case, "ab3", 6000, 0.3 if case == "a" else 0.05, 52_000),
            _row(case, "rk4", 3000, 0.01, 52_000),                 # same cost as AB3, lower E_score: tie-break
            _row(case, "be", 3000, 0.6, 500_000),                   # E_score too large
            _row(case, "be", 6000, 0.3, 1_000_000, helium=1e-7),    # helium drift over the BE limit
        ]
    out = select_timestep(study, dt_limits=limits)
    sel = out["selected"]
    assert (sel["method"], sel["n_cut"]) == ("rk4", 3000)
    assert sel["e_score"] == pytest.approx(0.01) and sel["rhs_calls"] == 104_000
    by = {(c["method"], c["n_cut"]): c for c in out["candidates"]}
    assert not by[("fe", 6000)]["checks"]["stability_margin"]
    assert 1.25 / 1.37 > STABILITY_DT_FACTOR
    assert not by[("ab3", 3000)]["checks"]["completed"]
    assert by[("ab3", 6000)]["passed"] and by[("ab3", 6000)]["worst_case"] == "a"
    assert not by[("be", 3000)]["checks"]["e_score"] and 0.6 > E_SCORE_MAX
    assert not by[("be", 6000)]["checks"]["helium_balance"]
    assert [c["rhs_calls"] for c in out["candidates"]] == sorted(c["rhs_calls"] for c in out["candidates"])


def test_select_timestep_requires_every_case_and_reports_none():
    study = [_row("a", "rk4", 3000, 0.1, 1), _row("b", "rk4", 3000, 0.1, 1), _row("a", "fe", 3000, 0.1, 1)]
    out = select_timestep(study)
    by = {(c["method"], c["n_cut"]): c for c in out["candidates"]}
    assert not by[("fe", 3000)]["checks"]["all_cases"]
    assert out["selected"]["method"] == "rk4"
    assert select_timestep([_row("a", "rk4", 3000, 0.6, 1)])["selected"] is None


def test_select_timestep_requires_converged_history_sampling():
    """Otherwise passing rows fail when a history error comes from unconverged sampling."""
    study = [_row("a", "rk4", 3000, 0.1, 1), _row("b", "rk4", 3000, 0.1, 1, sampled=False)]
    out = select_timestep(study)
    assert out["selected"] is None and not out["candidates"][0]["checks"]["history_sampling"]


def test_add_observed_orders_first_order():
    rows = [_row("a", "fe", n, 3000 / n * 0.1, 1) for n in (3000, 6000, 12000)]
    rows[2]["errors"]["apogee_h"] = 1e-9  # below the round-off floor: no order there
    add_observed_orders(rows)
    assert rows[0]["observed_order"]["pressure_history"] == pytest.approx(1.0)
    assert rows[1]["observed_order"]["apogee_h"] is None
    assert rows[2]["observed_order"] is None


def test_observed_order_floor_follows_reference_uncertainty():
    """An error within ORDER_REFERENCE_FACTOR of the reference's own refinement change gives no order."""
    rows = [_row("a", "rk4", n, 3000 / n * 1e-3, 1) for n in (3000, 6000)]  # pressure errors 8, 4 Pa
    change = {k: 0.0 for k in TOLERANCES}
    add_observed_orders(rows, {"a": {"refinement": {"change": change}}})
    assert rows[0]["observed_order"]["pressure_history"] == pytest.approx(1.0)
    change["pressure_history"] = 0.5  # Pa; 10 x 0.5 = 5 Pa > the 4 Pa error at dt/2
    add_observed_orders(rows, {"a": {"refinement": {"change": change}}})
    assert rows[0]["observed_order"]["pressure_history"] is None
    assert rows[0]["observed_order"]["apogee_h"] == pytest.approx(1.0)


def _fake_run(n_cut, helium=1e-15, status="completed"):
    return {"status": status, "failure": None, "n_cut": n_cut, "dt": 7.5 / n_cut, "cost": {},
            "helium_drift": helium, "be_max_residual": math.nan, "powered": None}


def test_build_references_refuses_unchecked_reference(monkeypatch):
    """A refinement change above REFERENCE_CHANGE_MAX, unconverged sampling, or helium drift raises."""
    def patch(change_ratio=0.0, sampled=True, helium=1e-15):
        errors = {k: change_ratio * tol for k, tol in TOLERANCES.items()}
        monkeypatch.setattr(convergence, "run_jobs", lambda fn, jobs, labels, **kw: [_fake_run(96000, helium),
                                                                                    _fake_run(192000, helium)])
        monkeypatch.setattr(convergence, "compare_runs", lambda *a: (errors, {"sampling_converged": sampled}))

    patch(change_ratio=0.5 * REFERENCE_CHANGE_MAX)
    assert build_references({"nominal": {}})["nominal"]["refinement"]["accepted"]
    for bad in ({"change_ratio": 2 * REFERENCE_CHANGE_MAX}, {"sampled": False}, {"helium": 1e-6}):
        patch(**bad)
        with pytest.raises(RuntimeError):
            build_references({"nominal": {}})


def test_be_algebraic_check_refuses_large_change_or_unconverged_sampling(monkeypatch):
    """A tightening change above REFERENCE_CHANGE_MAX or unconverged history sampling raises."""
    run = {**_fake_run(3000), "cost": {"nonlinear_iters": 1, "rhs_calls": 1}}
    monkeypatch.setattr(convergence, "run_jobs", lambda fn, jobs, labels, **kw: [run, run])

    def patch(change_ratio=0.0, sampled=True):
        errors = {k: change_ratio * tol for k, tol in TOLERANCES.items()}
        monkeypatch.setattr(convergence, "compare_runs", lambda *a: (errors, {"sampling_converged": sampled}))

    patch(change_ratio=0.5 * REFERENCE_CHANGE_MAX)
    check = be_algebraic_check({"nominal": {}})["nominal"]
    assert check["accepted"] and check["history_sampling"]["sampling_converged"]
    for bad in ({"change_ratio": 2 * REFERENCE_CHANGE_MAX}, {"sampled": False}):
        patch(**bad)
        with pytest.raises(RuntimeError, match="BE algebraic"):
            be_algebraic_check({"nominal": {}})


def test_verify_finite_grid_reuses_scored_cases_and_applies_the_selection_rule(monkeypatch):
    """Required-case rows are reused by inputs; only the other cases get references and trials, and a single
    failing grid row fails the grid."""
    built = []

    def fake_references(cases):
        built.extend(cases)
        return {name: {"refinement": {"max_change_ratio": 1e-6}, "n_cut": 96000, "dt": 7.5 / 96000,
                       "status": "completed", "metrics": {}, "cost": {}, "helium_drift": 0.0} for name in cases}

    def fake_study(cases, methods, n_cuts, references, sampled=True):
        return [_row(name, "ab3", 6000, 0.2, 10, changes=ch, sampled=sampled) for name, ch in cases.items()]

    monkeypatch.setattr(timestep_selection, "build_references", fake_references)
    monkeypatch.setattr(timestep_selection, "print_reference_checks", lambda refs: None)
    monkeypatch.setattr(timestep_selection, "convergence_study", fake_study)
    grid = {"tau20_kappa20": {"tau_v": 0.02, "kappa": 20.0}, "tau5_kappa5": {"tau_v": 0.005, "kappa": 5.0},
            "cd0.4_gamma1.1": {"Cd": 0.4, "gamma0": 1.1}}
    scored = [_row("nominal", "ab3", 6000, 0.3, 10), _row("nominal", "rk4", 6000, 9.0, 10),
              _row("tau5_kappa5", "ab3", 6000, 0.1, 10, changes={"tau_v": 0.005, "kappa": 5.0})]
    out = verify_finite_grid("ab3", 6000, grid, scored)
    assert built == ["cd0.4_gamma1.1"] and out["reused_cases"] == ["tau20_kappa20", "tau5_kappa5"]
    assert out["passed"] and out["e_score"] == pytest.approx(0.3) and out["worst_case"] == "tau20_kappa20"
    monkeypatch.setattr(timestep_selection, "convergence_study",
                        lambda *a: fake_study(*a, sampled=False))
    out = verify_finite_grid("ab3", 6000, grid, scored)
    assert not out["passed"] and not out["checks"]["history_sampling"]


def _write(path, report):
    path.write_text(json.dumps(json_safe(report)))
    return str(path)


def test_saved_outputs_must_match_the_study_baseline(tmp_path):
    good = _write(tmp_path / "good.json", {"study_baseline": STUDY_BASELINE, "x": 1})
    assert load_study_output(good, "m")["x"] == 1
    for report in ({"x": 1}, {"study_baseline": {**STUDY_BASELINE, "Cd": 0.61}}):
        with pytest.raises(RuntimeError, match="STUDY_BASELINE"):
            load_study_output(_write(tmp_path / "stale.json", report), "m")


def test_load_selection_refuses_failed_verification(tmp_path):
    selected = {"method": "ab3", "dt": 1.25e-3, "n_cut": 6000}
    ok = _write(tmp_path / "ok.json", {"study_baseline": STUDY_BASELINE, "selected": selected,
                                       "verification": {"passed": True}})
    assert load_selection(ok) == selected
    failed = _write(tmp_path / "failed.json", {"study_baseline": STUDY_BASELINE, "selected": selected,
                                               "verification": {"passed": False}})
    with pytest.raises(RuntimeError, match="full-grid"):
        load_selection(failed)


def test_required_cases_match_the_grids():
    cases = required_cases()
    assert len(cases) == 10
    taus = {c.get("tau_v") for c in STUDY_CASES.values()} - {None}
    kappas = {c.get("kappa") for c in STUDY_CASES.values()} - {None}
    assert taus == {min(TAU_V_VALUES), max(TAU_V_VALUES)} and kappas == {min(KAPPA_VALUES), max(KAPPA_VALUES)}
    assert cases["fixed_orifice"] == {"scenario": "fixed_orifice"}
    assert {(c["Cd"], c["gamma0"]) for c in cases.values() if "Cd" in c} == \
        {(cd, g) for cd in (min(CD_VALUES), max(CD_VALUES)) for g in (min(GAMMA0_VALUES), max(GAMMA0_VALUES))}
    for changes in cases.values():
        build_params({**STUDY_BASELINE, **changes})


def test_grid_and_verification_case_sets():
    reg = regulator_grid_cases(STUDY_BASELINE)
    cd = cd_gamma0_grid_cases(STUDY_BASELINE)
    assert len(reg) == 25 and len(cd) == 9
    assert len({c["name"] for c in reg + cd}) == 34
    assert "tau20_kappa20" in {c["name"] for c in reg} and "cd0.6_gamma1.1" in {c["name"] for c in cd}
    # The nominal case appears in both grids; the verification set keeps one copy, plus the fixed orifice.
    assert len(verification_cases()) == 25 + 8 + 1


def test_rhs_us76_changes_only_the_speed_rate(params, short_run):
    u = short_run.powered.U[1000]
    a, b = rhs(0.0, u, params, "powered"), rhs_us76(0.0, u, params, "powered")
    mask = np.arange(8) != IV
    np.testing.assert_array_equal(a[mask], b[mask])
    assert b[IV] != a[IV]
