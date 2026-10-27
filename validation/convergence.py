"""Error targets, reference solutions, history errors, observed orders, and measured cost.

Run: python -m validation.convergence
"""

from __future__ import annotations

import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from src.parameters import STUDY_BASELINE, build_params
from src.physics import IMP, IMU
from src.simulation import Segment, compute_metrics, hermite_interpolate, simulate

# Error targets (SI units). Pressure and thrust are absolute equivalents of 0.1%; the impulse
# target is relative to the nominal axial impulse K_F Pset cos(delta) t_cut. Helium balance is a validity
# check (HELIUM_DRIFT_LIMIT), not an E_score output.
TOLERANCES: Mapping[str, float] = {
    "pressure_history": 8.0e3,       # Pa, 0.1% of Pset
    "pressure_extrema": 8.0e3,       # Pa; powered Pu minimum, maximum, and overshoot peak
    "thrust_history": 534.0,         # N, 0.1% of nominal axial force
    "apogee_h": 10.0,                # m
    "x_eval": 10.0,                  # m
    "h_eval": 10.0,                  # m
    "apogee_x": 10.0,                # m
    "apogee_t": 0.05,                # s
    "axial_impulse_rel": 1.0e-3,     # relative to nominal axial impulse
    "peak_specific_force_g": 0.01,   # g0
}
HELIUM_DRIFT_LIMIT: Mapping[str, float] = {"fe": 1e-10, "rk4": 1e-10, "ab3": 1e-10, "be": 1e-8}  # relative
EXTREMA_KEYS = ("Pu_min", "Pu_max", "Pu_overshoot")
_DIRECT_KEYS = ("apogee_h", "x_eval", "h_eval", "apogee_x", "apogee_t", "peak_specific_force_g")

# Halving sequence of steps that each divide t_cut = 7.5 s (dt = 2.5 ms / 2^j), starting at tau_v,min / 2.
# FE and BE still miss the pressure-history target at 0.156 ms, so their sequences continue until they pass.
N_CUT_VALUES = (3000, 6000, 12000, 24000, 48000)
EXTRA_N_CUT_VALUES: Mapping[str, tuple[int, ...]] = {"fe": (96000, 192000), "be": (96000,)}
# RK4 reference at dt = 78.125 us, checked by one more halving. Its refinement change is
# ~1e-6 of every tolerance, so it also serves the FE/BE trials at the same or a finer step.
REFERENCE_N_CUT = 96000
REFERENCE_CHANGE_MAX = 0.05   # refining the reference may move each output by at most this fraction of its tolerance
SAMPLING_CHANGE_MAX = 0.05    # history sampling is refined until E_P and E_T move by less than this fraction
SAMPLES_PER_STEP_MAX = 64
EXTREMUM_POINTS = 64          # extra samples on each side of a pressure extremum, spread over one trial step
# Observed orders only where both errors exceed ORDER_FLOOR of the tolerance and ORDER_REFERENCE_FACTOR times the
# reference's own change on refinement, so neither round-off nor the reference's error sets the slope.
ORDER_FLOOR = 1e-6
ORDER_REFERENCE_FACTOR = 10.0
# BE algebraic tightening. It passes when the tight settings move every output by at most
# REFERENCE_CHANGE_MAX of its tolerance. A Newton step_tol of 1e-15 did not stop in the tested cases (round-off).
BE_TIGHT_OPTIONS: Mapping[str, Any] = {"be_residual_tol": 1e-12, "be_solver_options": {"step_tol": 1e-14, "max_iter": 50}}
BE_CHECK_N_CUT = 3000         # the coarsest trial step, where each Newton solve does the most work

# Screening cases: the nominal case and the four regulator corners.
STUDY_CASES: Mapping[str, Mapping[str, Any]] = {
    "nominal": {},
    "tau5_kappa5": {"tau_v": 0.005, "kappa": 5.0},
    "tau5_kappa100": {"tau_v": 0.005, "kappa": 100.0},
    "tau100_kappa5": {"tau_v": 0.1, "kappa": 5.0},
    "tau100_kappa100": {"tau_v": 0.1, "kappa": 100.0},
}
METHODS = ("fe", "rk4", "ab3", "be")
RHS_CALLS_PER_STEP = {"fe": 1, "ab3": 1, "rk4": 4, "be": 20}  # rough, only to submit the longest jobs first
WORKERS = max(1, (os.cpu_count() or 2) // 2)
OUTPUT_PATH = os.path.join("results", "data", "convergence.json")


def run_case(inputs: Mapping[str, Any], method: str, n_cut: int,
             options: Optional[Mapping[str, Any]] = None) -> dict[str, Any]:
    """Simulate one full input set at dt = t_cut / n_cut; return a picklable summary: status, metrics, cost
    counters, validity diagnostics, and the powered segment (for history errors)."""
    params = build_params(inputs)
    result = simulate(method, params, {**(options or {}), "dt": params.t_cut / n_cut})
    failure = result.diagnostics["invalid"] or result.diagnostics["solver_failure"]
    n_steps = sum(seg.t.size - 1 for seg in (result.powered, result.coast) if seg is not None)
    return {
        "method": method, "n_cut": n_cut, "dt": result.diagnostics["dt"], "status": result.status,
        "metrics": compute_metrics(result),
        "cost": {**result.counters, "n_steps": n_steps},
        "helium_drift": result.diagnostics["helium_drift"],
        "be_max_residual": result.diagnostics["be_max_residual"],
        "failure": None if failure is None else {k: failure[k] for k in ("phase", "t_last_accepted", "message")},
        "powered": result.powered,
    }


def run_jobs(fn: Callable, jobs: Sequence[tuple], labels: Sequence[str],
             weights: Optional[Sequence[float]] = None, workers: int = WORKERS) -> list[Any]:
    """fn(*job) for every job in a process pool, heaviest first; results keep the order of jobs."""
    order = sorted(range(len(jobs)), key=lambda i: -weights[i]) if weights else range(len(jobs))
    results: list[Any] = [None] * len(jobs)
    with ProcessPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        futures = {pool.submit(fn, *jobs[i]): i for i in order}
        for done, fut in enumerate(as_completed(futures), 1):
            i = futures[fut]
            results[i] = fut.result()
            print(f"  [{done}/{len(jobs)}] {labels[i]}", flush=True)
    return results


def powered_pressure_thrust(U: np.ndarray, params: Any) -> tuple[np.ndarray, np.ndarray]:
    """Pu and T_ax at powered states (rows of U), vectorized from the formulas of derived_quantities."""
    Pu = U[:, IMU] * params.R_he * params.Tu / (params.Vtank - U[:, IMP] / params.rho_p)
    return Pu, params.K_F * math.cos(params.delta) * Pu


def sample_segment(seg: Segment, t_query: np.ndarray) -> np.ndarray:
    """States at times inside one phase segment from its per-step cubic Hermite interpolant (vectorized
    sample_at_time; never crosses t_cut)."""
    k = np.clip(np.searchsorted(seg.t, t_query, side="right") - 1, 0, seg.t.size - 2)
    return hermite_interpolate(seg.t[k, np.newaxis], seg.t[k + 1, np.newaxis], seg.U[k], seg.U[k + 1],
                               seg.F[k], seg.F[k + 1], t_query[:, np.newaxis])


def _extremum_times(seg: Segment, params: Any) -> list[float]:
    """Times of the stored powered Pu minimum and of the overshoot peak after the first Pset crossing.
    The maximum is the initial state, which every run has exactly."""
    Pu, _ = powered_pressure_thrust(seg.U, params)
    times = [float(seg.t[np.argmin(Pu)])]
    crossed = np.flatnonzero(Pu <= params.Pset)
    if crossed.size:
        times.append(float(seg.t[crossed[0] + np.argmax(Pu[crossed[0]:])]))
    return times


def _powered_complete(seg: Segment, t_cut: float) -> bool:
    return seg.t.size >= 2 and abs(seg.t[-1] - t_cut) <= 1e-9 * t_cut


def history_errors(result: Segment, reference: Segment, params: Any,
                   tolerances: Mapping[str, float] = TOLERANCES) -> dict[str, Any]:
    """E_P and E_T: max |Pu - Pu_ref| and |T_ax - T_ax_ref| over 0 <= t <= t_cut.

    Both powered segments are sampled at the same physical times through their own Hermite interpolants, so
    values at t_cut are powered-side. The shared grid splits every step of `result` into k equal parts and adds
    dense points within one of its steps of the pressure minimum and overshoot peak of either run; k doubles
    until E_P and E_T change by less than SAMPLING_CHANGE_MAX of their tolerances. NaN unless both runs
    reached t_cut.
    """
    nan = {"pressure_history": math.nan, "thrust_history": math.nan, "t_pressure_history": math.nan,
           "samples_per_step": None, "sampling_converged": False}
    if not (_powered_complete(result, params.t_cut) and _powered_complete(reference, params.t_cut)):
        return nan
    step = params.t_cut / (result.t.size - 1)
    extra = np.concatenate([np.linspace(te - step, te + step, 2 * EXTREMUM_POINTS + 1)
                            for te in _extremum_times(result, params) + _extremum_times(reference, params)])
    extra = extra[(extra >= 0.0) & (extra <= params.t_cut)]
    widths = np.diff(result.t)[:, np.newaxis]

    previous = None
    k = 1
    while True:
        grid = np.concatenate([(result.t[:-1, np.newaxis] + widths * (np.arange(k) / k)).ravel(),
                               result.t[-1:], extra])
        P, T = powered_pressure_thrust(sample_segment(result, grid), params)
        P_ref, T_ref = powered_pressure_thrust(sample_segment(reference, grid), params)
        dP, dT = np.abs(P - P_ref), np.abs(T - T_ref)
        i = int(np.argmax(dP))
        current = (float(dP[i]), float(np.max(dT)))
        converged = previous is not None and \
            current[0] - previous[0] <= SAMPLING_CHANGE_MAX * tolerances["pressure_history"] and \
            current[1] - previous[1] <= SAMPLING_CHANGE_MAX * tolerances["thrust_history"]
        if converged or k >= SAMPLES_PER_STEP_MAX:
            return {"pressure_history": current[0], "thrust_history": current[1],
                    "t_pressure_history": float(grid[i]), "samples_per_step": k, "sampling_converged": converged}
        previous = current
        k *= 2


def nominal_axial_impulse(params: Any) -> float:
    """Nominal axial force at Pset times t_cut, the base of the 0.1% impulse target."""
    return params.K_F * params.Pset * math.cos(params.delta) * params.t_cut


def output_errors(metrics: Mapping[str, Any], reference: Mapping[str, Any], params: Any) -> dict[str, float]:
    """|Q - Q_ref| for the scalar outputs in TOLERANCES. Pressure extrema are compared separately (Pu_min, Pu_max,
    Pu_overshoot) and combined as their maximum; impulse is relative to the nominal axial impulse. NaN where
    either run lacks the output."""
    err = {k: abs(metrics[k] - reference[k]) for k in _DIRECT_KEYS + EXTREMA_KEYS}
    err["pressure_extrema"] = float(np.max([err[k] for k in EXTREMA_KEYS]))
    err["axial_impulse_rel"] = abs(metrics["I_ax"] - reference["I_ax"]) / nominal_axial_impulse(params)
    return {k: float(v) for k, v in err.items()}


def compare_runs(run: Mapping[str, Any], reference: Mapping[str, Any], params: Any) -> tuple[dict, dict]:
    """(errors, sampling): output and history errors of one run_case summary against another."""
    hist = history_errors(run["powered"], reference["powered"], params)
    errors = {**output_errors(run["metrics"], reference["metrics"], params),
              "pressure_history": hist.pop("pressure_history"), "thrust_history": hist.pop("thrust_history")}
    return errors, hist


def error_ratio(errors: Mapping[str, Any], tolerances: Mapping[str, float] = TOLERANCES) -> tuple[float, str]:
    """Largest |Q - Q_ref| / tol_Q over the tolerance keys and the output that sets it; a missing error
    (None or NaN, e.g. a failed run) counts as inf."""
    worst, worst_key = -math.inf, ""
    for key, tol in tolerances.items():
        e = errors.get(key)
        ratio = math.inf if e is None or not math.isfinite(e) else e / tol
        if ratio > worst:
            worst, worst_key = ratio, key
    return worst, worst_key


def _check_entry(change: Mapping[str, Any]) -> dict[str, Any]:
    ratio, key = error_ratio(change)
    return {"change": change, "max_change_ratio": ratio, "max_change_output": key}


def build_references(cases: Mapping[str, Mapping[str, Any]], n_cut: int = REFERENCE_N_CUT) -> dict[str, dict]:
    """Self-written RK4 at t_cut / n_cut for each case, refined once more at half the step.

    Each reference carries the change of every E_score output on refinement, absolute and as a fraction of its
    tolerance; it is accepted when no change exceeds REFERENCE_CHANGE_MAX and the history sampling converged.
    Raises if a reference run fails, drifts in helium beyond the RK4 limit, or is not accepted, so no trial is
    ever scored against an unchecked reference.
    """
    names = list(cases)
    jobs = [({**STUDY_BASELINE, **cases[name]}, "rk4", m) for name in names for m in (n_cut, 2 * n_cut)]
    labels = [f"reference {name} rk4 n_cut={m}" for name in names for m in (n_cut, 2 * n_cut)]
    runs = run_jobs(run_case, jobs, labels, weights=[job[2] for job in jobs])
    refs, rejected = {}, []
    for i, name in enumerate(names):
        ref, fine = runs[2 * i], runs[2 * i + 1]
        for run in (ref, fine):
            if run["status"] != "completed":
                raise RuntimeError(f"reference run {name} n_cut={run['n_cut']} ended with {run['status']}: {run['failure']}")
            if not run["helium_drift"] <= HELIUM_DRIFT_LIMIT["rk4"]:
                raise RuntimeError(f"reference run {name} n_cut={run['n_cut']} helium drift {run['helium_drift']:.2e}")
        change, sampling = compare_runs(fine, ref, build_params(jobs[2 * i][0]))
        check = _check_entry(change)
        accepted = check["max_change_ratio"] <= REFERENCE_CHANGE_MAX and sampling["sampling_converged"]
        ref["refinement"] = {"n_cut": fine["n_cut"], "dt": fine["dt"], **check, "history_sampling": sampling,
                             "accepted": accepted, "cost": fine["cost"]}
        refs[name] = ref
        if not accepted:
            rejected.append(f"{name}: change {check['max_change_ratio']:.3g} tol ({check['max_change_output']}), "
                            f"sampling converged {sampling['sampling_converged']}")
    if rejected:
        raise RuntimeError(f"reference refinement check failed (limit {REFERENCE_CHANGE_MAX} tol): {'; '.join(rejected)}")
    return refs


def be_algebraic_check(cases: Mapping[str, Mapping[str, Any]], n_cut: int = BE_CHECK_N_CUT) -> dict[str, dict]:
    """BE at the default and at the BE_TIGHT_OPTIONS algebraic settings; the change in every output, as a
    fraction of its tolerance, shows whether the nonlinear solve contributes to the error.

    A case is accepted when both runs complete, no output moves by more than REFERENCE_CHANGE_MAX of its
    tolerance, and the history sampling of the comparison converged; raises if any case is not accepted.
    """
    names = list(cases)
    jobs = [({**STUDY_BASELINE, **cases[name]}, "be", n_cut, opts) for name in names for opts in ({}, BE_TIGHT_OPTIONS)]
    labels = [f"BE check {name} {tag}" for name in names for tag in ("default", "tight")]
    runs = run_jobs(run_case, jobs, labels)
    out, rejected = {}, []
    for i, name in enumerate(names):
        base, tight = runs[2 * i], runs[2 * i + 1]
        change, sampling = compare_runs(tight, base, build_params(jobs[2 * i][0]))
        check = _check_entry(change)
        accepted = (base["status"] == tight["status"] == "completed" and check["max_change_ratio"] <= REFERENCE_CHANGE_MAX
                    and sampling["sampling_converged"])
        out[name] = {
            "n_cut": n_cut, "dt": base["dt"], "tight_options": BE_TIGHT_OPTIONS, **check, "history_sampling": sampling,
            "accepted": accepted,
            "status": {"default": base["status"], "tight": tight["status"]},
            "be_max_residual": {"default": base["be_max_residual"], "tight": tight["be_max_residual"]},
            "nonlinear_iters": {"default": base["cost"]["nonlinear_iters"], "tight": tight["cost"]["nonlinear_iters"]},
            "rhs_calls": {"default": base["cost"]["rhs_calls"], "tight": tight["cost"]["rhs_calls"]},
            "helium_drift": {"default": base["helium_drift"], "tight": tight["helium_drift"]},
        }
        if not accepted:
            rejected.append(f"{name}: status {out[name]['status']}, change {check['max_change_ratio']:.3g} tol, "
                            f"sampling converged {sampling['sampling_converged']}")
    if rejected:
        raise RuntimeError(f"BE algebraic tightening check failed: {'; '.join(rejected)}")
    return out


def trial_row(case: str, changes: Mapping[str, Any], run: Mapping[str, Any], reference: Mapping[str, Any]) -> dict[str, Any]:
    """One method/case/dt row: status, errors against the reference, the worst error ratio, cost, and validity."""
    errors, sampling = compare_runs(run, reference, build_params({**STUDY_BASELINE, **changes}))
    score, worst = error_ratio(errors)
    return {
        "case": case, "changes": dict(changes), "method": run["method"], "n_cut": run["n_cut"], "dt": run["dt"],
        "status": run["status"], "errors": errors, "score": score, "worst_output": worst,
        "history_sampling": sampling, "observed_order": None, "cost": run["cost"],
        "helium_drift": run["helium_drift"], "be_max_residual": run["be_max_residual"], "failure": run["failure"],
        "metrics": run["metrics"],
    }


def add_observed_orders(rows: Sequence[dict[str, Any]],
                        references: Optional[Mapping[str, Mapping[str, Any]]] = None,
                        tolerances: Mapping[str, float] = TOLERANCES) -> None:
    """observed_order[q] = log2(E_q(dt) / E_q(dt/2)) on each row that has a dt/2 partner (same case and method),
    kept only where both errors exceed ORDER_FLOOR of the tolerance and, given the references, ORDER_REFERENCE_FACTOR
    times that case's reference change on refinement; below that the reference and round-off set the slope."""
    by_key = {(r["case"], r["method"], r["n_cut"]): r for r in rows}
    for row in rows:
        finer = by_key.get((row["case"], row["method"], 2 * row["n_cut"]))
        if finer is None:
            continue
        ref_change = references[row["case"]]["refinement"]["change"] if references else {}
        orders = {}
        for q, tol in tolerances.items():
            floor = max(ORDER_FLOOR * tol, ORDER_REFERENCE_FACTOR * (ref_change.get(q) or 0.0))
            e0, e1 = row["errors"][q], finer["errors"][q]
            usable = all(e is not None and math.isfinite(e) and e > floor for e in (e0, e1))
            orders[q] = math.log2(e0 / e1) if usable else None
        row["observed_order"] = orders


def trial_n_cut_values(method: str) -> tuple[int, ...]:
    return N_CUT_VALUES + EXTRA_N_CUT_VALUES.get(method, ())


def convergence_study(cases: Mapping[str, Mapping[str, Any]], methods: Sequence[str],
                      n_cut_values: Mapping[str, Sequence[int]],
                      references: Mapping[str, Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One row per method/case/dt (n_cut_values[method]): status, history and output errors, observed order,
    RHS/Jacobian calls, AB3 restarts, nonlinear iterations, wall time."""
    keys = [(name, method, n) for name in cases for method in methods for n in n_cut_values[method]]
    jobs = [({**STUDY_BASELINE, **cases[name]}, method, n) for name, method, n in keys]
    labels = [f"{name} {method} n_cut={n}" for name, method, n in keys]
    runs = run_jobs(run_case, jobs, labels, weights=[RHS_CALLS_PER_STEP[m] * n for _, m, n in keys])
    rows = [trial_row(name, cases[name], run, references[name]) for (name, _, _), run in zip(keys, runs)]
    add_observed_orders(rows, references)
    return rows


def json_safe(obj: Any) -> Any:
    """Plain JSON types; non-finite floats become null (a failed run, or nothing to limit)."""
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return float(obj) if math.isfinite(obj) else None
    if isinstance(obj, Mapping):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj


def load_study_output(path: str, producer: str) -> dict[str, Any]:
    """A saved study report, refused unless it was computed from the current STUDY_BASELINE (a changed input
    invalidates every downstream study)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found; run python -m {producer} first")
    with open(path, encoding="utf-8") as fh:
        report = json.load(fh)
    if report.get("study_baseline") != json_safe(STUDY_BASELINE):
        raise RuntimeError(f"{path} was not computed from the current STUDY_BASELINE; rerun python -m {producer}")
    return report


def reference_summary(references: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """The JSON part of each reference: run settings, metrics, cost, and the refinement check."""
    return {name: {k: ref[k] for k in ("n_cut", "dt", "status", "metrics", "cost", "helium_drift", "refinement")}
            for name, ref in references.items()}


def print_reference_checks(references: Mapping[str, Mapping[str, Any]]) -> None:
    for name, ref in references.items():
        check = ref["refinement"]
        print(f"  {name:<22} apogee {ref['metrics']['apogee_h']:.3f} m, max change on refinement "
              f"{check['max_change_ratio']:.2e} tol ({check['max_change_output']}) "
              f"{'ok' if check['accepted'] else 'TOO LARGE'}")


def _fmt(x: Optional[float], spec: str = ".2e") -> str:
    return "-" if x is None or not math.isfinite(x) else format(x, spec)


def main() -> None:
    """All four methods on the nominal case and the regulator corners against RK4 references, plus the BE
    algebraic tightening check; save rows to results/data/convergence.json."""
    print(f"RK4 references at n_cut = {REFERENCE_N_CUT} and {2 * REFERENCE_N_CUT} ({WORKERS} workers)")
    references = build_references(STUDY_CASES)
    print_reference_checks(references)

    print(f"BE algebraic tightening check at n_cut = {BE_CHECK_N_CUT}")
    be_check = be_algebraic_check({k: STUDY_CASES[k] for k in ("nominal", "tau5_kappa100")})
    for name, check in be_check.items():
        print(f"  {name:<22} max change {check['max_change_ratio']:.2e} tol ({check['max_change_output']}), "
              f"residual {check['be_max_residual']['default']:.1e} -> {check['be_max_residual']['tight']:.1e}")

    n_cut_values = {method: trial_n_cut_values(method) for method in METHODS}
    print(f"Trials: {len(STUDY_CASES)} cases x {sum(map(len, n_cut_values.values()))} method/step pairs")
    rows = convergence_study(STUDY_CASES, METHODS, n_cut_values, references)

    for name in STUDY_CASES:
        print(f"\n== {name} {dict(STUDY_CASES[name]) or ''}: worst |Q - Q_ref| / tol (output), "
              f"order of E_P and apogee, RHS calls")
        for row in (r for r in rows if r["case"] == name):
            order = row["observed_order"] or {}
            status = "" if row["status"] == "completed" else f"  {row['status']}"
            print(f"  {row['method']:>3} dt = {row['dt'] * 1e3:7.4f} ms  score {_fmt(row['score'])} "
                  f"({row['worst_output'] if math.isfinite(row['score']) else '-'})  "
                  f"p_P {_fmt(order.get('pressure_history'), '.2f'):>5}  p_apo {_fmt(order.get('apogee_h'), '.2f'):>5}  "
                  f"rhs {row['cost']['rhs_calls']:>9d}{status}")

    report = {
        "study_baseline": STUDY_BASELINE,
        "tolerances": TOLERANCES,
        "helium_drift_limit": HELIUM_DRIFT_LIMIT,
        "n_cut_values": n_cut_values,
        "methods": METHODS,
        "cases": STUDY_CASES,
        "workers": WORKERS,
        "reference": {"method": "rk4", "n_cut": REFERENCE_N_CUT, "change_max": REFERENCE_CHANGE_MAX,
                      "cases": reference_summary(references)},
        "be_algebraic_check": be_check,
        "rows": rows,
    }
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(json_safe(report), fh, indent=1)
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
