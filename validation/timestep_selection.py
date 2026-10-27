"""E_score and selection of one method and one nominal dt.

Run: python -m validation.timestep_selection (after validation.stability and validation.convergence)
"""

from __future__ import annotations

import json
import math
import os
from collections import defaultdict
from typing import Any, Mapping, Optional, Sequence

from results.sweeps import (
    CD_VALUES, GAMMA0_VALUES,
    cd_gamma0_grid_cases, fixed_orifice_case, regulator_grid_cases,
)
from src.parameters import STUDY_BASELINE
from validation.convergence import (
    HELIUM_DRIFT_LIMIT,
    METHODS,
    OUTPUT_PATH as CONVERGENCE_PATH,
    STUDY_CASES,
    TOLERANCES,
    build_references,
    convergence_study,
    error_ratio,
    json_safe,
    load_study_output,
    print_reference_checks,
    reference_summary,
)
from validation.stability import OUTPUT_PATH as STABILITY_PATH

E_SCORE_MAX = 0.5
# Interior stability margin, applied as a factor on dt: dt <= 0.8 x the smallest frozen-Jacobian dt_max over
# the stability pilots. An engineering choice, not a derived constant.
STABILITY_DT_FACTOR = 0.8
OUTPUT_PATH = os.path.join("results", "data", "timestep_selection.json")


def required_cases() -> dict[str, dict[str, Any]]:
    """Nominal; regulator corners tau_v {5, 100} ms x kappa {5, 100}; fixed orifice; Cd x gamma0 corners."""
    corners = {f"cd{cd:g}_gamma{gamma0:g}": {"Cd": cd, "gamma0": gamma0}
               for cd in (min(CD_VALUES), max(CD_VALUES)) for gamma0 in (min(GAMMA0_VALUES), max(GAMMA0_VALUES))}
    return {**STUDY_CASES, "fixed_orifice": dict(fixed_orifice_case(STUDY_BASELINE)["changes"]), **corners}


def verification_cases() -> dict[str, dict[str, Any]]:
    """The complete finite case set: both grids and the fixed orifice, one entry per distinct input set."""
    cases: dict[str, dict[str, Any]] = {}
    seen = set()
    for case in (regulator_grid_cases(STUDY_BASELINE) + cd_gamma0_grid_cases(STUDY_BASELINE)
                 + [fixed_orifice_case(STUDY_BASELINE)]):
        key = tuple(sorted(case["inputs"].items()))
        if key not in seen:
            seen.add(key)
            cases[case["name"]] = case["changes"]
    return cases


def e_score(rows: Sequence[Mapping[str, Any]], tolerances: Mapping[str, float] = TOLERANCES) -> float:
    """max over tested cases and required outputs of |Q_dt - Q_ref| / tol_Q; inf if any output is missing."""
    return max((error_ratio(row["errors"], tolerances)[0] for row in rows), default=math.inf)


def stability_dt_limits(path: str = STABILITY_PATH) -> dict[str, dict[str, Any]]:
    """Smallest frozen-Jacobian dt_max per method over the stability pilot cases, and the case that sets it."""
    report = load_study_output(path, "validation.stability")
    limits = {}
    for method in METHODS:
        per_case = {name: case["summary"][method]["dt_max"] for name, case in report["cases"].items()}
        name = min(per_case, key=lambda k: math.inf if per_case[k] is None else per_case[k])
        limits[method] = {"dt_max": math.inf if per_case[name] is None else per_case[name], "case": name}
    return limits


def select_timestep(study: Sequence[Mapping[str, Any]], tolerances: Mapping[str, float] = TOLERANCES,
                    dt_limits: Optional[Mapping[str, Mapping[str, Any]]] = None) -> dict[str, Any]:
    """Choose method/dt with E_score <= 0.5, no invalid states, successful solves, acceptable cost;
    return the choice and the worst-case evidence row.

    A method/dt passes when its rows cover every case in the study, all end with status completed and no
    rejected BE step, helium drift stays within HELIUM_DRIFT_LIMIT, every history error comes from converged
    sampling, E_score <= E_SCORE_MAX, and (given dt_limits) dt <= STABILITY_DT_FACTOR x the frozen-Jacobian
    limit. The passing candidate with the fewest
    total RHS calls over the cases is chosen (every call is counted, including BE finite-difference
    columns); E_score breaks ties. "selected" is None if nothing passes. Candidates are sorted by cost.
    """
    cases = sorted({row["case"] for row in study})
    groups: dict[tuple[str, int], list] = defaultdict(list)
    for row in study:
        groups[(row["method"], row["n_cut"])].append(row)

    candidates = []
    for (method, n_cut), rows in groups.items():
        dt = rows[0]["dt"]
        worst = max(rows, key=lambda r: error_ratio(r["errors"], tolerances)[0])
        score, output = error_ratio(worst["errors"], tolerances)
        limit = None if dt_limits is None else dt_limits[method]["dt_max"]
        checks = {
            "all_cases": sorted(r["case"] for r in rows) == cases,
            "completed": all(r["status"] == "completed" for r in rows),
            "nonlinear_solves": all(not r["cost"]["be_failures"] for r in rows),
            "helium_balance": all(r["helium_drift"] is not None and r["helium_drift"] <= HELIUM_DRIFT_LIMIT[method]
                                  for r in rows),
            "history_sampling": all(r["history_sampling"]["sampling_converged"] for r in rows),
            "e_score": score <= E_SCORE_MAX,
            "stability_margin": limit is None or dt <= STABILITY_DT_FACTOR * limit,
        }
        candidates.append({
            "method": method, "n_cut": n_cut, "dt": dt, "e_score": score,
            "worst_case": worst["case"], "worst_output": output,
            "rhs_calls": sum(r["cost"]["rhs_calls"] for r in rows),
            "wall_time": sum(r["cost"]["wall_time"] for r in rows),
            "dt_over_stability_limit": None if limit is None else dt / limit,
            "checks": checks, "passed": all(checks.values()),
        })
    candidates.sort(key=lambda c: (c["rhs_calls"], c["e_score"]))

    passing = [c for c in candidates if c["passed"]]
    selected = None
    if passing:
        choice = passing[0]
        rows = groups[(choice["method"], choice["n_cut"])]
        evidence = next(r for r in rows if r["case"] == choice["worst_case"])
        selected = {**{k: choice[k] for k in ("method", "n_cut", "dt", "e_score", "worst_case", "worst_output",
                                              "rhs_calls", "wall_time", "dt_over_stability_limit")},
                    "cases": cases, "evidence": evidence}
    return {"selected": selected, "candidates": candidates}


def _inputs_key(changes: Mapping[str, Any]) -> tuple:
    return tuple(sorted({**STUDY_BASELINE, **changes}.items()))


def verify_finite_grid(method: str, n_cut: int, cases: Mapping[str, Mapping[str, Any]],
                       scored_rows: Sequence[Mapping[str, Any]] = (),
                       dt_limits: Optional[Mapping[str, Mapping[str, Any]]] = None) -> dict[str, Any]:
    """Apply the selection rule to the chosen method/dt on every case of the complete finite grid, each against its
    own refined RK4 reference.

    Rows of this method/dt already scored against a reference with the same inputs (the required cases) are reused
    under the grid name; the other cases get new references, which build_references refuses unless their
    refinement check passes. The grid passes when select_timestep passes the method/dt on all of its rows.
    """
    scored = {_inputs_key(row["changes"]): row for row in scored_rows
              if (row["method"], row["n_cut"]) == (method, n_cut)}
    reused = {name: changes for name, changes in cases.items() if _inputs_key(changes) in scored}
    new = {name: changes for name, changes in cases.items() if name not in reused}
    references = build_references(new) if new else {}
    if new:
        print_reference_checks(references)
    rows = convergence_study(new, [method], {method: [n_cut]}, references) if new else []
    rows += [{**scored[_inputs_key(changes)], "case": name} for name, changes in reused.items()]
    candidate = select_timestep(rows, dt_limits=dt_limits)["candidates"][0]
    return {"method": method, "n_cut": n_cut, "dt": candidate["dt"], "cases": sorted(cases),
            "e_score": candidate["e_score"], "worst_case": candidate["worst_case"],
            "worst_output": candidate["worst_output"], "checks": candidate["checks"], "passed": candidate["passed"],
            "rhs_calls": candidate["rhs_calls"], "reused_cases": sorted(reused),
            "max_reference_change_ratio": max((ref["refinement"]["max_change_ratio"] for ref in references.values()),
                                              default=None),
            "references": reference_summary(references), "rows": rows}


def _fmt(x: Optional[float], spec: str = ".3g") -> str:
    return "inf" if x is None or not math.isfinite(x) else format(x, spec)


def _print_candidates(candidates: Sequence[Mapping[str, Any]]) -> None:
    print(f"  {'method':>6}{'dt ms':>9}{'E_score':>10}  {'worst case / output':<42}{'RHS calls':>11}"
          f"{'wall s':>8}{'dt/limit':>9}  failed checks")
    for c in candidates:
        failed = ", ".join(k for k, ok in c["checks"].items() if not ok) or "-"
        print(f"  {c['method']:>6}{c['dt'] * 1e3:9.4f}{_fmt(c['e_score']):>10}  "
              f"{c['worst_case'] + ' / ' + c['worst_output']:<42}{c['rhs_calls']:11d}{c['wall_time']:8.1f}"
              f"{_fmt(c['dt_over_stability_limit'], '.2f'):>9}  {failed}")


def main() -> None:
    """Screen every method/dt on the convergence study cases, confirm the cheapest passing one on the rest of the
    required cases, verify it on the complete finite grid, and save to results/data/."""
    convergence = load_study_output(CONVERGENCE_PATH, "validation.convergence")
    unchecked = [name for name, ref in convergence["reference"]["cases"].items() if not ref["refinement"]["accepted"]]
    unchecked += [f"BE check {name}" for name, check in convergence["be_algebraic_check"].items() if not check["accepted"]]
    if unchecked:
        raise RuntimeError(f"{CONVERGENCE_PATH} has failed reference or BE checks: {unchecked}")
    study = convergence["rows"]
    dt_limits = stability_dt_limits()
    for method, lim in dt_limits.items():
        print(f"stability: {method.upper()} dt_max = {_fmt(lim['dt_max'] * 1e3, '.4g')} ms ({lim['case']})")

    screening = select_timestep(study, dt_limits=dt_limits)
    print(f"\nScreening on {', '.join(sorted(STUDY_CASES))}, cheapest first:")
    _print_candidates(screening["candidates"])

    remaining = {k: v for k, v in required_cases().items() if k not in STUDY_CASES}
    print(f"\nRK4 references for the other required cases: {', '.join(remaining)}")
    references = build_references(remaining)
    print_reference_checks(references)

    final, attempts = None, []
    for cand in (c for c in screening["candidates"] if c["passed"]):
        method, n_cut = cand["method"], cand["n_cut"]
        print(f"\nCandidate {method.upper()} dt = {cand['dt'] * 1e3:g} ms on the other required cases")
        extra = convergence_study(remaining, [method], {method: [n_cut]}, references)
        rows = [r for r in study if (r["method"], r["n_cut"]) == (method, n_cut)] + extra
        result = select_timestep(rows, dt_limits=dt_limits)
        attempts.append({"method": method, "n_cut": n_cut, "candidate": result["candidates"][0], "rows": extra})
        _print_candidates(result["candidates"])
        if result["selected"] is not None:
            final = result["selected"]
            break
    if final is None:
        raise RuntimeError("no screened method/dt passes every required case")
    ev = final["evidence"]
    print(f"\nSelected {final['method'].upper()} dt = {final['dt'] * 1e3:g} ms: E_score = {final['e_score']:.3g} "
          f"over {len(final['cases'])} required cases; worst {ev['case']} / {final['worst_output']} "
          f"(error {ev['errors'][final['worst_output']]:.4g}, tol {TOLERANCES[final['worst_output']]:g})")

    grid = verification_cases()
    required_rows = [r for r in study if (r["method"], r["n_cut"]) == (final["method"], final["n_cut"])] \
        + attempts[-1]["rows"]
    print(f"\nVerifying on the complete finite case set ({len(grid)} cases), each against its own RK4 reference")
    verification = verify_finite_grid(final["method"], final["n_cut"], grid, required_rows, dt_limits)
    failed = ", ".join(k for k, ok in verification["checks"].items() if not ok) or "none"
    print(f"  E_score over {len(verification['cases'])} cases = {_fmt(verification['e_score'])} "
          f"({verification['worst_case']} / {verification['worst_output']}); largest new-reference change on "
          f"refinement {_fmt(verification['max_reference_change_ratio'])} tol; failed checks: {failed}: "
          f"{'passed' if verification['passed'] else 'FAILED'}")

    report = {
        "study_baseline": STUDY_BASELINE,
        "tolerances": TOLERANCES,
        "e_score_max": E_SCORE_MAX,
        "helium_drift_limit": HELIUM_DRIFT_LIMIT,
        "stability": {"dt_factor": STABILITY_DT_FACTOR, "dt_limits": dt_limits},
        "required_cases": required_cases(),
        "screening": {"cases": sorted(STUDY_CASES), "candidates": screening["candidates"]},
        "references": reference_summary(references),
        "attempts": attempts,
        "selected": final,
        "verification": verification,
    }
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(json_safe(report), fh, indent=1)
    print(f"\nwrote {OUTPUT_PATH}")
    if not verification["passed"]:
        raise SystemExit("the selected method/dt failed the full-grid verification")


if __name__ == "__main__":
    main()
