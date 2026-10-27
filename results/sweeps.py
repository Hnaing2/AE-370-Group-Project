"""Final parameter studies with the selected method and dt.

Run: python -m results.sweeps
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Mapping, Sequence

import numpy as np

from src.parameters import STUDY_BASELINE, build_params
from src.physics import IA, IMP
from validation.convergence import (
    TOLERANCES, json_safe, load_study_output, powered_pressure_thrust, run_case, run_jobs,
)

KAPPA_VALUES = (5.0, 10.0, 20.0, 50.0, 100.0)
TAU_V_VALUES = (0.005, 0.01, 0.02, 0.05, 0.1)   # s
CD_VALUES = (0.4, 0.6, 0.8)
GAMMA0_VALUES = (0.93, 1.10, 1.27)               # rad

DATA_DIR = os.path.join("results", "data")
SELECTION_PATH = os.path.join(DATA_DIR, "timestep_selection.json")
OUTPUT_PATH = os.path.join(DATA_DIR, "sweeps.json")
HISTORY_PATH = os.path.join(DATA_DIR, "powered_histories.csv")
TRANSIENT_PATH = os.path.join(DATA_DIR, "transient_histories.csv")
# Start-up transient window and gains for the pressure-transient figure: every tau_v at these kappa values.
TRANSIENT_T_END = 0.5            # s
TRANSIENT_KAPPAS = (20.0, 100.0)
# Changing Cd or gamma0 must leave the pressure subsystem untouched; allow 0.1% of the pressure
# tolerance so a BE selection, whose Newton solve couples all states, can pass at round-off.
COUPLING_TOL = 1e-3 * TOLERANCES["pressure_history"]  # Pa
CSV_METRICS = ("status", "Pu_min", "Pu_droop", "Pu_overshoot", "Pu_initial_overpressure", "T_ax_peak", "T_ax_mean",
               "I_ax", "mp_used", "mp_cut", "ms_cut", "mu_cut", "a_cut", "Pu_cut", "Ps_cut", "V_cut", "h_cut",
               "x_eval", "h_eval", "h_eval_gain", "apogee_h", "apogee_t", "apogee_x", "peak_specific_force_g",
               "peak_mach", "helium_drift", "engine_validity_extrapolation", "subcritical_regulator_flow")
COMPARISON_METRICS = ("Pu_min", "Pu_droop", "Pu_cut", "T_ax_peak", "T_ax_mean", "I_ax", "mp_used", "mp_cut",
                      "ms_cut", "mu_cut", "a_cut", "V_cut", "h_cut", "x_eval", "h_eval", "apogee_h", "apogee_t",
                      "apogee_x", "peak_specific_force_g", "peak_mach")


def _case(name: str, group: str, base: Mapping[str, Any], changes: Mapping[str, Any]) -> dict[str, Any]:
    return {"name": name, "group": group, "changes": dict(changes), "inputs": {**base, **changes}}


def regulator_grid_cases(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    """5 x 5 tau_v x kappa grid at nominal Cd and gamma0."""
    return [_case(f"tau{round(tau * 1e3, 6):g}_kappa{kappa:g}", "regulator", base, {"tau_v": tau, "kappa": kappa})
            for tau in TAU_V_VALUES for kappa in KAPPA_VALUES]


def cd_gamma0_grid_cases(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    """3 x 3 Cd x gamma0 grid at nominal regulator settings (research question 3)."""
    return [_case(f"cd{cd:g}_gamma{gamma0:g}", "cd_gamma0", base, {"Cd": cd, "gamma0": gamma0})
            for cd in CD_VALUES for gamma0 in GAMMA0_VALUES]


def fixed_orifice_case(base: Mapping[str, Any]) -> dict[str, Any]:
    """Same hardware and helium with the valve held at a_fix."""
    return _case("fixed_orifice", "fixed_orifice", base, {"scenario": "fixed_orifice"})


def parameter_sweep(cases: Sequence[Mapping[str, Any]], method: str, dt: float) -> list[dict[str, Any]]:
    """One metrics row per case; save a parameter snapshot for each case.

    Each row keeps the full input set it ran with, its status, metrics, cost, and helium drift; the powered
    segment stays under "powered" for history checks and is not saved.
    """
    n_cuts = [round(case["inputs"]["t_cut"] / dt) for case in cases]
    jobs = [(case["inputs"], method, n) for case, n in zip(cases, n_cuts)]
    runs = run_jobs(run_case, jobs, [case["name"] for case in cases], weights=n_cuts)
    return [{"name": case["name"], "group": case["group"], "changes": case["changes"], "inputs": case["inputs"],
             "method": method, "n_cut": run["n_cut"], "dt": run["dt"], "status": run["status"],
             "metrics": run["metrics"], "cost": run["cost"], "helium_drift": run["helium_drift"],
             "failure": run["failure"], "powered": run["powered"]}
            for case, run in zip(cases, runs)]


def pressure_coupling_check(rows: Sequence[Mapping[str, Any]], nominal_name: str) -> dict[str, Any]:
    """Largest difference of each Cd x gamma0 run's powered mp, ms, mu, a, and Pu histories from the nominal run.
    The pressure subsystem does not depend on h, V, or gamma, so they must agree."""
    nominal = next(row for row in rows if row["name"] == nominal_name)
    U0 = nominal["powered"].U
    P0, _ = powered_pressure_thrust(U0, build_params(nominal["inputs"]))
    out = {}
    for row in rows:
        U = row["powered"].U
        P, _ = powered_pressure_thrust(U, build_params(row["inputs"]))
        same_shape = U.shape == U0.shape
        out[row["name"]] = {
            "max_Pu_difference": float(np.max(np.abs(P - P0))) if same_shape else None,
            "bitwise_identical": bool(same_shape and np.array_equal(U[:, IMP:IA + 1], U0[:, IMP:IA + 1])),
        }
    worst = max((v["max_Pu_difference"] for v in out.values()), key=lambda d: np.inf if d is None else d)
    return {"tolerance_Pa": COUPLING_TOL, "max_Pu_difference": worst,
            "passed": worst is not None and worst <= COUPLING_TOL, "cases": out}


def _write_csv(path: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _grid_csv(path: str, rows: Sequence[Mapping[str, Any]], inputs: Sequence[str]) -> None:
    _write_csv(path, ("name",) + tuple(inputs) + CSV_METRICS,
               [[row["name"]] + [row["inputs"][k] for k in inputs]
                + [row["status"] if k == "status" else row["metrics"].get(k, row.get(k)) for k in CSV_METRICS]
                for row in rows])


def load_selection(path: str = SELECTION_PATH) -> dict[str, Any]:
    """The selected method/dt, refused unless it was chosen for the current STUDY_BASELINE and passed the
    full-grid verification."""
    report = load_study_output(path, "validation.timestep_selection")
    if report["selected"] is None or not report["verification"]["passed"]:
        raise RuntimeError(f"{path} holds no selection that passed the full-grid verification")
    return report["selected"]


def main() -> None:
    """Regulator grid, Cd x gamma0 grid (check identical pressure histories), fixed-orifice case;
    write CSV/JSON to results/data/."""
    selected = load_selection()
    method, dt = selected["method"], selected["dt"]
    regulator = regulator_grid_cases(STUDY_BASELINE)
    cd_gamma0 = cd_gamma0_grid_cases(STUDY_BASELINE)
    fixed = fixed_orifice_case(STUDY_BASELINE)
    print(f"Sweeps with {method.upper()} at dt = {dt * 1e3:g} ms: {len(regulator)} regulator, "
          f"{len(cd_gamma0)} Cd x gamma0, 1 fixed orifice")
    rows = parameter_sweep(regulator + cd_gamma0 + [fixed], method, dt)
    reg_rows, cd_rows, fixed_row = rows[:len(regulator)], rows[len(regulator):-1], rows[-1]
    bad = [row["name"] for row in rows if row["status"] != "completed"]
    if bad:
        raise RuntimeError(f"sweep runs did not complete: {bad}")

    nominal_cd = f"cd{STUDY_BASELINE['Cd']:g}_gamma{STUDY_BASELINE['gamma0']:g}"
    coupling = pressure_coupling_check(cd_rows, nominal_cd)
    print(f"Cd x gamma0 pressure coupling: max |dPu| = {coupling['max_Pu_difference']:.3g} Pa "
          f"({'identical' if all(c['bitwise_identical'] for c in coupling['cases'].values()) else 'not bitwise identical'}), "
          f"{'ok' if coupling['passed'] else 'FAILED'}")

    nominal_reg = f"tau{round(STUDY_BASELINE['tau_v'] * 1e3, 6):g}_kappa{STUDY_BASELINE['kappa']:g}"
    regulated = next(row for row in reg_rows if row["name"] == nominal_reg)
    comparison = {k: {"regulated": regulated["metrics"][k], "fixed_orifice": fixed_row["metrics"][k],
                      "difference": fixed_row["metrics"][k] - regulated["metrics"][k]} for k in COMPARISON_METRICS}

    print(f"\n{'tau_v ms':>8}{'kappa':>7}{'Pu_min MPa':>12}{'droop %':>9}{'I_ax MN s':>11}{'x(t_eval) m':>13}"
          f"{'apogee m':>10}{'apogee_x m':>12}")
    for row in reg_rows:
        m = row["metrics"]
        print(f"{row['inputs']['tau_v'] * 1e3:8g}{row['inputs']['kappa']:7g}{m['Pu_min'] / 1e6:12.4f}"
              f"{m['Pu_droop'] * 100:9.2f}{m['I_ax'] / 1e6:11.4f}{m['x_eval']:13.1f}{m['apogee_h']:10.1f}{m['apogee_x']:12.1f}")
    print(f"\n{'Cd':>5}{'gamma0':>8}{'x(t_eval) m':>13}{'apogee m':>10}{'apogee_x m':>12}{'t_apo s':>9}")
    for row in cd_rows:
        m = row["metrics"]
        print(f"{row['inputs']['Cd']:5g}{row['inputs']['gamma0']:8g}{m['x_eval']:13.1f}{m['apogee_h']:10.1f}"
              f"{m['apogee_x']:12.1f}{m['apogee_t']:9.2f}")
    print(f"\n{'metric':<24}{'regulated':>14}{'fixed orifice':>16}")
    for k, v in comparison.items():
        print(f"{k:<24}{v['regulated']:14.6g}{v['fixed_orifice']:16.6g}")

    def strip(row):
        return {k: v for k, v in row.items() if k != "powered"}

    report = {
        "study_baseline": STUDY_BASELINE,
        "method": method, "dt": dt, "n_cut": selected["n_cut"], "selection": SELECTION_PATH,
        "grids": {"tau_v": TAU_V_VALUES, "kappa": KAPPA_VALUES, "Cd": CD_VALUES, "gamma0": GAMMA0_VALUES},
        "regulator_grid": [strip(row) for row in reg_rows],
        "cd_gamma0_grid": [strip(row) for row in cd_rows],
        "fixed_orifice": strip(fixed_row),
        "coupling_check": coupling,
        "regulated_vs_fixed_orifice": comparison,
    }
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(json_safe(report), fh, indent=1)
    _grid_csv(os.path.join(DATA_DIR, "regulator_grid.csv"), reg_rows, ("tau_v", "kappa"))
    _grid_csv(os.path.join(DATA_DIR, "cd_gamma0_grid.csv"), cd_rows, ("Cd", "gamma0"))
    _write_csv(os.path.join(DATA_DIR, "regulated_vs_fixed_orifice.csv"),
               ("metric", "regulated", "fixed_orifice", "difference"),
               [[k, v["regulated"], v["fixed_orifice"], v["difference"]] for k, v in comparison.items()])

    # Powered histories of the nominal regulated and fixed-orifice runs, same step, for Figure 3.
    columns, header = [regulated["powered"].t], ["t"]
    for tag, row in (("regulated", regulated), ("fixed_orifice", fixed_row)):
        Pu, T_ax = powered_pressure_thrust(row["powered"].U, build_params(row["inputs"]))
        columns += [Pu, T_ax, row["powered"].U[:, IA], row["powered"].U[:, IMP]]
        header += [f"Pu_{tag}", f"T_ax_{tag}", f"a_{tag}", f"mp_{tag}"]
    _write_csv(HISTORY_PATH, header, np.column_stack(columns).tolist())

    # Start-up transients (Pu and a for t <= TRANSIENT_T_END) at every tau_v for the TRANSIENT_KAPPAS gains,
    # where the valve lag shows up: first-crossing dip and any later overshoot (pressure-transient figure).
    window = regulated["powered"].t <= TRANSIENT_T_END + 1e-12
    columns, header = [regulated["powered"].t[window]], ["t"]
    for row in reg_rows:
        if row["inputs"]["kappa"] in TRANSIENT_KAPPAS:
            Pu, _ = powered_pressure_thrust(row["powered"].U[window], build_params(row["inputs"]))
            columns += [Pu, row["powered"].U[window, IA]]
            header += [f"Pu_{row['name']}", f"a_{row['name']}"]
    _write_csv(TRANSIENT_PATH, header, np.column_stack(columns).tolist())
    print(f"\nwrote {OUTPUT_PATH}, regulator_grid.csv, cd_gamma0_grid.csv, regulated_vs_fixed_orifice.csv, "
          f"{os.path.basename(HISTORY_PATH)}, {os.path.basename(TRANSIENT_PATH)}")
    if not coupling["passed"]:
        raise SystemExit("Cd x gamma0 runs changed the pressure history")


if __name__ == "__main__":
    main()
