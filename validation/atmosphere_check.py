"""Nominal atmosphere sensitivity check for the exponential density model (scale height H).

Compares exponential density (H = 7500 m) with the piecewise 1976 US Standard Atmosphere over 19-40 km,
and the nominal RK4 apogee under each (expected ~40.06 km vs ~40.45 km).

Run: python -m validation.atmosphere_check
"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from src.integrators import step_rk4
from src.parameters import STUDY_BASELINE, build_initial_state, build_params
from src.physics import IH, IV, IX, derived_quantities, rhs, standard_atmosphere_1976
from src.simulation import Segment, compute_metrics, find_apogee, simulate
from validation.convergence import json_safe, sample_segment

DT = 1.25e-3   # s; RK4 here is far inside every accuracy target (validation.convergence)
H_RANGE = (19000.0, 40000.0)  # m
OUTPUT_PATH = os.path.join("results", "data", "atmosphere_check.json")


def density_comparison(params: Any, n: int = 211) -> dict[str, Any]:
    """rho0 exp(-h/H) relative to the 1976 Standard Atmosphere density over H_RANGE."""
    h = np.linspace(*H_RANGE, n)
    rho_exp = params.rho0 * np.exp(-h / params.H)
    rho_76 = np.array([standard_atmosphere_1976(x)[0] for x in h])
    rel = rho_exp / rho_76 - 1.0
    every_km = np.isclose(h % 1000.0, 0.0)
    return {
        "max_rel_diff": float(rel.max()), "h_at_max": float(h[rel.argmax()]),
        "min_rel_diff": float(rel.min()), "h_at_min": float(h[rel.argmin()]),
        "table": [{"h": float(a), "rho_exp": float(b), "rho_1976": float(c), "rel_diff": float(d)}
                  for a, b, c, d in zip(h[every_km], rho_exp[every_km], rho_76[every_km], rel[every_km])],
    }


def rhs_us76(t: float, u: np.ndarray, params: Any, phase: str) -> np.ndarray:
    """The production RHS with drag from the 1976 Standard Atmosphere density instead of rho0 exp(-h/H)."""
    du = rhs(t, u, params, phase)
    q = derived_quantities(u, params, phase)
    rho_76 = standard_atmosphere_1976(float(u[IH]))[0]
    du[IV] -= (rho_76 - q["rho"]) * 0.5 * u[IV] ** 2 * params.Cd * params.A_ref / q["M"]
    return du


def flight_us76(params: Any, dt: float = DT) -> dict[str, float]:
    """RK4 with rhs_us76 through steps 1..N_cut powered, then coast to the step after apogee, as simulate does;
    apogee and x/h at t_eval from the coast Hermite interpolant."""
    n_cut = round(params.t_cut / dt)
    powered = lambda t, u: rhs_us76(t, u, params, "powered")
    coast = lambda t, u: rhs_us76(t, u, params, "coast")
    u = build_initial_state(params)
    for n in range(n_cut):
        u, _ = step_rk4(powered, n * dt, u, dt)
    n, t_acc, U_acc, F_acc = n_cut, [], [], []
    while n * dt <= params.t_end:
        t_acc.append(n * dt)
        U_acc.append(u)
        F_acc.append(coast(n * dt, u))
        if len(F_acc) > 1 and F_acc[-2][IH] > 0.0 >= F_acc[-1][IH]:
            break
        u, _ = step_rk4(coast, n * dt, u, dt, F_acc[-1])
        n += 1
    seg = Segment("coast", np.array(t_acc), np.array(U_acc), np.array(F_acc))
    apogee = find_apogee(seg)
    if apogee is None:
        raise RuntimeError("no apogee before t_end with the 1976 atmosphere")
    u_eval = sample_segment(seg, np.array([params.t_eval]))[0]
    return {"apogee_h": apogee["u"][IH], "apogee_t": apogee["t"], "apogee_x": apogee["u"][IX],
            "x_eval": u_eval[IX], "h_eval": u_eval[IH]}


def main() -> None:
    params = build_params(STUDY_BASELINE)
    density = density_comparison(params)
    print(f"exponential vs 1976 density over {H_RANGE[0] / 1e3:g}-{H_RANGE[1] / 1e3:g} km: "
          f"{density['min_rel_diff'] * 100:+.1f}% (h = {density['h_at_min'] / 1e3:.1f} km) to "
          f"{density['max_rel_diff'] * 100:+.1f}% (h = {density['h_at_max'] / 1e3:.1f} km)")

    m = compute_metrics(simulate("rk4", params, {"dt": DT}))
    exponential = {k: m[k] for k in ("apogee_h", "apogee_t", "apogee_x", "x_eval", "h_eval")}
    us76 = flight_us76(params)
    print(f"{'output':<10}{'exponential':>14}{'1976':>14}{'difference':>12}")
    for k in exponential:
        print(f"{k:<10}{exponential[k]:14.3f}{us76[k]:14.3f}{us76[k] - exponential[k]:12.3f}")

    report = {"method": "rk4", "dt": DT, "density": density, "exponential": exponential, "us1976": us76,
              "difference": {k: us76[k] - exponential[k] for k in exponential}}
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(json_safe(report), fh, indent=1)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
