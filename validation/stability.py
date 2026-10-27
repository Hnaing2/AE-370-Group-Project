"""Frozen-Jacobian stability characterization along a pilot trajectory.

Run: python -m validation.stability
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping

import numpy as np

from src.integrators import is_absolutely_stable
from src.parameters import STUDY_BASELINE, build_params
from src.physics import (
    IA, IMP, IMU,
    analytic_jacobian_choked,
    derived_quantities,
    jacobian,
    state_scales,
)
from src.simulation import derived_history, simulate

METHODS = ("fe", "rk4", "ab3", "be")
# 1/s; |lambda| at or below this is a neutral direction. Exact zeros (x column, helium sum, frozen coast
# states) come out below ~1e-13; the slowest physical mode, supply depletion just after the valve
# opens, is ~1e-6.
NEUTRAL_TOL = 1e-10
PILOT_DT = 1.25e-3  # s; RK4 pilot step, well inside every case's stability limit
BURN_SAMPLE_SPACING = 0.5  # s; uniform powered samples added to the event-based samples
Z_SCAN_STEP = 5e-3  # |z| resolution of the scan that brackets the first instability along a ray
Z_SCAN_MAX = 4.0    # beyond every explicit method's stability region (RK4 reaches |z| ~ 2.83)
FD_REL_STEP = 1e-6  # the physics.jacobian default

# Pilot cases: the nominal case, the four regulator corners, and the fixed orifice.
CASES: Mapping[str, Mapping[str, Any]] = {
    "nominal": {},
    "tau5_kappa5": {"tau_v": 0.005, "kappa": 5.0},
    "tau5_kappa100": {"tau_v": 0.005, "kappa": 100.0},
    "tau100_kappa5": {"tau_v": 0.1, "kappa": 5.0},
    "tau100_kappa100": {"tau_v": 0.1, "kappa": 100.0},
    "fixed_orifice": {"scenario": "fixed_orifice"},
}
OUTPUT_PATH = os.path.join("results", "data", "stability.json")


def command_branch(u: np.ndarray, params: Any) -> str:
    """Regulator command branch at a powered state: closed (clipped at 0), modulating, open (clipped at 1),
    or fixed for the fixed orifice. The bounds count as saturated, as in analytic_jacobian_choked."""
    if params.scenario != "regulated":
        return "fixed"
    c = params.K_p * (params.Pset - derived_quantities(u, params, "powered")["Pu"])
    return "closed" if c <= 0.0 else ("open" if c >= 1.0 else "modulating")


def fd_crosses_command_corner(u: np.ndarray, params: Any, rel_step: float = FD_REL_STEP) -> bool:
    """True if a central-difference perturbation of mp or mu (the states that set Pu) changes the
    command branch, i.e. the finite-difference Jacobian would straddle a corner."""
    branch = command_branch(u, params)
    steps = rel_step * np.maximum(np.abs(u), state_scales(params))
    for j in (IMP, IMU):
        for sign in (1.0, -1.0):
            v = u.copy()
            v[j] += sign * steps[j]
            if command_branch(v, params) != branch:
                return True
    return False


def pilot_sample_times(result: Any) -> list[tuple[float, str, str]]:
    """(t, phase, label) samples: initialization, first Pset crossing, regulator transitions, late burn,
    just before cutoff, and coast; plus powered states every BURN_SAMPLE_SPACING so the smallest step
    limit is not missed between the listed events.

    Samples are accepted states, so no interpolation enters the Jacobian. A sample whose finite-difference
    stencil would straddle a command corner moves to the next powered state.
    """
    params = result.params
    powered, coast = result.powered, result.coast
    Pu = derived_history(result)["Pu"][: powered.t.size]
    n_cut = powered.t.size - 1

    picks: list[tuple[int, str]] = [(0, "initialization")]
    below = np.flatnonzero(Pu < params.Pset)
    if below.size:
        picks.append((int(below[0]), "first Pset crossing"))
    branches = [command_branch(u, params) for u in powered.U]
    for k in range(1, n_cut + 1):
        if branches[k] != branches[k - 1] and not (below.size and k == below[0]):
            picks.append((k, f"command {branches[k - 1]} -> {branches[k]}"))
    choked = [derived_quantities(u, params, "powered")["choked"] for u in powered.U]
    for k in range(1, n_cut + 1):
        if choked[k] != choked[k - 1]:
            picks.append((k, "unchoked" if choked[k - 1] else "choked again"))
    picks += [(round(0.8 * n_cut), "late burn"), (n_cut - 1, "just before cutoff")]
    every = max(1, round(BURN_SAMPLE_SPACING / (powered.t[1] - powered.t[0])))
    picks += [(k, "burn") for k in range(every, n_cut - 1, every)]

    samples = []
    for k, label in picks:
        while k < n_cut and fd_crosses_command_corner(powered.U[k], params):
            k += 1
        samples.append((float(powered.t[k]), "powered", label))
    if coast is not None:
        samples.append((float(coast.t[0]), "coast", "coast start"))
        apogee = result.events.get("apogee")
        if apogee is not None:
            samples.append((float(coast.t[apogee["step"]]), "coast", "near apogee"))

    unique: dict[tuple[float, str], str] = {}
    for t, phase, label in samples:
        unique.setdefault((t, phase), label)
    return sorted(((t, phase, label) for (t, phase), label in unique.items()),
                  key=lambda s: (s[0], s[1] == "coast"))


def max_stable_dt(method: str, lam: complex) -> float:
    """Largest dt such that z = lam dt' is absolutely stable for every 0 < dt' <= dt.

    FE uses the closed form -2 Re(lam)/|lam|^2; BE is A-stable (inf). RK4 and AB3 scan |z| along the
    ray of lam until the first unstable point, then bisect at the actual complex z. lam must have
    Re(lam) < 0.
    """
    lam = complex(lam)
    if not lam.real < 0.0:
        raise ValueError(f"stability limit needs Re(lambda) < 0, got {lam}")
    if method == "be":
        return math.inf
    r = abs(lam)
    if method == "fe":
        return -2.0 * lam.real / (r * r)
    direction = lam / r
    s_lo = 0.0
    for s in np.arange(Z_SCAN_STEP, Z_SCAN_MAX + Z_SCAN_STEP / 2, Z_SCAN_STEP):
        if not is_absolutely_stable(method, s * direction):
            s_hi = float(s)
            break
        s_lo = float(s)
    else:
        raise RuntimeError(f"{method} still stable at |z| = {Z_SCAN_MAX} along {direction}")
    for _ in range(60):
        mid = 0.5 * (s_lo + s_hi)
        if is_absolutely_stable(method, mid * direction):
            s_lo = mid
        else:
            s_hi = mid
    return s_lo / r


def classify_eigenvalues(eigs: np.ndarray, neutral_tol: float = NEUTRAL_TOL) -> dict[str, np.ndarray]:
    """Split eigenvalues into neutral (|lambda| <= tol), stable (Re < 0), and unstable (Re >= 0) sets."""
    neutral = np.abs(eigs) <= neutral_tol
    stable = ~neutral & (eigs.real < 0.0)
    return {"neutral": eigs[neutral], "stable": eigs[stable], "unstable": eigs[~neutral & ~stable]}


def _complex_list(z: np.ndarray) -> list[list[float]]:
    return [[float(v.real), float(v.imag)] for v in z]


def characterize(result: Any, params: Any) -> list[dict[str, Any]]:
    """Per sample: eigenvalues, neutral/unstable flags, and the largest stable dt for each method,
    checked at the actual complex z = lambda dt (AB3 via all characteristic roots).

    Eigenvalues are those of the scaled matrix S^-1 J S (the same spectrum, better balanced), from the
    production central-difference Jacobian. On powered, choked states the analytic matrix is
    compared too. Frozen eigenvalues are diagnostics, not a nonlinear stability proof.
    """
    S = state_scales(params)
    rows = []
    for t, phase, label in pilot_sample_times(result):
        u = result.powered.U[np.searchsorted(result.powered.t, t)] if phase == "powered" else \
            result.coast.U[np.searchsorted(result.coast.t, t)]
        q = derived_quantities(u, params, phase)
        J = jacobian(t, u, params, phase)
        Js = J * S[np.newaxis, :] / S[:, np.newaxis]
        eigs = np.linalg.eigvals(Js)
        eigs = eigs[np.argsort(-np.abs(eigs))]
        sets = classify_eigenvalues(eigs)

        dt_max, limiting = {}, {}
        for method in METHODS:
            limits = [(max_stable_dt(method, lam), lam) for lam in sets["stable"]]
            dt, lam = min(limits, key=lambda x: x[0]) if limits else (math.inf, None)
            dt_max[method] = dt
            limiting[method] = None if lam is None or math.isinf(dt) else [float(lam.real), float(lam.imag)]

        jac_check = None
        if phase == "powered" and q["choked"]:
            Ja = analytic_jacobian_choked(t, u, params) * S[np.newaxis, :] / S[:, np.newaxis]
            jac_check = float(np.max(np.abs(Ja - Js)) / np.max(np.abs(Ja)))

        fastest = sets["stable"][0] if sets["stable"].size else None
        least_damped = min(sets["stable"], key=lambda lam: -lam.real / abs(lam)) if sets["stable"].size else None
        rows.append({
            "t": t, "phase": phase, "label": label,
            "Pu": q["Pu"], "a": float(u[IA]), "a_cmd": q["a_cmd"], "r": q["r"], "choked": bool(q["choked"]),
            "command": command_branch(u, params) if phase == "powered" else "held",
            "eigenvalues": _complex_list(eigs),
            "stable": _complex_list(sets["stable"]),
            "neutral_count": int(sets["neutral"].size),
            "unstable": _complex_list(sets["unstable"]),
            "max_stable_rate": None if fastest is None else float(abs(fastest)),
            "min_damping_ratio": None if least_damped is None else float(-least_damped.real / abs(least_damped)),
            "dt_max": dt_max,
            "limiting_eigenvalue": limiting,
            "analytic_jacobian_rel_diff": jac_check,
        })
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Smallest dt_max over the samples for each method, with the sample that sets it."""
    out = {}
    for method in METHODS:
        worst = min(rows, key=lambda row: row["dt_max"][method])
        out[method] = {"dt_max": worst["dt_max"][method], "t": worst["t"], "phase": worst["phase"],
                       "label": worst["label"], "eigenvalue": worst["limiting_eigenvalue"][method]}
    return out


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, float) and not math.isfinite(obj):
        return None  # unconditionally stable (BE), or nothing to limit the step
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def run_case(changes: Mapping[str, Any], dt: float = PILOT_DT) -> dict[str, Any]:
    params = build_params({**STUDY_BASELINE, **changes})
    result = simulate("rk4", params, {"dt": dt})
    if result.status != "completed":
        raise RuntimeError(f"pilot run ended with status {result.status}: {result.diagnostics}")
    rows = characterize(result, params)
    return {"changes": dict(changes), "pilot": {"method": "rk4", "dt": dt}, "samples": rows,
            "summary": summarize(rows)}


def _fmt_dt(dt: float) -> str:
    return "inf" if math.isinf(dt) else f"{dt * 1e3:.4g}"


def main() -> None:
    """Characterize the nominal case, the four regulator corners (including the fastest-valve/highest-gain
    one), and the fixed orifice; save to results/data/stability.json."""
    report = {
        "study_baseline": STUDY_BASELINE,
        "neutral_tol": NEUTRAL_TOL,
        "notes": "dt_max in s at |R| = 1 (no interior margin); null = unconditionally stable. Eigenvalues "
                 "in 1/s of the frozen central-difference Jacobian at accepted RK4 pilot states.",
        "cases": {},
    }
    for name, changes in CASES.items():
        case = run_case(changes)
        report["cases"][name] = case
        print(f"\n== {name} {dict(changes) or ''}")
        print(f"{'label':<30}{'t [s]':>8}{'|lam|max':>10}{'zeta_min':>9}{'unst':>5}{'neut':>5}"
              f"{'FE ms':>10}{'RK4 ms':>10}{'AB3 ms':>10}{'jac diff':>10}")
        for row in case["samples"]:
            zeta = row["min_damping_ratio"]
            jd = row["analytic_jacobian_rel_diff"]
            print(f"{row['label']:<30}{row['t']:8.3f}{row['max_stable_rate'] or 0.0:10.2f}"
                  f"{zeta if zeta is not None else math.nan:9.3f}{len(row['unstable']):5d}{row['neutral_count']:5d}"
                  f"{_fmt_dt(row['dt_max']['fe']):>10}{_fmt_dt(row['dt_max']['rk4']):>10}"
                  f"{_fmt_dt(row['dt_max']['ab3']):>10}{'' if jd is None else f'{jd:10.1e}'}")
        for method in ("fe", "rk4", "ab3"):
            s = case["summary"][method]
            print(f"  {method.upper():>3} dt_max = {s['dt_max'] * 1e3:.3f} ms at {s['label']} (t = {s['t']:.3f} s), "
                  f"lambda = {s['eigenvalue'][0]:.2f} {s['eigenvalue'][1]:+.2f}i")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(_json_safe(report), fh, indent=1)
    print(f"\nwrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
