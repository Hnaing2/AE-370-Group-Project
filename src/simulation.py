"""Powered/coast driver, apogee localization, interpolation, and metrics."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional

import numpy as np

from .integrators import step_ab3, step_be, step_fe, step_rk4
from .parameters import build_initial_state, validate
from .physics import (
    IA, IH, IMP, IMS, IMU, IV, IX,
    R_AIR_1976,
    PhysicalModelInvalid,
    derived_quantities,
    jacobian,
    rhs,
    standard_atmosphere_1976,
    state_scales,
)

METHODS = ("fe", "rk4", "ab3", "be")
STATUSES = ("completed", "no_apogee", "physical_model_invalid", "solver_failed")

DEFAULT_OPTIONS: dict[str, Any] = {
    "method": "rk4",
    "dt": 5.0e-4,             # s; t_cut / dt must be an integer
    "be_residual_tol": 1e-10,
    "be_solver_options": {},  # Newton overrides of NEWTON_DEFAULTS, or {"method": <scipy root method>, ...}
    "t_end": None,            # None -> params.t_end
}

GAMMA_AIR = 1.4  # post-processing only, for Mach number
_DERIVED_KEYS = ("M", "Vu", "Pu", "Ps", "Ts", "r", "rho", "g", "D", "F_eng", "T_ax", "qp", "qr",
                 "a_cmd", "choked", "Pu_below_half_Pset")


@dataclass
class Segment:
    phase: str                # "powered" or "coast"
    t: np.ndarray             # (n+1,), computed as n*dt with the global step index n
    U: np.ndarray             # (n+1, 8) accepted states
    F: np.ndarray             # (n+1, 8) RHS at each accepted state, with this segment's phase


@dataclass
class SimulationResult:
    powered: Segment
    coast: Optional[Segment]
    status: str
    events: dict[str, Any] = field(default_factory=dict)     # "apogee": {"t": ..., "u": ...}
    counters: dict[str, Any] = field(default_factory=dict)   # rhs_calls, jac_calls, nonlinear_iters, ab3_restarts, be_failures, wall_time
    diagnostics: dict[str, Any] = field(default_factory=dict)  # BE max residual, helium drift, validity flags
    params: Any = None
    options: Mapping[str, Any] = field(default_factory=dict)


class _SolverFailed(Exception):
    """A BE step was not accepted; the run ends with status solver_failed."""

    def __init__(self, message: str, residual: float):
        super().__init__(message)
        self.residual = residual


# advance(n, u, F_hist) -> (u_{n+1}, f(t_{n+1}, u_{n+1})), with F_hist the derivatives at this
# segment's accepted states so far (F_hist[-1] = f(t_n, u)).
Advance = Callable[[int, np.ndarray, list], tuple[np.ndarray, np.ndarray]]


def _counting(f: Callable, counters: dict[str, Any], key: str = "rhs_calls") -> Callable:
    """f that adds one to counters[key] before every call, so a call that raises (an invalid stage
    or iterate) is counted too."""
    def counted(*args):
        counters[key] += 1
        return f(*args)
    return counted


def _explicit_advance(method: str, f: Callable, dt: float) -> Advance:
    """FE, RK4, or AB3 on the counting f, reusing the stored f(t_n, u_n) as f0 / k1 / f_n.

    AB3 takes RK4 steps until this segment holds three derivatives, so it starts at t = 0 and
    restarts at t_cut with two RK4 steps on the coast-side history only.
    """
    def advance(n, u, F_hist):
        if method == "fe":
            u_next, _ = step_fe(f, n * dt, u, dt, F_hist[-1])
        elif method == "rk4" or len(F_hist) < 3:
            u_next, _ = step_rk4(f, n * dt, u, dt, F_hist[-1])
        else:
            u_next = step_ab3(u, (F_hist[-1], F_hist[-2], F_hist[-3]), dt)
        return u_next, f((n + 1) * dt, u_next)
    return advance


def _be_advance(f: Callable, params: Any, phase: str, dt: float, opts: Mapping[str, Any],
                counters: dict[str, Any], be_stats: dict[str, float]) -> Advance:
    """BE on the counting f with the phase's finite-difference Jacobian and state scales.

    Residuals and finite differences all call f, and Jacobian evaluations are counted by their own
    closure, so a step that fails part-way is counted too. The derivative step_be evaluated at the
    accepted state is stored as F. An iterate that leaves the physical domain is a solver failure,
    not an invalid accepted state; nothing is clipped.
    """
    jac = _counting(lambda t, u: jacobian(t, u, params, phase, f=f), counters, "jac_calls")
    scales = state_scales(params)

    def advance(n, u, F_hist):
        try:
            res = step_be(f, n * dt, u, dt, jac=jac, scales=scales,
                          residual_tol=opts["be_residual_tol"], solver_options=opts["be_solver_options"])
        except PhysicalModelInvalid as exc:
            counters["be_failures"] += 1
            raise _SolverFailed(f"nonlinear iterate left the physical domain: {exc}", math.nan) from exc
        if res.n_iter is not None:
            counters["nonlinear_iters"] = (counters["nonlinear_iters"] or 0) + res.n_iter
        if not res.success:
            counters["be_failures"] += 1
            raise _SolverFailed(res.message, res.residual)
        be_stats["max_residual"] = max(be_stats["max_residual"], res.residual)
        return res.u, res.f
    return advance


def _integrate_segment(
    advance: Advance, f: Callable, phase: str, n0: int, u0: np.ndarray, dt: float, n_max: int,
    stop_after_apogee: bool,
) -> tuple[Segment, Optional[dict[str, Any]]]:
    """Accept steps n0+1..n_max of one phase (fewer if stopping after apogee, on an invalid state,
    or on a rejected BE step).

    F[k] = f(t_k, U[k]) is stored with every accepted state and reused by the next step, so a
    state and its derivative are appended together. Returns (segment, error or None); the error
    carries the run status it implies.
    """
    t_acc, U_acc, F_acc = [], [], []
    error = None
    n, u = n0, u0
    try:
        fu = f(n * dt, u)
        while True:
            t_acc.append(n * dt)
            U_acc.append(u)
            F_acc.append(fu)
            if n >= n_max or (stop_after_apogee and len(F_acc) > 1 and F_acc[-2][IH] > 0.0 >= fu[IH]):
                break
            u, fu = advance(n, u, F_acc)
            n += 1
    except PhysicalModelInvalid as exc:
        error = {"status": "physical_model_invalid", "phase": phase,
                 "t_last_accepted": t_acc[-1] if t_acc else None, "message": str(exc)}
    except _SolverFailed as exc:
        error = {"status": "solver_failed", "phase": phase, "t_last_accepted": t_acc[-1],
                 "message": str(exc), "residual": exc.residual}
    seg = Segment(phase, np.array(t_acc), np.array(U_acc).reshape(-1, 8), np.array(F_acc).reshape(-1, 8))
    return seg, error


def _phase_rhs(params: Any, phase: str) -> Callable[[float, np.ndarray], np.ndarray]:
    return lambda t, u: rhs(t, u, params, phase)


def simulate(method: str, params: Any, options: Mapping[str, Any]) -> SimulationResult:
    """Integrate powered steps 1..N_cut, then coast until the step after apogee (or t_end).

    The cutoff state starts the coast segment; AB3 restarts there from two RK4 steps.
    Invalid options raise ValueError. A state leaving the physical domain ends the run with status
    physical_model_invalid, and a rejected BE step with solver_failed; either keeps the states
    accepted so far.
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    opts = {**DEFAULT_OPTIONS, **options, "method": method}
    u0 = build_initial_state(params)
    n_cut = validate(params, u0, opts)["n_cut"]
    dt = float(opts["dt"])
    t_end = params.t_end if opts["t_end"] is None else float(opts["t_end"])
    n_end = int(math.floor(t_end / dt + 1e-9))
    if n_end <= n_cut:
        raise ValueError(f"t_end = {t_end} must exceed t_cut = {params.t_cut}")

    # nonlinear_iters stays None for BE if the solver does not expose iterations (scipy hybr does not).
    counters = {"rhs_calls": 0, "jac_calls": 0, "nonlinear_iters": None if method == "be" else 0,
                "ab3_restarts": 0, "be_failures": 0, "wall_time": 0.0}
    be_stats = {"max_residual": 0.0}

    def make_advance(phase):
        f = _counting(_phase_rhs(params, phase), counters)  # every RHS call of the run goes through here
        if method == "be":
            return _be_advance(f, params, phase, dt, opts, counters, be_stats), f
        return _explicit_advance(method, f, dt), f

    start = time.perf_counter()
    powered, error = _integrate_segment(*make_advance("powered"), "powered", 0, u0, dt, n_cut, False)
    coast = None
    if error is None:
        if method == "ab3":
            counters["ab3_restarts"] += 1  # the coast segment starts a fresh derivative history
        coast, error = _integrate_segment(*make_advance("coast"), "coast", n_cut, powered.U[-1], dt, n_end, True)
    counters["wall_time"] = time.perf_counter() - start

    events: dict[str, Any] = {}
    if error is not None:
        status = error["status"]
    else:
        apogee = find_apogee(coast)
        status = "no_apogee" if apogee is None else "completed"
        if apogee is not None:
            events["apogee"] = apogee

    helium = np.concatenate([seg.U[:, IMS] + seg.U[:, IMU] for seg in (powered, coast) if seg is not None])
    helium0 = u0[IMS] + u0[IMU]
    diagnostics = {
        "dt": dt,
        "n_cut": n_cut,
        "t_cut": n_cut * dt,
        "helium_drift": float(np.max(np.abs(helium - helium0)) / helium0) if helium.size else math.nan,
        "invalid": error if status == "physical_model_invalid" else None,
        "solver_failure": error if status == "solver_failed" else None,
        "be_max_residual": be_stats["max_residual"] if method == "be" else math.nan,  # over accepted steps
    }
    return SimulationResult(powered, coast, status, events, counters, diagnostics, params, opts)


def hermite_interpolate(t0: float, t1: float, u0: np.ndarray, u1: np.ndarray,
                        f0: np.ndarray, f1: np.ndarray, t_query: float) -> np.ndarray:
    """Cubic Hermite interpolant within one step of one phase segment."""
    dt = t1 - t0
    s = (t_query - t0) / dt
    s2, s3 = s * s, s * s * s
    return ((2.0 * s3 - 3.0 * s2 + 1.0) * u0 + (s3 - 2.0 * s2 + s) * dt * f0
            + (3.0 * s2 - 2.0 * s3) * u1 + (s3 - s2) * dt * f1)


def _hermite_rate(dt: float, u0: float, u1: float, f0: float, f1: float, s: float) -> float:
    """Time derivative of one component of the cubic Hermite interpolant at fraction s of the step."""
    return (6.0 * s * s - 6.0 * s) * (u0 - u1) / dt + (3.0 * s * s - 4.0 * s + 1.0) * f0 + (3.0 * s * s - 2.0 * s) * f1


def find_apogee(segment: Segment) -> Optional[dict[str, Any]]:
    """Bracket the sign change of V sin(gamma) in the coast segment; bisect the Hermite interpolant.

    The interpolant's vertical rate equals V sin(gamma) at both step ends, so the bracket carries
    over to it. Returns {"t", "u", "step"} for the first positive-to-nonpositive change, or None.
    """
    vz = segment.F[:, IH]
    hits = np.flatnonzero((vz[:-1] > 0.0) & (vz[1:] <= 0.0))
    if hits.size == 0:
        return None
    k = int(hits[0])
    t0, t1 = float(segment.t[k]), float(segment.t[k + 1])
    u0, u1, f0, f1 = segment.U[k], segment.U[k + 1], segment.F[k], segment.F[k + 1]
    lo, hi = 0.0, 1.0  # vertical rate > 0 at lo, <= 0 at hi
    while True:
        mid = 0.5 * (lo + hi)
        if mid <= lo or mid >= hi:
            break
        if _hermite_rate(t1 - t0, u0[IH], u1[IH], f0[IH], f1[IH], mid) > 0.0:
            lo = mid
        else:
            hi = mid
    t_apo = t0 + mid * (t1 - t0)
    return {"t": t_apo, "u": hermite_interpolate(t0, t1, u0, u1, f0, f1, t_apo), "step": k}


def _sample_segment(seg: Segment, t_query: float) -> np.ndarray:
    t = seg.t
    if t.size == 0 or not t[0] <= t_query <= t[-1]:
        span = f"[{t[0]}, {t[-1]}]" if t.size else "empty"
        raise ValueError(f"t = {t_query} outside the {seg.phase} segment {span}")
    if t.size == 1:
        return seg.U[0].copy()
    k = min(max(int(np.searchsorted(t, t_query, side="right")) - 1, 0), t.size - 2)
    return hermite_interpolate(t[k], t[k + 1], seg.U[k], seg.U[k + 1], seg.F[k], seg.F[k + 1], t_query)


def sample_at_time(result: SimulationResult, t_query: float, phase: Optional[str] = None) -> np.ndarray:
    """State at t_query by Hermite interpolation inside one segment; never across t_cut.

    At t_query == t_cut, phase selects the powered or coast side; by default t <= t_cut samples
    the powered segment and t > t_cut the coast segment.
    """
    if phase is None:
        phase = "powered" if t_query <= result.diagnostics["t_cut"] else "coast"
    if phase == "powered":
        return _sample_segment(result.powered, t_query)
    if phase == "coast":
        if result.coast is None:
            raise ValueError("run has no coast segment")
        return _sample_segment(result.coast, t_query)
    raise ValueError(f"phase must be 'powered', 'coast', or None, got {phase!r}")


def _atmosphere_temperature(h: float) -> float:
    try:
        return standard_atmosphere_1976(h)[1]
    except ValueError:
        return math.nan


def derived_history(result: SimulationResult) -> dict[str, np.ndarray]:
    """Recompute Pu, T_ax, qr, r, Mach, specific force, ... from the stored U of each segment.

    Arrays run over the powered then the coast states, so the cutoff state appears twice; the
    boolean "powered" mask selects the powered-side values. Mach uses the 1976 US Standard
    Atmosphere temperature (post-processing only; NaN outside its table).
    """
    segments = [seg for seg in (result.powered, result.coast) if seg is not None]
    rows: dict[str, list] = {k: [] for k in _DERIVED_KEYS}
    T_atm = []
    for seg in segments:
        for u in seg.U:
            q = derived_quantities(u, result.params, seg.phase)
            for k in _DERIVED_KEYS:
                rows[k].append(q[k])
            T_atm.append(_atmosphere_temperature(float(u[IH])))
    out = {k: np.array(v, dtype=bool if k in ("choked", "Pu_below_half_Pset") else np.float64)
           for k, v in rows.items()}
    U = np.concatenate([seg.U for seg in segments])
    out["t"] = np.concatenate([seg.t for seg in segments])
    out["powered"] = np.concatenate([np.full(seg.t.size, seg.phase == "powered") for seg in segments])
    out["specific_force"] = (out["T_ax"] - out["D"]) / out["M"]
    out["T_atm"] = np.array(T_atm)
    out["mach"] = U[:, IV] / np.sqrt(GAMMA_AIR * R_AIR_1976 * out["T_atm"])
    return out


def _max(a: np.ndarray) -> float:
    return float(np.nanmax(a)) if a.size and not np.all(np.isnan(a)) else math.nan


def compute_metrics(result: SimulationResult) -> dict[str, Any]:
    """Pressure extrema/droop, thrust and impulse (mass-use identity), propellant/pressurant at cutoff,
    x/h at t_eval, apogee/time/downrange, peak axial specific force, peak Mach, helium drift, status.

    Extrema are taken over the accepted states; quantities a failed run never reached are NaN.
    """
    p = result.params
    nan = math.nan
    hist = derived_history(result)
    powered = hist["powered"]
    Pu = hist["Pu"][powered]
    reached_cut = result.coast is not None

    m: dict[str, Any] = {"status": result.status}

    # Pressure (powered states only). Overshoot is max Pu - Pset after the first Pset crossing, counting
    # the crossing instant itself (Pu = Pset), so it is 0 when Pu never rises back above Pset. Without
    # that, the first stored state below Pset would set it, a value that shrinks with dt.
    m["Pu_min"] = float(np.min(Pu)) if Pu.size else nan
    m["Pu_max"] = _max(Pu)
    m["Pu_droop"] = (p.Pset - m["Pu_min"]) / p.Pset
    m["Pu_initial_overpressure"] = (Pu[0] - p.Pset) / p.Pset if Pu.size else nan
    crossed = np.flatnonzero(Pu <= p.Pset)
    m["Pu_overshoot"] = max(float(np.max(Pu[crossed[0]:]) - p.Pset), 0.0) if crossed.size else nan

    # Thrust and impulse; impulse from the mass-use identity.
    cut = result.powered.U[-1] if reached_cut else np.full(8, nan)
    m["T_ax_peak"] = _max(hist["T_ax"][powered])
    m["I_ax"] = p.Isp * p.g0 * math.cos(p.delta) * (p.mp0 - cut[IMP])
    m["T_ax_mean"] = m["I_ax"] / p.t_cut

    # Cutoff state.
    m["V_cut"], m["h_cut"], m["x_cut"] = cut[IV], cut[IH], cut[IX]
    m["mp_cut"], m["ms_cut"], m["mu_cut"], m["a_cut"] = cut[IMP], cut[IMS], cut[IMU], cut[IA]
    m["mp_used"] = p.mp0 - cut[IMP]
    m["Pu_cut"] = Pu[-1] if reached_cut else nan
    m["Ps_cut"] = hist["Ps"][powered][-1] if reached_cut else nan

    # Flight outputs.
    try:
        u_eval = sample_at_time(result, p.t_eval)
        m["x_eval"], m["h_eval"] = u_eval[IX], u_eval[IH]
    except ValueError:
        m["x_eval"] = m["h_eval"] = nan
    m["h_eval_gain"] = m["h_eval"] - p.h0
    apogee = result.events.get("apogee")
    m["apogee_t"] = apogee["t"] if apogee else nan
    m["apogee_h"] = apogee["u"][IH] if apogee else nan
    m["apogee_x"] = apogee["u"][IX] if apogee else nan
    m["peak_specific_force"] = _max(hist["specific_force"])
    m["peak_specific_force_g"] = m["peak_specific_force"] / p.g0
    m["peak_mach"] = _max(hist["mach"])

    # Validity.
    m["helium_drift"] = result.diagnostics["helium_drift"]
    m["engine_validity_extrapolation"] = bool(np.any(hist["Pu_below_half_Pset"]))
    m["subcritical_regulator_flow"] = bool(np.any(powered & ~hist["choked"]))
    return {k: float(v) if isinstance(v, np.floating) else v for k, v in m.items()}
