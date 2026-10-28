"""Inputs, derived quantities, validation, and initial state."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .physics import A_BOUND_TOL, critical_pressure_ratio, flow_function

# Baseline inputs (flattened). Never edit REFERENCE_CASE.
_BASELINE_INPUTS: dict[str, Any] = {
    # mission conditions (approximate IFA) and assumed mass
    "h0": 19000.0,            # m
    "V0": 536.0,              # m/s
    "gamma0": 1.10,           # rad, calibrated to ~40 km apogee
    "x0": 0.0,                # m
    "t_cut": 7.5,             # s, scheduled cutoff
    "m_dry": 9900.0,          # kg, assumed benchmark incl. trunk
    "mp0": 2562.0,            # kg
    # engines
    "N_engines": 8,
    "force_engine_ref": 71000.0,  # N per engine, engine axis
    "Isp": 235.0,             # s, engine axis
    "delta": 0.349,           # rad, cant
    "Cf": 1.5,                # only to report implied At; not used in the RHS
    "P_cal": 8.0e6,           # Pa, = Pset
    # propellant
    "rho_o": 1441.4,          # kg/m^3, NTO 20 C
    "rho_f": 879.8,           # kg/m^3, MMH 20 C
    "mixture_ratio": 1.65,    # O/F mass ratio
    # pressurization
    "Pset": 8.0e6,            # Pa
    "Ps0": 35.0e6,            # Pa
    "Pu0": 8.28e6,            # Pa
    "Ts0": 293.15,            # K
    "Tu": 293.15,             # K
    "R_he": 2077.1,           # J/(kg K)
    "gamma_he": 1.667,
    "mu0": 1.64,              # kg
    "supply_mode": "sized",   # or "specified_mass" (then "ms0" must be given)
    "Ps_f": 10.0e6,           # Pa, end-of-burn sizing margin
    # regulator
    "a_ref": 0.8,             # reference opening for C_reg sizing (not a0)
    "kappa": 20.0,            # K_p = kappa / Pset
    "tau_v": 0.02,            # s
    "scenario": "regulated",  # or "fixed_orifice"
    # aero / environment
    "Cd": 0.6,
    "A_ref": 12.56,           # m^2
    "rho0": 1.225,            # kg/m^3
    "H": 7500.0,              # m
    "R_E": 6371000.0,         # m
    "g0": 9.80665,            # m/s^2
    # run control
    "t_eval": 15.0,           # s
    "t_end": 200.0,           # s cap; stop at the step after apogee
}

REFERENCE_CASE: Mapping[str, Any] = MappingProxyType(dict(_BASELINE_INPUTS))
STUDY_BASELINE: dict[str, Any] = dict(_BASELINE_INPUTS)  # record every change in the frozen input file
FIXED_ORIFICE_CASE: Mapping[str, Any] = MappingProxyType({**_BASELINE_INPUTS, "scenario": "fixed_orifice"})
GAS_STARVED_CASE: Mapping[str, Any] = MappingProxyType(
    {**_BASELINE_INPUTS, "supply_mode": "specified_mass", "ms0": 5.75}
)  # optional

# Provenance label for each independent input.
PROVENANCE: Mapping[str, str] = MappingProxyType({
    **{k: "sourced" for k in ("h0", "V0", "t_cut", "mp0", "N_engines", "force_engine_ref",
                              "A_ref", "rho_o", "rho_f", "R_he", "gamma_he", "g0")},
    "Isp": "nominal (secondary source)",
    "gamma0": "calibrated",
    "P_cal": "calibrated",
    **{k: "assumed" for k in ("m_dry", "delta", "Cf", "mixture_ratio", "Pset", "Ps0", "Ts0", "Tu",
                              "Pu0", "mu0", "Cd", "rho0", "H", "R_E", "x0")},
    **{k: "sized" for k in ("Ps_f", "a_ref")},
    **{k: "exploratory range" for k in ("kappa", "tau_v")},
    **{k: "study choice" for k in ("t_eval", "t_end", "scenario", "supply_mode", "ms0")},
})


@dataclass(frozen=True)
class Params:
    """Independent inputs plus derived quantities at full precision."""

    # independent inputs
    h0: float
    V0: float
    gamma0: float
    x0: float
    t_cut: float
    m_dry: float
    mp0: float
    N_engines: int
    force_engine_ref: float
    Isp: float
    delta: float
    Cf: float
    P_cal: float
    rho_o: float
    rho_f: float
    mixture_ratio: float
    Pset: float
    Ps0: float
    Pu0: float
    Ts0: float
    Tu: float
    R_he: float
    gamma_he: float
    mu0: float
    supply_mode: str
    Ps_f: float
    a_ref: float
    kappa: float
    tau_v: float
    scenario: str
    Cd: float
    A_ref: float
    rho0: float
    H: float
    R_E: float
    g0: float
    t_eval: float
    t_end: float
    # derived quantities, in derivation order
    rho_p: float
    Vu0: float
    Vtank: float
    K_F: float
    K_p: float
    q_p_ref: float
    ms0: float
    Vs: float
    q_r_req: float
    Ts_ref: float
    r_star: float
    C_reg: float
    a_fix: float
    a0: float
    M0: float
    # bookkeeping
    provenance: Mapping[str, str]


SCENARIOS = ("regulated", "fixed_orifice")
SUPPLY_MODES = ("sized", "specified_mass")

UNITS: Mapping[str, str] = MappingProxyType({
    "h0": "m", "V0": "m/s", "gamma0": "rad", "x0": "m", "t_cut": "s", "m_dry": "kg", "mp0": "kg",
    "N_engines": "-", "force_engine_ref": "N", "Isp": "s", "delta": "rad", "Cf": "-", "P_cal": "Pa",
    "rho_o": "kg/m^3", "rho_f": "kg/m^3", "mixture_ratio": "-", "Pset": "Pa", "Ps0": "Pa", "Pu0": "Pa",
    "Ts0": "K", "Tu": "K", "R_he": "J/(kg K)", "gamma_he": "-", "mu0": "kg", "supply_mode": "-",
    "Ps_f": "Pa", "a_ref": "-", "kappa": "-", "tau_v": "s", "scenario": "-", "Cd": "-", "A_ref": "m^2",
    "rho0": "kg/m^3", "H": "m", "R_E": "m", "g0": "m/s^2", "t_eval": "s", "t_end": "s",
    # derived
    "rho_p": "kg/m^3", "Vu0": "m^3", "Vtank": "m^3", "K_F": "N/Pa", "K_p": "1/Pa", "q_p_ref": "kg/s",
    "ms0": "kg", "Vs": "m^3", "q_r_req": "kg/s", "Ts_ref": "K", "r_star": "-",
    "C_reg": "kg sqrt(K)/(s Pa)", "a_fix": "-", "a0": "-", "M0": "kg",
})
REFERENCE_TEMPERATURE: Mapping[str, str] = MappingProxyType({"rho_o": "20 C", "rho_f": "20 C"})

_STRING_INPUTS = ("supply_mode", "scenario")
# Inputs that must be strictly positive; the rest of the numeric inputs need only be finite.
_POSITIVE_INPUTS = tuple(
    k for k in _BASELINE_INPUTS if k not in _STRING_INPUTS + ("x0", "gamma0", "delta")
)
_DERIVED_FIELDS = ("rho_p", "Vu0", "Vtank", "K_F", "K_p", "q_p_ref", "ms0", "Vs", "q_r_req",
                   "Ts_ref", "r_star", "C_reg", "a_fix", "a0", "M0")


def _check_inputs(inputs: Mapping[str, Any]) -> None:
    required = set(_BASELINE_INPUTS)
    mode = inputs.get("supply_mode")
    if mode == "specified_mass":
        required.add("ms0")
    missing = sorted(required - set(inputs))
    if missing:
        raise ValueError(f"missing required inputs: {missing}")
    unknown = sorted(set(inputs) - required)
    if unknown:
        raise ValueError(f"unknown inputs (ms0 is an input only in specified_mass mode): {unknown}")
    if mode not in SUPPLY_MODES:
        raise ValueError(f"supply_mode must be one of {SUPPLY_MODES}, got {mode!r}")
    if inputs["scenario"] not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}, got {inputs['scenario']!r}")
    for k in required - set(_STRING_INPUTS):
        v = inputs[k]
        if isinstance(v, bool) or not isinstance(v, (int, float, np.integer, np.floating)) or not np.isfinite(v):
            raise ValueError(f"input {k} must be a finite number, got {v!r}")
    for k in _POSITIVE_INPUTS + (("ms0",) if mode == "specified_mass" else ()):
        if inputs[k] <= 0:
            raise ValueError(f"input {k} must be positive, got {inputs[k]!r}")
    if inputs["N_engines"] != int(inputs["N_engines"]):
        raise ValueError(f"N_engines must be a whole number, got {inputs['N_engines']!r}")
    if not 0.0 <= inputs["delta"] < np.pi / 2:
        raise ValueError(f"cant delta must lie in [0, pi/2), got {inputs['delta']!r}")
    if inputs["gamma_he"] <= 1.0:
        raise ValueError("gamma_he must exceed 1")
    if not 0.0 < inputs["a_ref"] <= 1.0:
        raise ValueError("a_ref must lie in (0, 1]")
    if not inputs["Pset"] < inputs["Ps_f"] < inputs["Ps0"]:
        raise ValueError("sizing requires Pset < Ps_f < Ps0")
    if not inputs["t_cut"] < inputs["t_eval"] < inputs["t_end"]:
        raise ValueError("require t_cut < t_eval < t_end")


def build_params(inputs: Mapping[str, Any]) -> Params:
    """Derive every derived quantity from the independent inputs at full precision; raise on any missing input."""
    _check_inputs(inputs)
    p = {k: inputs[k] for k in _BASELINE_INPUTS}
    p["N_engines"] = int(p["N_engines"])
    for k in _BASELINE_INPUTS:
        if k not in _STRING_INPUTS and k != "N_engines":
            p[k] = float(p[k])

    R_T_u = p["R_he"] * p["Tu"]
    r = p["mixture_ratio"]
    rho_p = (1.0 + r) / (1.0 / p["rho_f"] + r / p["rho_o"])
    Vu0 = p["mu0"] * R_T_u / p["Pu0"]
    Vtank = p["mp0"] / rho_p + Vu0
    K_F = p["N_engines"] * p["force_engine_ref"] / p["P_cal"]
    K_p = p["kappa"] / p["Pset"]
    q_p_ref = K_F * p["Pset"] / (p["Isp"] * p["g0"])

    if p["supply_mode"] == "sized":
        mp_burn = q_p_ref * p["t_cut"]
        mu_needed = p["Pset"] * (Vu0 + mp_burn / rho_p) / R_T_u
        ms0 = (mu_needed - p["mu0"]) / (1.0 - (p["Ps_f"] / p["Ps0"]) ** (1.0 / p["gamma_he"]))
        if ms0 <= 0:
            raise ValueError(f"sized supply mass is nonpositive ({ms0}); check Pset, Pu0, mu0")
    else:
        ms0 = float(inputs["ms0"])
    Vs = ms0 * p["R_he"] * p["Ts0"] / p["Ps0"]

    q_r_req = p["Pset"] * q_p_ref / (rho_p * R_T_u)
    Ts_ref = p["Ts0"] * (p["Ps_f"] / p["Ps0"]) ** ((p["gamma_he"] - 1.0) / p["gamma_he"])
    r_star = critical_pressure_ratio(p["gamma_he"])
    r_ref = p["Pset"] / p["Ps_f"]  # sizing pressure ratio Pu,ref / Ps,ref; 0.8 at the baseline
    C_reg = q_r_req * np.sqrt(Ts_ref) / (p["a_ref"] * p["Ps_f"] * flow_function(r_ref, p["gamma_he"]))
    a_fix = q_r_req * np.sqrt(p["Ts0"]) / (C_reg * p["Ps0"] * flow_function(p["Pset"] / p["Ps0"], p["gamma_he"]))
    if not 0.0 < a_fix <= 1.0:
        raise ValueError(f"fixed-orifice opening a_fix = {a_fix} outside (0, 1]")
    a0 = 0.0 if p["scenario"] == "regulated" else a_fix
    M0 = p["m_dry"] + p["mp0"] + ms0 + p["mu0"]

    provenance = {k: PROVENANCE[k] for k in _BASELINE_INPUTS}
    provenance["ms0"] = "sized" if p["supply_mode"] == "sized" else PROVENANCE["ms0"]

    return Params(
        **p,
        rho_p=float(rho_p), Vu0=float(Vu0), Vtank=float(Vtank), K_F=float(K_F), K_p=float(K_p),
        q_p_ref=float(q_p_ref), ms0=float(ms0), Vs=float(Vs), q_r_req=float(q_r_req),
        Ts_ref=float(Ts_ref), r_star=float(r_star), C_reg=float(C_reg), a_fix=float(a_fix),
        a0=float(a0), M0=float(M0), provenance=MappingProxyType(provenance),
    )


def validate(params: Params, u0: np.ndarray, options: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validation report; raise ValueError on invalid inputs.

    Checks: finite values, Vu > 0, positive masses, 0 <= a0 <= 1, known scenario/supply_mode,
    and that t_cut / options["dt"] is an integer to round-off.
    """
    u0 = np.asarray(u0)
    if u0.shape != (8,) or u0.dtype != np.float64:
        raise ValueError(f"u0 must be float64 with shape (8,), got {u0.dtype} {u0.shape}")
    if not np.all(np.isfinite(u0)):
        raise ValueError(f"u0 has nonfinite entries: {u0}")
    if params.scenario not in SCENARIOS or params.supply_mode not in SUPPLY_MODES:
        raise ValueError("unknown scenario or supply_mode")

    _, h, V, _, mp, ms, mu, a = u0
    if h < 0:
        raise ValueError(f"initial altitude must be nonnegative, got {h}")
    if V <= 0:
        raise ValueError(f"initial speed must be positive, got {V}")
    for name, m in (("mp", mp), ("ms", ms), ("mu", mu)):
        if m <= 0:
            raise ValueError(f"initial {name} must be positive, got {m}")
    Vu = params.Vtank - mp / params.rho_p
    if Vu <= 0:
        raise ValueError(f"initial ullage volume must be positive, got {Vu}")
    if not -A_BOUND_TOL <= a <= 1.0 + A_BOUND_TOL:
        raise ValueError(f"initial valve opening outside [0, 1]: {a}")
    M = params.m_dry + mp + ms + mu
    if M <= 0:
        raise ValueError(f"initial mass must be positive, got {M}")

    if "dt" not in options:
        raise ValueError("options must contain dt")
    dt = float(options["dt"])
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be positive and finite, got {dt}")
    n_cut = round(params.t_cut / dt)
    if n_cut < 1 or abs(n_cut * dt - params.t_cut) > 64 * np.finfo(float).eps * params.t_cut:
        raise ValueError(f"t_cut / dt = {params.t_cut / dt!r} is not an integer")

    Pu = mu * params.R_he * params.Tu / Vu
    return {
        "ok": True,
        "n_cut": int(n_cut),
        "Vu0": float(Vu),
        "Pu0": float(Pu),
        "M0": float(M),
        "flags": {"Pu_below_half_Pset": bool(Pu < 0.5 * params.Pset)},
    }


def build_initial_state(params: Params) -> np.ndarray:
    """u0 = [x0, h0, V0, gamma0, mp0, ms0, mu0, a0], shape (8,)."""
    return np.array([params.x0, params.h0, params.V0, params.gamma0,
                     params.mp0, params.ms0, params.mu0, params.a0], dtype=np.float64)


def write_input_manifest(params: Params, path: str, extra: Mapping[str, Any] | None = None) -> None:
    """Write the frozen input file: value, SI unit, and provenance label per input,
    the derived quantities, and any extra top-level entries (e.g. the swept grids and their labels)."""
    import json

    independent = list(_BASELINE_INPUTS) + (["ms0"] if params.supply_mode == "specified_mass" else [])
    manifest = {
        "inputs": {
            k: {"value": getattr(params, k), "unit": UNITS[k], "label": params.provenance[k],
                **({"reference_temperature": REFERENCE_TEMPERATURE[k]} if k in REFERENCE_TEMPERATURE else {})}
            for k in independent
        },
        "derived": {
            k: {"value": getattr(params, k), "unit": UNITS[k]}
            for k in _DERIVED_FIELDS if k not in independent
        },
        **(extra or {}),
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
