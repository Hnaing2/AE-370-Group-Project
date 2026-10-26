"""Inputs, derived quantities, validation, and initial state.

Spec: docs/spec/model.md (Sections 3.4, 3.6, 3.7 and the derivation formulas).
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

# Section 3.6 baseline (Revision 4, flattened). Never edit REFERENCE_CASE.
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

# Section 3.7 provenance label for each independent input.
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

    # independent inputs (Section 3.6)
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
    # derived (Section 3.6 derived table, in derivation order)
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


def build_params(inputs: Mapping[str, Any]) -> Params:
    """Derive all Section 3.6 quantities from independent inputs; raise on any missing input.

    Spec: model.md, "Derivation formulas" and Section 3.6 derived table.
    """
    raise NotImplementedError


def validate(params: Params, u0: np.ndarray, options: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validation report; raise ValueError on invalid inputs.

    Checks: finite values, Vu > 0, positive masses, 0 <= a0 <= 1, known scenario/supply_mode,
    and that t_cut / options["dt"] is an integer to round-off (Section 8.5).
    """
    raise NotImplementedError


def build_initial_state(params: Params) -> np.ndarray:
    """u0 = [x0, h0, V0, gamma0, mp0, ms0, mu0, a0], shape (8,)."""
    raise NotImplementedError


def write_input_manifest(params: Params, path: str) -> None:
    """Write the frozen input file: value, SI unit, and provenance label per input (Section 3.7 freeze rule)."""
    raise NotImplementedError
