"""Constitutive model, right-hand side, and Jacobians.

Spec: docs/spec/model.md (Sections 5, 6); docs/spec/numerics.md (Sections 7.2, 9.1, 9.2).
"""

from __future__ import annotations

import math
from typing import Any, Callable, Optional

import numpy as np

# State indices for u = [x, h, V, gamma, mp, ms, mu, a].
IX, IH, IV, IGAMMA, IMP, IMS, IMU, IA = range(8)
STATE_NAMES = ("x", "h", "V", "gamma", "mp", "ms", "mu", "a")
PHASES = ("powered", "coast")
A_BOUND_TOL = 1e-12  # documented roundoff allowance at the valve-opening bounds (Section 5.2)


class PhysicalModelInvalid(ValueError):
    """A state left the physical domain; the driver marks the run physical_model_invalid (Section 8.5)."""


def critical_pressure_ratio(gamma_he: float) -> float:
    """r_* = (2/(gamma_he+1))^(gamma_he/(gamma_he-1)) (Section 5.4)."""
    return (2.0 / (gamma_he + 1.0)) ** (gamma_he / (gamma_he - 1.0))


def flow_function(r: float, gamma_he: float) -> float:
    """Phi(r) for the choked/subcritical orifice law, r = Pu/Ps (Section 5.4)."""
    if r < 0.0:
        raise ValueError(f"pressure ratio must be nonnegative, got {r}")
    if r >= 1.0:
        return 0.0  # check valve: no reverse flow
    if r <= critical_pressure_ratio(gamma_he):
        return 1.0
    g = gamma_he
    num = np.sqrt(2.0 / (g - 1.0) * (r ** (2.0 / g) - r ** ((g + 1.0) / g)))
    return float(num / (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0))))


def derived_quantities(u: np.ndarray, params: Any, phase: str) -> dict[str, Any]:
    """Return M, Vu, Pu, Ps, Ts, r, rho, g, D, F_eng, T_ax, qp, qr, a_cmd and validity flags.

    Engine force, qp, and qr are zero in coast (Section 6.2). Raise on invalid geometry, gas
    state, masses, or V <= 0 rather than patching the state (Section 5.2).
    """
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}, got {phase!r}")
    if len(u) != 8:
        raise ValueError(f"state must have 8 entries, got {len(u)}")
    x, h, V, gamma, mp, ms, mu, a = (float(v) for v in u)
    if not all(math.isfinite(v) for v in (x, h, V, gamma, mp, ms, mu, a)):
        raise PhysicalModelInvalid(f"nonfinite state: {u}")
    if V <= 0.0:
        raise PhysicalModelInvalid(f"speed must be positive, got V = {V}")
    if mp <= 0.0 or ms <= 0.0 or mu <= 0.0:
        raise PhysicalModelInvalid(f"masses must be positive, got mp = {mp}, ms = {ms}, mu = {mu}")
    if not -A_BOUND_TOL <= a <= 1.0 + A_BOUND_TOL:
        raise PhysicalModelInvalid(f"valve opening outside [0, 1]: a = {a}")
    Vu = params.Vtank - mp / params.rho_p
    if Vu <= 0.0:
        raise PhysicalModelInvalid(f"ullage volume must be positive, got Vu = {Vu}")
    M = params.m_dry + mp + ms + mu
    if M <= 0.0:
        raise PhysicalModelInvalid(f"vehicle mass must be positive, got M = {M}")
    if params.R_E + h <= 0.0:
        raise PhysicalModelInvalid(f"altitude below Earth's center: h = {h}")

    # Thermodynamics (Section 5.2): isothermal ullage, isentropic supply.
    gam = params.gamma_he
    Pu = mu * params.R_he * params.Tu / Vu
    ms_ratio = ms / params.ms0
    Ps = params.Ps0 * ms_ratio ** gam
    Ts = params.Ts0 * ms_ratio ** (gam - 1.0)
    r = Pu / Ps

    # Environment and drag.
    rho = params.rho0 * math.exp(-h / params.H)
    g = params.g0 * (params.R_E / (params.R_E + h)) ** 2
    D = 0.5 * rho * V * V * params.Cd * params.A_ref

    powered = phase == "powered"
    regulated = params.scenario == "regulated"
    if powered:
        F_eng = params.K_F * Pu                           # Section 5.1
        T_ax = F_eng * math.cos(params.delta)
        qp = F_eng / (params.Isp * params.g0)
        qr = a * params.C_reg * Ps / math.sqrt(Ts) * flow_function(r, gam)  # Section 5.4
    else:
        F_eng = T_ax = qp = qr = 0.0                      # Section 6.2
    if powered and regulated:
        a_cmd = min(max(params.K_p * (params.Pset - Pu), 0.0), 1.0)  # Section 5.3
    else:
        a_cmd = a  # valve held: fixed orifice, or frozen at its cutoff value in coast

    return {
        "M": M, "Vu": Vu, "Pu": Pu, "Ps": Ps, "Ts": Ts, "r": r,
        "rho": rho, "g": g, "D": D,
        "F_eng": F_eng, "T_ax": T_ax, "qp": qp, "qr": qr, "a_cmd": a_cmd,
        "choked": r <= params.r_star,
        "Pu_below_half_Pset": powered and Pu < 0.5 * params.Pset,  # assumed engine-validity flag
    }


def rhs(t: float, u: np.ndarray, params: Any, phase: str) -> np.ndarray:
    """du/dt, shape (8,); pure function of its arguments (Sections 6.1, 6.2, 7.2)."""
    q = derived_quantities(u, params, phase)
    V, gamma, a = float(u[IV]), float(u[IGAMMA]), float(u[IA])
    sin_g, cos_g = math.sin(gamma), math.cos(gamma)

    du = np.zeros(8)
    du[IX] = V * cos_g
    du[IH] = V * sin_g
    du[IV] = (q["T_ax"] - q["D"]) / q["M"] - q["g"] * sin_g
    du[IGAMMA] = -q["g"] * cos_g / V
    du[IMP] = -q["qp"]
    du[IMS] = -q["qr"]
    du[IMU] = q["qr"]
    du[IA] = (q["a_cmd"] - a) / params.tau_v
    if params.scenario == "fixed_orifice":
        du[IA] = 0.0
    if phase == "coast":
        du[IMP:] = 0.0  # freeze mp, ms, mu, a
    return du


def state_scales(params: Any) -> np.ndarray:
    """Characteristic magnitude of each state: the floor for FD perturbation sizes (Section 9.2)
    and the BE variable/residual scaling S (Section 8.4)."""
    return np.array([params.h0, params.h0, params.V0, 1.0,
                     params.mp0, params.ms0, params.mu0, 1.0], dtype=np.float64)


def jacobian(t: float, u: np.ndarray, params: Any, phase: str, rel_step: float = 1e-6,
             f: Optional[Callable[[float, np.ndarray], np.ndarray]] = None) -> np.ndarray:
    """J = d(rhs)/du by scaled central finite differences, shape (8, 8) (Section 9.2).

    Each perturbation is rel_step * max(|u_j|, scale_j). Where a central perturbation of the valve
    opening would leave [0, 1], a second-order one-sided difference is used instead, so the
    Jacobian is defined on the bounds (e.g. a0 = 0) without evaluating outside the physical domain.
    f, if given, is the phase closure f(t, u) = rhs(t, u, params, phase) to difference; the BE
    driver passes a counting closure so every finite-difference RHS call is counted.
    """
    if f is None:
        f = lambda t, v: rhs(t, v, params, phase)
    u = np.asarray(u, dtype=np.float64)
    steps = rel_step * np.maximum(np.abs(u), state_scales(params))
    J = np.empty((8, 8))
    for j in range(8):
        e = np.zeros(8)
        e[j] = steps[j]
        if j == IA and u[j] - steps[j] < -A_BOUND_TOL:
            sides = (0.0, 1.0, 2.0)    # forward
        elif j == IA and u[j] + steps[j] > 1.0 + A_BOUND_TOL:
            sides = (0.0, -1.0, -2.0)  # backward
        else:
            sides = None
        if sides is None:
            J[:, j] = (f(t, u + e) - f(t, u - e)) / (2.0 * steps[j])
        else:
            f0, f1, f2 = (f(t, u + s * e) for s in sides)
            J[:, j] = (-3.0 * f0 + 4.0 * f1 - f2) / (2.0 * sides[1] * steps[j])
    return J


def analytic_jacobian_choked(t: float, u: np.ndarray, params: Any) -> np.ndarray:
    """Analytic Jacobian for the regulated, powered, choked branch only; cross-check (Section 9.1)."""
    raise NotImplementedError


# 1976 US Standard Atmosphere: layers by geopotential altitude (base m, base temperature K, lapse K/m).
_US76_LAYERS = (
    (0.0, 288.15, -0.0065),
    (11000.0, 216.65, 0.0),
    (20000.0, 216.65, 0.001),
    (32000.0, 228.65, 0.0028),
    (47000.0, 270.65, 0.0),
    (51000.0, 270.65, -0.0028),
    (71000.0, 214.65, -0.002),
)
_US76_TOP = 84852.0           # m geopotential, top of the last layer (~86 km geometric)
_US76_R0 = 6356766.0          # m, Earth radius used for geopotential altitude
_US76_G0 = 9.80665            # m/s^2
_US76_P0 = 101325.0           # Pa
R_AIR_1976 = 8314.32 / 28.9644  # J/(kg K), R* / M0


def _us76_layer_pressure(Pb: float, Tb: float, lapse: float, dH: float) -> float:
    if lapse == 0.0:
        return Pb * math.exp(-_US76_G0 * dH / (R_AIR_1976 * Tb))
    return Pb * (Tb / (Tb + lapse * dH)) ** (_US76_G0 / (R_AIR_1976 * lapse))


def _us76_base_pressures() -> tuple[float, ...]:
    P = [_US76_P0]
    for (Hb, Tb, lapse), (Hn, _, _) in zip(_US76_LAYERS[:-1], _US76_LAYERS[1:]):
        P.append(_us76_layer_pressure(P[-1], Tb, lapse, Hn - Hb))
    return tuple(P)


_US76_BASE_P = _us76_base_pressures()


def standard_atmosphere_1976(h: float) -> tuple[float, float]:
    """(density, temperature) from the piecewise 1976 US Standard Atmosphere at geometric altitude h.

    Valid 0-86 km (the seven layers below 84.852 km geopotential). Post-processing only: Mach
    number and the nominal atmosphere sensitivity check (Section 3.7).
    """
    H = _US76_R0 * h / (_US76_R0 + h)  # geopotential altitude
    if not 0.0 <= H <= _US76_TOP:
        raise ValueError(f"altitude outside the 1976 Standard Atmosphere table: h = {h}")
    i = max(k for k, layer in enumerate(_US76_LAYERS) if layer[0] <= H)
    Hb, Tb, lapse = _US76_LAYERS[i]
    T = Tb + lapse * (H - Hb)
    P = _us76_layer_pressure(_US76_BASE_P[i], Tb, lapse, H - Hb)
    return P / (R_AIR_1976 * T), T
