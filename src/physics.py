"""Constitutive model, right-hand side, and Jacobians.

Spec: docs/spec/model.md (Sections 5, 6); docs/spec/numerics.md (Sections 7.2, 9.1, 9.2).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# State indices for u = [x, h, V, gamma, mp, ms, mu, a].
IX, IH, IV, IGAMMA, IMP, IMS, IMU, IA = range(8)
STATE_NAMES = ("x", "h", "V", "gamma", "mp", "ms", "mu", "a")
PHASES = ("powered", "coast")


def critical_pressure_ratio(gamma_he: float) -> float:
    """r_* = (2/(gamma_he+1))^(gamma_he/(gamma_he-1)) (Section 5.4)."""
    raise NotImplementedError


def flow_function(r: float, gamma_he: float) -> float:
    """Phi(r) for the choked/subcritical orifice law, r = Pu/Ps (Section 5.4)."""
    raise NotImplementedError


def derived_quantities(u: np.ndarray, params: Any, phase: str) -> dict[str, Any]:
    """Return M, Vu, Pu, Ps, Ts, r, rho, g, D, F_eng, T_ax, qp, qr, a_cmd and validity flags.

    Engine force, qp, and qr are zero in coast (Section 6.2). Raise on invalid geometry, gas
    state, masses, or V <= 0 rather than patching the state (Section 5.2).
    """
    raise NotImplementedError


def rhs(t: float, u: np.ndarray, params: Any, phase: str) -> np.ndarray:
    """du/dt, shape (8,); pure function of its arguments (Sections 6.1, 6.2, 7.2)."""
    raise NotImplementedError


def jacobian(t: float, u: np.ndarray, params: Any, phase: str, rel_step: float = 1e-6) -> np.ndarray:
    """J = d(rhs)/du by scaled central finite differences, shape (8, 8) (Section 9.2)."""
    raise NotImplementedError


def analytic_jacobian_choked(t: float, u: np.ndarray, params: Any) -> np.ndarray:
    """Analytic Jacobian for the regulated, powered, choked branch only; cross-check (Section 9.1)."""
    raise NotImplementedError


def standard_atmosphere_1976(h: float) -> tuple[float, float]:
    """(density, temperature) from the piecewise 1976 US Standard Atmosphere, 0-50 km.

    Post-processing only: Mach number and the nominal atmosphere sensitivity check (Section 3.7).
    """
    raise NotImplementedError
