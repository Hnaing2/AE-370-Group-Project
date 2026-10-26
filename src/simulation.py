"""Powered/coast driver, apogee localization, interpolation, and metrics.

Spec: docs/spec/numerics.md (Sections 7.1, 7.2, 8.3, 8.5); metrics in
docs/spec/verification_studies.md (Section 12).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import numpy as np

METHODS = ("fe", "rk4", "ab3", "be")
STATUSES = ("completed", "no_apogee", "physical_model_invalid", "solver_failed")

DEFAULT_OPTIONS: dict[str, Any] = {
    "method": "rk4",
    "dt": 5.0e-4,             # s; t_cut / dt must be an integer
    "be_residual_tol": 1e-10,
    "be_solver_options": {},
    "t_end": None,            # None -> params.t_end
}


@dataclass
class Segment:
    phase: str                # "powered" or "coast"
    t: np.ndarray             # (n+1,), computed as t0 + k*dt
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


def simulate(method: str, params: Any, options: Mapping[str, Any]) -> SimulationResult:
    """Integrate powered steps 1..N_cut, then coast until the step after apogee (or t_end).

    The cutoff state starts the coast segment; AB3 restarts there from two RK4 steps (Section 8.3).
    """
    raise NotImplementedError


def hermite_interpolate(t0: float, t1: float, u0: np.ndarray, u1: np.ndarray,
                        f0: np.ndarray, f1: np.ndarray, t_query: float) -> np.ndarray:
    """Cubic Hermite interpolant within one step of one phase segment (Section 7.2)."""
    raise NotImplementedError


def find_apogee(segment: Segment) -> Optional[dict[str, Any]]:
    """Bracket the sign change of V sin(gamma) in the coast segment; bisect the Hermite interpolant."""
    raise NotImplementedError


def sample_at_time(result: SimulationResult, t_query: float, phase: Optional[str] = None) -> np.ndarray:
    """State at t_query by Hermite interpolation inside one segment; never across t_cut.

    At t_query == t_cut, phase selects the powered or coast side.
    """
    raise NotImplementedError


def derived_history(result: SimulationResult) -> dict[str, np.ndarray]:
    """Recompute Pu, T_ax, qr, r, Mach, specific force, ... from the stored U of each segment."""
    raise NotImplementedError


def compute_metrics(result: SimulationResult) -> dict[str, Any]:
    """Pressure extrema/droop, thrust and impulse (mass-use identity), propellant/pressurant at cutoff,
    x/h at t_eval, apogee/time/downrange, peak axial specific force, peak Mach, helium drift, status.
    """
    raise NotImplementedError
