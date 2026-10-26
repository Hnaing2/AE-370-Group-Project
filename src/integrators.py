"""Self-written single-step integrators and their stability functions.

Each step takes f(t, u) and never sees params or phase, so it also runs on scalar test problems.
Spec: docs/spec/numerics.md (Sections 8.1-8.4, 9.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np

RHS = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class BEStepResult:
    u: np.ndarray            # accepted next state (only meaningful when success is True)
    success: bool
    residual: float          # achieved max scaled residual
    n_rhs: int               # every RHS call, including finite differences
    n_jac: int
    n_iter: Optional[int]    # nonlinear iterations, when the solver exposes them
    message: str


def step_fe(f: RHS, t: float, u: np.ndarray, dt: float, f0: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
    """Forward Euler. Reuse f0 = f(t, u) if supplied. Returns (u_next, rhs_calls)."""
    raise NotImplementedError


def step_rk4(f: RHS, t: float, u: np.ndarray, dt: float, f0: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
    """Classical RK4. Reuse f0 = f(t, u) as k1 if supplied. Returns (u_next, rhs_calls)."""
    raise NotImplementedError


def step_ab3(u: np.ndarray, f_hist: Sequence[np.ndarray], dt: float) -> np.ndarray:
    """AB3 with f_hist = (f_n, f_{n-1}, f_{n-2}), equally spaced; history is managed by the driver."""
    raise NotImplementedError


def step_be(
    f: RHS,
    t: float,
    u: np.ndarray,
    dt: float,
    jac: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
    scales: Optional[np.ndarray] = None,
    residual_tol: float = 1e-10,
    solver_options: Optional[dict] = None,
) -> BEStepResult:
    """Backward Euler: solve the scaled residual G(z) = z - u - dt f(t+dt, z) with scipy.optimize.root.

    Accept only on solver success and achieved scaled residual <= residual_tol (Section 8.4).
    """
    raise NotImplementedError


def amplification_factor(method: str, z: complex) -> complex:
    """R(z) for "fe", "rk4", "be" (Section 9.4)."""
    raise NotImplementedError


def ab3_characteristic_roots(z: complex) -> np.ndarray:
    """Roots of q^3 - (1 + 23z/12) q^2 + (16z/12) q - 5z/12 = 0 (Section 9.4)."""
    raise NotImplementedError


def is_absolutely_stable(method: str, z: complex, margin: float = 0.0) -> bool:
    """|R(z)| <= 1 - margin; for AB3 all roots inside the unit disk, unit-modulus roots simple."""
    raise NotImplementedError
