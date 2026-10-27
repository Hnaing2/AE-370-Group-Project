"""Self-written single-step integrators and their stability functions.

Each step takes f(t, u) and never sees params or phase, so it also runs on scalar test problems.
Spec: docs/spec/numerics.md (Sections 8.1-8.4, 9.4).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from scipy import optimize

RHS = Callable[[float, np.ndarray], np.ndarray]


@dataclass
class BEStepResult:
    u: np.ndarray            # accepted next state (only meaningful when success is True)
    f: np.ndarray            # f(t + dt, u), the derivative at the returned state
    success: bool
    residual: float          # achieved max scaled residual
    n_rhs: int               # every call to f, including finite-difference Jacobians when jac is None
    n_jac: int               # Jacobian evaluations, supplied or finite-difference
    n_iter: Optional[int]    # nonlinear iterations, when the solver exposes them
    message: str


# Default BE solve: Newton on the scaled residual (Section 8.4). It stops when the scaled correction
# max|dw| <= step_tol; acceptance is then decided by the achieved residual alone. scipy's hybr is kept
# as a cross-check (solver_options={"method": "hybr", ...}); it stops on its trust-region size, which
# misreports "no progress" once the residual already sits at round-off (README notes).
NEWTON_DEFAULTS = {"step_tol": 1e-12, "max_iter": 20}
_FD_REL_STEP = 1.49e-8  # sqrt(machine epsilon), forward differences when no Jacobian is supplied


def _newton(residual: Callable, residual_jac: Callable, w: np.ndarray, step_tol: float, max_iter: int):
    """Newton iteration for G(w) = 0, with residual(w) -> (G, f) and residual_jac(w, G) -> dG/dw.

    The Newton matrix is kept while the residual decreases and re-evaluated at the current iterate
    when it does not. Returns (w, G, f, iterations, converged, message) at the last iterate.
    """
    G, fw = residual(w)
    J = residual_jac(w, G)
    for k in range(1, max_iter + 1):
        try:
            dw = np.linalg.solve(J, -G)
        except np.linalg.LinAlgError as exc:
            return w, G, fw, k - 1, False, f"singular Newton matrix ({exc})"
        w = w + dw
        G_new, fw = residual(w)
        if np.max(np.abs(dw)) <= step_tol:
            return w, G_new, fw, k, True, "converged"
        if not np.max(np.abs(G_new)) < np.max(np.abs(G)):
            J = residual_jac(w, G_new)
        G = G_new
    return w, G, fw, max_iter, False, f"no convergence in {max_iter} Newton iterations"


def step_fe(f: RHS, t: float, u: np.ndarray, dt: float, f0: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
    """Forward Euler. Reuse f0 = f(t, u) if supplied. Returns (u_next, rhs_calls)."""
    calls = 0
    if f0 is None:
        f0 = f(t, u)
        calls += 1
    return u + dt * f0, calls


def step_rk4(f: RHS, t: float, u: np.ndarray, dt: float, f0: Optional[np.ndarray] = None) -> tuple[np.ndarray, int]:
    """Classical RK4. Reuse f0 = f(t, u) as k1 if supplied. Returns (u_next, rhs_calls)."""
    calls = 3
    k1 = f0
    if k1 is None:
        k1 = f(t, u)
        calls += 1
    half = 0.5 * dt
    k2 = f(t + half, u + half * k1)
    k3 = f(t + half, u + half * k2)
    k4 = f(t + dt, u + dt * k3)
    return u + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4), calls


def step_ab3(u: np.ndarray, f_hist: Sequence[np.ndarray], dt: float) -> np.ndarray:
    """AB3 with f_hist = (f_n, f_{n-1}, f_{n-2}), equally spaced; history is managed by the driver.

    No RHS calls: the driver stores f at every accepted state and starts (and restarts) with RK4.
    """
    f_n, f_n1, f_n2 = f_hist
    return u + (dt / 12.0) * (23.0 * f_n - 16.0 * f_n1 + 5.0 * f_n2)


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
    """Backward Euler: solve the scaled residual G(z) = z - u - dt f(t+dt, z) for z = u_{n+1}.

    With z = S w the solve is in w: G_w = w - w_n - dt S^-1 f(t+dt, S w), Jacobian I - dt S^-1 J S,
    where jac(t, z) = df/dz (forward differences of G_w if None) and S = scales (default ones). The
    iteration starts from the current state and never clips an iterate. Accept only on solver
    convergence and achieved scaled residual max|G_w| <= residual_tol (Section 8.4); the residual and
    the returned f are from f evaluated at the returned state itself.

    solver_options: Newton settings overriding NEWTON_DEFAULTS, or {"method": <scipy.optimize.root
    method>, ...} with the remaining keys passed to scipy.optimize.root. Exceptions raised by f or
    jac propagate.
    """
    shape = np.shape(u)
    u_n = np.atleast_1d(np.asarray(u, dtype=np.float64))
    S = np.ones_like(u_n) if scales is None else np.broadcast_to(np.asarray(scales, dtype=np.float64), u_n.shape)
    if not np.all(S > 0.0):
        raise ValueError(f"scales must be positive, got {scales}")
    t_new = t + dt
    w_n = u_n / S
    n_rhs = n_jac = 0

    def residual(w):
        nonlocal n_rhs
        n_rhs += 1
        fw = np.atleast_1d(f(t_new, (S * w).reshape(shape)))
        return w - w_n - dt * fw / S, fw

    def residual_jac(w, G):
        nonlocal n_jac
        n_jac += 1
        if jac is None:
            h = _FD_REL_STEP * np.maximum(np.abs(w), 1.0)
            return np.column_stack([(residual(w + h[j] * np.eye(w.size)[j])[0] - G) / h[j]
                                    for j in range(w.size)])
        J = np.atleast_2d(jac(t_new, (S * w).reshape(shape)))
        return np.eye(w.size) - dt * J * S[np.newaxis, :] / S[:, np.newaxis]

    settings = dict(solver_options or {})
    method = settings.pop("method", "newton")
    if method == "newton":
        w, G, f_new, n_iter, converged, why = _newton(residual, residual_jac, w_n.copy(),
                                                      **{**NEWTON_DEFAULTS, **settings})
    else:
        sol = optimize.root(lambda w: residual(w)[0], w_n.copy(), method=method,
                            jac=None if jac is None else (lambda w: residual_jac(w, None)), **settings)
        w = sol.x
        G, f_new = residual(w)
        n_iter, converged, why = getattr(sol, "nit", None), bool(sol.success), str(sol.message)
    achieved = float(np.max(np.abs(G)))
    success = converged and achieved <= residual_tol
    if success:
        message = "converged"
    elif converged:
        message = f"solver converged but scaled residual {achieved:.3e} > {residual_tol:.1e}"
    else:
        message = f"solver failed: {why}"
    return BEStepResult(
        u=(S * w).reshape(shape), f=f_new.reshape(shape), success=success, residual=achieved,
        n_rhs=n_rhs, n_jac=n_jac, n_iter=n_iter, message=message,
    )


_UNIT_ROOT_TOL = 1e-12  # |q| within this of 1 counts as a unit-modulus root; roots closer than this coincide


def amplification_factor(method: str, z: complex) -> complex:
    """R(z) for "fe", "rk4", "be" (Section 9.4)."""
    if method == "fe":
        return 1.0 + z
    if method == "rk4":
        return 1.0 + z * (1.0 + z / 2.0 * (1.0 + z / 3.0 * (1.0 + z / 4.0)))
    if method == "be":
        return 1.0 / (1.0 - z)
    raise ValueError(f"amplification factor defined for 'fe', 'rk4', 'be', got {method!r}")


def ab3_characteristic_roots(z: complex) -> np.ndarray:
    """Roots of q^3 - (1 + 23z/12) q^2 + (16z/12) q - 5z/12 = 0 (Section 9.4)."""
    return np.roots([1.0, -1.0 - 23.0 * z / 12.0, 16.0 * z / 12.0, -5.0 * z / 12.0])


def is_absolutely_stable(method: str, z: complex, margin: float = 0.0) -> bool:
    """|R(z)| <= 1 - margin; for AB3 all roots inside the unit disk, unit-modulus roots simple.

    margin = 0 accepts the boundary itself (z = 0 is stable for every method); a positive margin
    demands |R| or every |q| at most 1 - margin, so it only makes sense for a trial z away from 0.
    """
    bound = 1.0 - margin
    if method != "ab3":
        return bool(abs(amplification_factor(method, z)) <= bound + (_UNIT_ROOT_TOL if margin == 0.0 else 0.0))
    q = ab3_characteristic_roots(z)
    if margin > 0.0:
        return bool(np.all(np.abs(q) <= bound))
    if np.any(np.abs(q) > 1.0 + _UNIT_ROOT_TOL):
        return False
    on_circle = q[np.abs(np.abs(q) - 1.0) <= _UNIT_ROOT_TOL]
    return all(abs(a - b) > _UNIT_ROOT_TOL for i, a in enumerate(on_circle) for b in on_circle[i + 1:])
