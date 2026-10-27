"""Exact-solution convergence of the four production steppers on scalar problems with known solutions, for the
report's convergence table and figure. validation/test_scalar_orders.py checks the same orders.

Run: python -m validation.scalar_convergence
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Callable

from src.integrators import step_ab3, step_be, step_fe, step_rk4

EXPECTED_ORDER = {"fe": 1, "rk4": 4, "ab3": 3, "be": 1}
T_END = 1.0
# dt = 0.2 / 2^j, j = 0..6; the finest RK4 error is still well above round-off.
N_STEPS_SEQUENCE = tuple(5 * 2 ** j for j in range(7))
PROBLEMS: dict[str, dict[str, Any]] = {
    "decay": {"equation": "u' = -u, u(0) = 1", "f": lambda t, u: -u, "u0": 1.0,
              "exact": lambda t: math.exp(-t)},
    "time_dependent": {"equation": "u' = t - u, u(0) = 1", "f": lambda t, u: t - u, "u0": 1.0,
                       "exact": lambda t: t - 1.0 + 2.0 * math.exp(-t)},
}
OUTPUT_PATH = os.path.join("results", "data", "scalar_convergence.json")
_SINGLE_STEPS = {"fe": step_fe, "rk4": step_rk4}


def integrate(method: str, f: Callable[[float, float], float], u0: float, t_end: float, n: int) -> tuple[float, int]:
    """Take n fixed steps of dt = t_end / n with times k * dt; returns (u(t_end), rhs_calls).

    AB3 caches f at every accepted state and takes its first two steps with RK4, as the driver does.
    """
    dt = t_end / n
    u, calls = u0, 0
    if method == "ab3":
        f_hist = [f(0.0, u0)]
        calls = 1
    for k in range(n):
        if method == "ab3":
            if k < 2:
                u, c = step_rk4(f, k * dt, u, dt, f_hist[-1])
            else:
                u, c = step_ab3(u, (f_hist[-1], f_hist[-2], f_hist[-3]), dt), 0
            f_hist.append(f((k + 1) * dt, u))
            c += 1
        elif method == "be":
            res = step_be(f, k * dt, u, dt)
            if not res.success:
                raise RuntimeError(f"BE step {k} failed: {res.message}")
            u, c = res.u, res.n_rhs
        else:
            u, c = _SINGLE_STEPS[method](f, k * dt, u, dt)
        calls += c
    return u, calls


def convergence_table(problem: str, method: str, n_steps: tuple[int, ...] = N_STEPS_SEQUENCE) -> dict[str, Any]:
    """Global error at T_END, RHS calls, and observed order log2(E(dt) / E(dt/2)) for each step count."""
    spec = PROBLEMS[problem]
    exact = spec["exact"](T_END)
    runs = [integrate(method, spec["f"], spec["u0"], T_END, n) for n in n_steps]
    errors = [abs(u - exact) for u, _ in runs]
    return {"dt": [T_END / n for n in n_steps], "n_steps": list(n_steps), "error": errors,
            "rhs_calls": [calls for _, calls in runs],
            "observed_order": [math.log2(e0 / e1) for e0, e1 in zip(errors, errors[1:])]}


def main() -> None:
    report = {"t_end": T_END, "expected_order": EXPECTED_ORDER, "problems": {}}
    for problem, spec in PROBLEMS.items():
        report["problems"][problem] = {"equation": spec["equation"],
                                       "methods": {m: convergence_table(problem, m) for m in EXPECTED_ORDER}}
        print(f"{spec['equation']}: observed order at the finest pair "
              + ", ".join(f"{m.upper()} {t['observed_order'][-1]:.3f}"
                          for m, t in report["problems"][problem]["methods"].items()))
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
