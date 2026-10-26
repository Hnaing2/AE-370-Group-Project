"""M7: reference solutions, error histories, observed orders, and measured cost (verification_studies.md Sections 11.2, 11.3).

Run: python -m validation.convergence
"""

from __future__ import annotations

from typing import Any, Sequence

# Halving sequence of steps that each divide t_cut = 7.5 s (dt = 2.5 ms / 2^j).
N_CUT_VALUES = (3000, 6000, 12000, 24000, 48000)


def build_reference(params: Any, n_cut: int) -> dict[str, Any]:
    """Self-written RK4 at a fine dt, refined once more; report the change in every metric vs its tolerance."""
    raise NotImplementedError


def history_errors(result: Any, reference: Any) -> dict[str, float]:
    """E_P and E_T over the powered interval on a shared grid with phase-aware interpolation, plus pressure-extrema errors."""
    raise NotImplementedError


def convergence_study(cases: Sequence[Any], methods: Sequence[str], n_cut_values: Sequence[int],
                      references: Any) -> list[dict[str, Any]]:
    """One row per method/case/dt: status, history and output errors, observed order, RHS/Jacobian calls,
    AB3 restarts, nonlinear iterations, wall time."""
    raise NotImplementedError


def main() -> None:
    """All four methods on the nominal case and difficult corners; save rows to results/data/."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
