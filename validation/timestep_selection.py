"""M7: tolerances, E_score, and selection of one method and one nominal dt (verification_studies.md Sections 11.1, 11.4).

Run: python -m validation.timestep_selection
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

# Section 11.1 error targets (SI units). Pressure/thrust/impulse are absolute equivalents of 0.1%.
TOLERANCES: Mapping[str, float] = {
    "pressure_history": 8.0e3,       # Pa, 0.1% of Pset
    "thrust_history": 534.0,         # N, 0.1% of nominal axial force
    "apogee_h": 10.0,                # m
    "x_eval": 10.0,                  # m
    "h_eval": 10.0,                  # m
    "apogee_x": 10.0,                # m
    "apogee_t": 0.05,                # s
    "axial_impulse_rel": 1.0e-3,     # relative to nominal axial impulse
    "peak_specific_force_g": 0.01,   # g0
}
HELIUM_DRIFT_LIMIT = {"fe": 1e-10, "rk4": 1e-10, "ab3": 1e-10, "be": 1e-8}  # validity check, not in E_score
E_SCORE_MAX = 0.5


def required_cases() -> list[dict[str, Any]]:
    """Nominal; regulator corners tau_v {5, 100} ms x kappa {5, 100}; fixed orifice; Cd x gamma0 corners."""
    raise NotImplementedError


def e_score(rows: Sequence[Mapping[str, Any]], tolerances: Mapping[str, float] = TOLERANCES) -> float:
    """max over tested cases and required outputs of |Q_dt - Q_ref| / tol_Q."""
    raise NotImplementedError


def select_timestep(study: Sequence[Mapping[str, Any]], tolerances: Mapping[str, float] = TOLERANCES) -> dict[str, Any]:
    """Choose method/dt with E_score <= 0.5, no invalid states, successful solves, acceptable cost;
    return the choice and the worst-case evidence row."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
