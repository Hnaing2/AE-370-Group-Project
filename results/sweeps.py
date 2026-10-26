"""M7: final parameter studies with the selected method and dt (verification_studies.md Section 12).

Run: python -m results.sweeps
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

KAPPA_VALUES = (5.0, 10.0, 20.0, 50.0, 100.0)
TAU_V_VALUES = (0.005, 0.01, 0.02, 0.05, 0.1)   # s
CD_VALUES = (0.4, 0.6, 0.8)
GAMMA0_VALUES = (0.93, 1.10, 1.27)               # rad


def regulator_grid_cases(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    """5 x 5 tau_v x kappa grid at nominal Cd and gamma0."""
    raise NotImplementedError


def cd_gamma0_grid_cases(base: Mapping[str, Any]) -> list[dict[str, Any]]:
    """3 x 3 Cd x gamma0 grid at nominal regulator settings (research question 3)."""
    raise NotImplementedError


def parameter_sweep(cases: Sequence[Mapping[str, Any]], method: str, dt: float) -> list[dict[str, Any]]:
    """One metrics row per case; save a parameter snapshot for each case."""
    raise NotImplementedError


def main() -> None:
    """Regulator grid, Cd x gamma0 grid (check identical pressure histories), fixed-orifice case;
    write CSV/JSON to results/data/."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
