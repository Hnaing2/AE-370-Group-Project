"""M7: the five report figures from saved results/data/ files (verification_studies.md Section 12).

Run: python -m results.figures
"""

from __future__ import annotations

FIGURE_DIR = "report/figures"


def fig_regulator_heatmaps() -> None:
    """1. tau_v x kappa heatmaps of minimum pressure and of apogee."""
    raise NotImplementedError


def fig_apogee_vs_tau_v() -> None:
    """2. Downrange displacement or apogee versus tau_v for selected gains."""
    raise NotImplementedError


def fig_regulated_vs_fixed_orifice() -> None:
    """3. Regulated versus fixed-orifice metrics."""
    raise NotImplementedError


def fig_error_vs_cost() -> None:
    """4. Error versus measured cost, showing why the method/dt was chosen."""
    raise NotImplementedError


def fig_cd_gamma0_sweep() -> None:
    """5. x(t_eval) and apogee versus Cd, one line per gamma0."""
    raise NotImplementedError


def main() -> None:
    raise NotImplementedError


if __name__ == "__main__":
    main()
