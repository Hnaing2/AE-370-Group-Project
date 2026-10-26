"""M6: frozen-Jacobian stability characterization along a pilot trajectory (numerics.md Sections 9.3, 9.4).

Run: python -m validation.stability
"""

from __future__ import annotations

from typing import Any


def pilot_sample_times(result: Any) -> list[tuple[float, str]]:
    """(t, phase) samples: initialization, first Pset crossing, regulator transitions, late burn,
    just before cutoff, and coast."""
    raise NotImplementedError


def characterize(result: Any, params: Any) -> list[dict[str, Any]]:
    """Per sample: eigenvalues, neutral/unstable flags, and the largest stable dt for each method,
    checked at the actual complex z = lambda dt (AB3 via all characteristic roots)."""
    raise NotImplementedError


def main() -> None:
    """Characterize the nominal case and the fastest-valve/highest-gain corner; save to results/data/."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
