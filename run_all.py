"""Reproduce the full workflow: tests, input/environment manifest, studies, timestep selection, sweeps, and figures.

Run from the repo root: python run_all.py
"""

from __future__ import annotations

import subprocess
import sys

STAGES = (
    [sys.executable, "-m", "pytest", "-q"],
    [sys.executable, "-m", "results.manifest"],
    [sys.executable, "-m", "validation.scalar_convergence"],
    [sys.executable, "-m", "validation.stability"],
    [sys.executable, "-m", "validation.convergence"],
    [sys.executable, "-m", "validation.timestep_selection"],
    [sys.executable, "-m", "validation.atmosphere_check"],
    [sys.executable, "-m", "results.sweeps"],
    [sys.executable, "-m", "results.figures"],
)


def main() -> int:
    for cmd in STAGES:
        print(">>", " ".join(cmd[1:]), flush=True)
        code = subprocess.call(cmd)
        if code != 0:
            print(f"stage failed with exit code {code}", file=sys.stderr)
            return code
    return 0


if __name__ == "__main__":
    sys.exit(main())
