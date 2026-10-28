"""Frozen input file and run environment, written before the studies so that the inputs, their provenance,
and a clean-run command reproduce the conclusions.

Run: python -m results.manifest (run_all.py runs it before the studies)
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from importlib import metadata
from typing import Any, Optional

from results.sweeps import CD_VALUES, DATA_DIR, GAMMA0_VALUES, KAPPA_VALUES, TAU_V_VALUES
from src.parameters import REFERENCE_CASE, STUDY_BASELINE, UNITS, build_params, write_input_manifest
from validation.convergence import WORKERS

INPUTS_PATH = os.path.join(DATA_DIR, "inputs.json")
ENVIRONMENT_PATH = os.path.join(DATA_DIR, "environment.json")
PACKAGES = ("numpy", "scipy", "matplotlib", "pytest")
PROVENANCE_SOURCES = ("Each input carries a provenance label (sourced, nominal, assumed, calibrated, sized, "
                      "study choice, exploratory range); the report's parameter table gives the source or "
                      "justification for each.")


def exploratory_grids() -> dict[str, Any]:
    """The swept values. Only the nominal values carry the inputs' own labels (gamma0 = 1.10 rad is calibrated,
    Cd = 0.6 assumed); the other grid values are exploratory bounds, not uncertainty intervals."""
    grids = {"tau_v": TAU_V_VALUES, "kappa": KAPPA_VALUES, "Cd": CD_VALUES, "gamma0": GAMMA0_VALUES}
    return {k: {"values": list(v), "unit": UNITS[k], "label": "exploratory range"} for k, v in grids.items()}


def _git(*args: str) -> Optional[str]:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def environment() -> dict[str, Any]:
    """Interpreter, platform, package versions, commit, and process-pool size of this run. Uncommitted changes
    count tracked files only, since the generated results/ and report/figures/ files are untracked."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: metadata.version(name) for name in PACKAGES},
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "uncommitted_changes": None if status is None else bool(status),
        "cpu_count": os.cpu_count(),
        "workers": WORKERS,
        "command": "python run_all.py",
    }


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    write_input_manifest(build_params(STUDY_BASELINE), INPUTS_PATH, extra={
        "study_baseline_equals_reference_case": dict(STUDY_BASELINE) == dict(REFERENCE_CASE),
        "exploratory_grids": exploratory_grids(),
        "provenance_sources": PROVENANCE_SOURCES,
    })
    env = environment()
    with open(ENVIRONMENT_PATH, "w", encoding="utf-8") as fh:
        json.dump(env, fh, indent=2)
    print(f"wrote {INPUTS_PATH} and {ENVIRONMENT_PATH} (commit {env['git_commit']}, "
          f"uncommitted changes: {env['uncommitted_changes']})")


if __name__ == "__main__":
    main()
