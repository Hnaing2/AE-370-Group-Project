"""Reproduce the mission reference outputs of an independent RK4 prototype with RK4 at dt = 0.5 ms."""

import functools

import pytest

from src.parameters import FIXED_ORIFICE_CASE, REFERENCE_CASE, build_params
from src.simulation import compute_metrics, simulate

REL_TOL = 1e-3  # ~0.1%
DT = 5.0e-4

CASES = {"regulated": REFERENCE_CASE, "fixed_orifice": FIXED_ORIFICE_CASE}

# Reference outputs from an independent RK4 prototype (SI units).
EXPECTED = {
    "regulated": {
        "apogee_h": 40057.0,
        "apogee_t": 65.2,
        "apogee_x": 21.1e3,
        "x_eval": 4954.0,
        "h_eval": 27501.0,
        "V_cut": 738.9,
        "h_cut": 23175.0,
        "mp_cut": 745.7,
        "Pu_cut": 7.767e6,
        "a_cut": 0.579,
        "peak_specific_force_g": 3.85,
        "peak_mach": 2.49,
    },
    "fixed_orifice": {
        "apogee_h": 38968.0,
        "apogee_t": 63.4,
        "apogee_x": 19.9e3,
        "x_eval": 4842.0,
        "h_eval": 27298.0,
        "V_cut": 713.4,
        "h_cut": 23126.0,
        "mp_cut": 881.5,
        "Pu_cut": 6.556e6,
        "a_cut": 0.231,
        "peak_specific_force_g": 3.65,
        "peak_mach": 2.40,
    },
}
# Values printed with 3 significant figures (apogee_t, apogee_x, a_cut, g, Mach) are compared at
# their printed precision rather than 0.1%: half a unit in the last printed digit.
PRINTED_HALF_UNIT = {
    "apogee_t": 0.05,
    "apogee_x": 50.0,
    "a_cut": 5e-4,
    "peak_specific_force_g": 5e-3,
    "peak_mach": 5e-3,
}


@functools.lru_cache(maxsize=None)
def reference_run(scenario):
    result = simulate("rk4", build_params(CASES[scenario]), {"dt": DT})
    return result, compute_metrics(result)


@pytest.mark.parametrize("scenario", list(EXPECTED))
def test_reference_outputs(scenario):
    result, metrics = reference_run(scenario)
    assert result.status == "completed"
    for key, expected in EXPECTED[scenario].items():
        if key in PRINTED_HALF_UNIT:
            assert metrics[key] == pytest.approx(expected, abs=PRINTED_HALF_UNIT[key]), key
        else:
            assert metrics[key] == pytest.approx(expected, rel=REL_TOL), key
    # The reference reports Pu at cutoff as the minimum powered Pu.
    assert metrics["Pu_min"] == pytest.approx(metrics["Pu_cut"], rel=1e-12)


@pytest.mark.parametrize("scenario", list(EXPECTED))
def test_helium_drift_rk4(scenario):
    """Relative helium drift ~1e-14."""
    _, metrics = reference_run(scenario)
    assert metrics["helium_drift"] < 1e-13
