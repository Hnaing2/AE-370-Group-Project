"""M4 gate: reproduce the Section 3.8 reference outputs with RK4 at dt = 0.5 ms (model.md Section 3.8)."""

import pytest

REL_TOL = 1e-3  # ~0.1%

# Section 3.8 reference outputs (SI units).
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
# their printed precision rather than 0.1%.


@pytest.mark.parametrize("scenario", list(EXPECTED))
def test_reference_outputs(scenario):
    pytest.skip("M4: not implemented")


@pytest.mark.parametrize("scenario", list(EXPECTED))
def test_helium_drift_rk4(scenario):
    """Relative helium drift ~1e-14."""
    pytest.skip("M4: not implemented")
