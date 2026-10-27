"""Parameter builder and validator."""

import json

import numpy as np
import pytest

from src.parameters import (
    FIXED_ORIFICE_CASE,
    GAS_STARVED_CASE,
    REFERENCE_CASE,
    build_initial_state,
    build_params,
    validate,
    write_input_manifest,
)
from src.physics import flow_function

# Derived quantities at the baseline: (value, printed decimals).
DERIVED_DISPLAY = {
    "rho_p": (1161.6, 1),
    "Vu0": (0.12060, 5),
    "Vtank": (2.3262, 4),
    "K_F": (0.071, 3),
    "q_p_ref": (246.47, 2),
    "ms0": (39.47, 2),
    "Vs": (0.6866, 4),
    "q_r_req": (2.788, 3),
    "a_fix": (0.2311, 4),
    "M0": (12503.1, 1),
}


@pytest.fixture(scope="module")
def ref():
    return build_params(REFERENCE_CASE)


def test_derived_quantities_match_baseline_values(ref):
    """rho_p 1161.6, Vu0 0.12060, Vtank 2.3262, K_F 0.071, q_p_ref 246.47, ms0 39.47, Vs 0.6866,
    q_r_req 2.788, C_reg 5.900e-6, a_fix 0.2311, M0 12503.1."""
    for name, (value, decimals) in DERIVED_DISPLAY.items():
        assert round(getattr(ref, name), decimals) == pytest.approx(value), name
    assert ref.C_reg == pytest.approx(5.900e-6, abs=0.0005e-6)
    assert ref.r_star == pytest.approx(0.4871, abs=5e-5)
    assert ref.Ts_ref == pytest.approx(178.0, abs=0.5)
    assert ref.K_p == pytest.approx(20.0 / 8.0e6)


def test_full_precision_identities(ref):
    """Derived values are internally consistent at full precision (no rounded intermediates)."""
    assert ref.Vtank - ref.mp0 / ref.rho_p == pytest.approx(ref.Vu0, rel=1e-14)
    assert ref.mu0 * ref.R_he * ref.Tu / ref.Vu0 == pytest.approx(ref.Pu0, rel=1e-14)
    assert ref.M0 == ref.m_dry + ref.mp0 + ref.ms0 + ref.mu0
    # Sized supply reaches Ps_f exactly when the ullage holds Pset over Vu0 + burned volume.
    ms_end = ref.ms0 * (ref.Ps_f / ref.Ps0) ** (1.0 / ref.gamma_he)
    mu_end = ref.mu0 + ref.ms0 - ms_end
    V_end = ref.Vu0 + ref.q_p_ref * ref.t_cut / ref.rho_p
    assert mu_end * ref.R_he * ref.Tu / V_end == pytest.approx(ref.Pset, rel=1e-12)


def test_missing_input_raises():
    inputs = dict(REFERENCE_CASE)
    del inputs["tau_v"]
    with pytest.raises(ValueError, match="tau_v"):
        build_params(inputs)


def test_unknown_or_ambiguous_input_raises():
    with pytest.raises(ValueError, match="unknown"):
        build_params({**REFERENCE_CASE, "Vtank": 1.328})
    with pytest.raises(ValueError, match="unknown"):
        build_params({**REFERENCE_CASE, "ms0": 5.75})  # ms0 is derived in sized mode


@pytest.mark.parametrize("key, value", [("Cd", float("nan")), ("tau_v", 0.0), ("mp0", -1.0),
                                        ("scenario", "open"), ("supply_mode", "auto"), ("N_engines", 7.5)])
def test_invalid_input_raises(key, value):
    with pytest.raises(ValueError):
        build_params({**REFERENCE_CASE, key: value})


def test_regulator_sized_at_actual_end_of_burn_ratio(ref):
    """C_reg delivers q_r_req at a_ref, Ts_ref, Ps_f and r_ref = Pset / Ps_f, for any Ps_f;
    at the baseline r_ref is exactly the literal 0.8, so REFERENCE_CASE is unchanged."""
    assert ref.Pset / ref.Ps_f == 0.8
    for Ps_f in (10.0e6, 12.0e6):
        p = build_params({**REFERENCE_CASE, "Ps_f": Ps_f})
        delivered = p.a_ref * p.C_reg * p.Ps_f / np.sqrt(p.Ts_ref) * flow_function(p.Pset / p.Ps_f, p.gamma_he)
        assert delivered == pytest.approx(p.q_r_req, rel=1e-14)


def test_specified_mass_mode_keeps_ms0_and_c_reg(ref):
    starved = build_params(GAS_STARVED_CASE)
    assert starved.ms0 == 5.75
    assert starved.Vs == pytest.approx(0.100, abs=5e-4)
    assert starved.C_reg == ref.C_reg
    assert starved.a_fix == ref.a_fix
    assert starved.m_dry == ref.m_dry


def test_a0_by_scenario(ref):
    """a0 = 0 regulated, a_fix fixed orifice."""
    fixed = build_params(FIXED_ORIFICE_CASE)
    assert ref.a0 == 0.0
    assert fixed.a0 == fixed.a_fix == ref.a_fix


def test_initial_state(ref):
    u0 = build_initial_state(ref)
    assert u0.dtype == np.float64 and u0.shape == (8,)
    np.testing.assert_array_equal(u0, [0.0, 19000.0, 536.0, 1.10, 2562.0, ref.ms0, 1.64, 0.0])


@pytest.mark.parametrize("n_cut", [3000, 6000, 12000, 15000])
def test_validate_accepts_integer_n_cut(ref, n_cut):
    report = validate(ref, build_initial_state(ref), {"dt": ref.t_cut / n_cut})
    assert report["n_cut"] == n_cut
    assert report["Pu0"] == pytest.approx(ref.Pu0, rel=1e-14)


def test_tcut_dt_mismatch_raises(ref):
    with pytest.raises(ValueError, match="integer"):
        validate(ref, build_initial_state(ref), {"dt": 7.0e-4})
    with pytest.raises(ValueError, match="dt"):
        validate(ref, build_initial_state(ref), {})


def test_validate_rejects_bad_state(ref):
    u0 = build_initial_state(ref)
    for idx, value in ((2, 0.0), (4, ref.Vtank * ref.rho_p), (7, 1.5), (5, np.inf)):
        bad = u0.copy()
        bad[idx] = value
        with pytest.raises(ValueError):
            validate(ref, bad, {"dt": 5e-4})


def test_reference_case_is_read_only():
    with pytest.raises(TypeError):
        REFERENCE_CASE["Cd"] = 0.8


def test_input_manifest(ref, tmp_path):
    path = tmp_path / "inputs.json"
    write_input_manifest(ref, str(path))
    data = json.loads(path.read_text())
    assert data["inputs"]["gamma0"]["label"] == "calibrated"
    assert data["inputs"]["rho_o"]["reference_temperature"] == "20 C"
    assert data["derived"]["ms0"]["value"] == ref.ms0
    assert "Vtank" in data["derived"] and "Vtank" not in data["inputs"]
