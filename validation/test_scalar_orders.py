"""M3/M5: solver order on exact scalar problems (verification_studies.md Section 10)."""

import pytest

EXPECTED_ORDER = {"fe": 1, "rk4": 4, "ab3": 3, "be": 1}


@pytest.mark.parametrize("method", list(EXPECTED_ORDER))
def test_decay_order(method):
    """u' = -u, u(0) = 1, exact exp(-t)."""
    pytest.skip("M3/M5: not implemented")


@pytest.mark.parametrize("method", list(EXPECTED_ORDER))
def test_time_dependent_order(method):
    """u' = t - u, u(0) = 1, exact t - 1 + 2 exp(-t); catches wrong stage times or argument order."""
    pytest.skip("M3/M5: not implemented")


def test_rk4_reuses_supplied_k1():
    pytest.skip("M3: not implemented")
