"""Tests for app/rmf_core.py -- the SDAIA-P145 risk-cycle business logic.

Ported from prooflayer-sdaia/P145/prooflayer-rmf/tests/test_rmf.py's matrix
and validation tests (9/9 passing on the original SQLite prototype). The
DB-backed tests from that suite (context/risk/review persistence) aren't
ported here since app/rmf_api.py's storage layer -- like every other
Postgres-backed app/*_api.py module in this repo -- has no direct unit-test
coverage; this file covers the pure, DB-free logic that both the prototype
and app/rmf_api.py share.
"""
import pytest

from app import rmf_core


def test_matrix_bands():
    assert rmf_core.assess(1, 1).band == "low"          # 1
    assert rmf_core.assess(2, 1).band == "low"           # 2
    assert rmf_core.assess(2, 2).band == "medium"        # 4
    assert rmf_core.assess(3, 2).band == "medium"        # 6
    assert rmf_core.assess(3, 3).band == "high"          # 9
    assert rmf_core.assess(4, 3).band == "high"          # 12
    assert rmf_core.assess(4, 4).band == "catastrophic"  # 16


def test_matrix_rejects_out_of_range():
    with pytest.raises(ValueError):
        rmf_core.assess(0, 2)
    with pytest.raises(ValueError):
        rmf_core.assess(2, 5)


def test_band_for_rejects_impossible_levels():
    # 7, 13, 14, 15 are not valid products of 1-4 x 1-4.
    for level in (7, 13, 14, 15):
        with pytest.raises(ValueError):
            rmf_core.band_for(level)


def test_seven_category_taxonomy():
    assert len(rmf_core.TAXONOMY) == 7
    rmf_core.validate_category("privacy_security")  # does not raise
    with pytest.raises(ValueError):
        rmf_core.validate_category("not_a_category")


def test_treatment_requires_rationale_and_approver():
    with pytest.raises(ValueError):
        rmf_core.validate_treatment("accept", rationale="ok", approver="")
    with pytest.raises(ValueError):
        rmf_core.validate_treatment("accept", rationale="", approver="lead")
    with pytest.raises(ValueError):
        rmf_core.validate_treatment("ignore", rationale="x", approver="lead")
    rmf_core.validate_treatment("accept", rationale="residual within tolerance", approver="lead")  # does not raise


def test_treatment_strategy_enum():
    assert rmf_core.TREATMENT_STRATEGIES == ["avoid", "mitigate", "transfer", "accept"]
