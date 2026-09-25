"""Stage-by-stage tests for the SDAIA-P145 RMF module.

Run:  pytest -q
Uses a temp DB so it never touches the real rmf_data.db.
"""
import os
import tempfile

# Point the module at a temp DB BEFORE importing app modules.
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
os.environ["RMF_DB_PATH"] = _tmp.name

import pytest  # noqa: E402

from app import sdaia_rmf, storage  # noqa: E402

storage.init_db()


@pytest.fixture(autouse=True)
def clean_state():
    yield
    # leave DB as-is between tests; each test uses fresh risk ids


def test_context_stage1():
    r = sdaia_rmf.register_context("a1", description="triage", human_role="approval")
    assert r["stage"] == 1
    ctx = storage.get_context("a1")
    assert ctx["description"] == "triage"
    # upsert same agent
    sdaia_rmf.register_context("a1", description="updated")
    assert storage.get_context("a1")["description"] == "updated"


def test_risk_stage2():
    r = sdaia_rmf.register_risk("a1", "privacy_security", "PII leak")
    assert r["stage"] == 2 and r["status"] == "open"
    with pytest.raises(ValueError):
        sdaia_rmf.register_risk("a1", "not_a_category", "x")


def test_matrix_bands():
    assert sdaia_rmf.assess(1, 1).band == "low"        # 1
    assert sdaia_rmf.assess(2, 1).band == "low"        # 2
    assert sdaia_rmf.assess(2, 2).band == "medium"     # 4
    assert sdaia_rmf.assess(3, 2).band == "medium"     # 6
    assert sdaia_rmf.assess(3, 3).band == "high"       # 9
    assert sdaia_rmf.assess(4, 3).band == "high"       # 12
    assert sdaia_rmf.assess(4, 4).band == "catastrophic"  # 16
    with pytest.raises(ValueError):
        sdaia_rmf.assess(0, 2)
    with pytest.raises(ValueError):
        sdaia_rmf.assess(2, 5)


def test_assessment_stage3():
    r = sdaia_rmf.register_risk("a1", "malicious_use", "prompt injection")
    out = sdaia_rmf.assess_risk(r["risk_id"], 2, 4, evidence="red team", assessed_by="tester")
    assert out["stage"] == 3 and out["risk_level"] == 8 and out["band"] == "high"
    entry = sdaia_rmf.risk_register(agent_id="a1")[0]
    assert entry["assessment"]["band"] == "high"


def test_treatment_stage4_requires_approver():
    r = sdaia_rmf.register_risk("a1", "misinformation", "hallucinated SLA")
    with pytest.raises(ValueError):
        sdaia_rmf.treat_risk(r["risk_id"], "accept", rationale="ok", approver="")
    with pytest.raises(ValueError):
        sdaia_rmf.treat_risk(r["risk_id"], "accept", rationale="", approver="lead")
    with pytest.raises(ValueError):
        sdaia_rmf.treat_risk(r["risk_id"], "ignore", rationale="x", approver="lead")


def test_documented_acceptance_stage4():
    r = sdaia_rmf.register_risk("a1", "safety_limitations", "misroute risk")
    out = sdaia_rmf.treat_risk(
        r["risk_id"], "accept",
        rationale="residual within tolerance, controls in place",
        approver="ai-governance-lead",
        approval_mechanism="system sign-off workflow",
        residual_likelihood=1, residual_impact=2,
    )
    assert out["strategy"] == "accept" and out["residual_level"] == 2
    entry = sdaia_rmf.risk_register(agent_id="a1")
    treatment = next(x["treatment"] for x in entry if x["risk_id"] == r["risk_id"])
    assert treatment["approver"] == "ai-governance-lead"
    assert treatment["approval_mechanism"] == "system sign-off workflow"


def test_review_stage5():
    sdaia_rmf.add_review("a1", "incident", findings="drift", incident_rca="data shift")
    rows = storage.list_reviews("a1")
    assert len(rows) == 1 and rows[0]["incident_rca"] == "data shift"


def test_five_stage_report():
    # fresh agent
    aid = "report-agent"
    sdaia_rmf.register_context(aid, description="demo")
    r1 = sdaia_rmf.register_risk(aid, "privacy_security", "pii")
    sdaia_rmf.assess_risk(r1["risk_id"], 2, 3)
    sdaia_rmf.treat_risk(r1["risk_id"], "mitigate", rationale="controls", approver="lead")
    sdaia_rmf.add_review(aid, "periodic", findings="ok")
    rep = sdaia_rmf.five_stage_report(aid)
    assert rep["stages_evidenced"] == 5
    assert rep["stage_1_context_and_scope"]["description"] == "demo"
    assert len(rep["stage_2_risk_identification"]) == 1
    assert len(rep["stage_4_risk_treatment"]) == 1


def test_fleet_matrix():
    sdaia_rmf.register_risk("a2", "privacy_security", "x")
    r = sdaia_rmf.register_risk("a2", "malicious_use", "y")
    sdaia_rmf.assess_risk(r["risk_id"], 4, 4)
    m = sdaia_rmf.fleet_matrix()
    assert m["catastrophic"] >= 1
