#!/usr/bin/env python3
"""Test suite for the SDAIA compliance module scaffold.

Plain stdlib test runner (no pytest dependency) — each test_* function
raises AssertionError on failure. Run directly: python3 tests/test_sdaia_compliance.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.sdaia_storage import SDAIAStorage
from app.sdaia_risk import classify_agent, assess_and_store, assess_agent_risk, RISK_CATEGORIES
from app import incidents as incidents_mod
from app.safety_reports import generate_safety_report, SAFETY_REPORT_ITEMS
from app.sdaia_labels import compute_ethics_label, TIERS


def _new_storage() -> SDAIAStorage:
    return SDAIAStorage(":memory:")


def test_storage_init():
    storage = _new_storage()
    cur = storage.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = {row[0] for row in cur.fetchall()}
    expected = {
        "agents", "risk_assessments", "decisions",
        "incidents", "safety_reports", "ethics_labels",
    }
    assert expected.issubset(tables), f"missing tables: {expected - tables}"


def test_register_agent():
    storage = _new_storage()
    agent = storage.register_agent(
        "agent_1", "Test Agent", "retail", handles_pii=False, autonomy_level="assisted"
    )
    assert agent["agent_id"] == "agent_1"
    fetched = storage.get_agent("agent_1")
    assert fetched is not None
    assert fetched["name"] == "Test Agent"


def test_get_agent_not_found():
    storage = _new_storage()
    assert storage.get_agent("does_not_exist") is None


def test_risk_classify_low():
    result = classify_agent(
        {"sector": "retail", "handles_pii": False, "autonomy_level": "assisted",
         "affected_population": "individual"}
    )
    assert result["overall"] == "LOW", result["overall"]


def test_risk_classify_critical_telecom_pii():
    result = classify_agent(
        {"sector": "telecom", "handles_pii": True, "autonomy_level": "autonomous",
         "affected_population": "public"}
    )
    assert result["overall"] == "CRITICAL", result["overall"]


def test_risk_overall_is_max_of_categories():
    result = classify_agent(
        {"sector": "telecom", "handles_pii": True, "autonomy_level": "autonomous",
         "affected_population": "public"}
    )
    levels = [c["level"] for c in result["categories"].values()]
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    assert result["overall"] == max(levels, key=lambda l: rank[l])


def test_seven_categories_assessed():
    result = classify_agent(
        {"sector": "retail", "handles_pii": False, "autonomy_level": "assisted"}
    )
    assert set(result["categories"].keys()) == set(RISK_CATEGORIES)
    assert len(RISK_CATEGORIES) == 7


def test_log_decision():
    storage = _new_storage()
    storage.register_agent("agent_2", "Agent Two", "finance")
    storage.log_decision(
        "agent_2", "loan_approval", "input", "output", "HIGH", requires_human_review=True
    )
    decisions = storage.list_decisions("agent_2")
    assert len(decisions) == 1
    assert decisions[0]["requires_human_review"] == 1


def test_log_incident_medium():
    storage = _new_storage()
    storage.register_agent("agent_3", "Agent Three", "retail")
    incident = incidents_mod.log_incident(
        storage, "agent_3", "MEDIUM", "financial", "Minor calculation error"
    )
    assert incident["severity"] == "MEDIUM"
    assert incident["reported_to_regulator"] == 0


def test_log_incident_critical_auto_reports():
    storage = _new_storage()
    storage.register_agent("agent_4", "Agent Four", "telecom")
    incident = incidents_mod.log_incident(
        storage, "agent_4", "CRITICAL", "legal", "Unauthorized autonomous action"
    )
    assert incident["reported_to_regulator"] == 1
    assert incident["regulator_report_ref"]
    assert incident["regulator_report_ref"].startswith("SDAIA-RPT-")


def test_resolve_incident():
    storage = _new_storage()
    storage.register_agent("agent_5", "Agent Five", "retail")
    incident = incidents_mod.log_incident(storage, "agent_5", "LOW", "safety", "Minor glitch")
    resolved = incidents_mod.resolve_incident(
        storage, incident["id"], "Config drift", "Reverted config"
    )
    assert resolved["resolved_at"] is not None
    assert resolved["root_cause"] == "Config drift"


def test_list_incidents_by_agent():
    storage = _new_storage()
    storage.register_agent("agent_6", "Agent Six", "retail")
    storage.register_agent("agent_7", "Agent Seven", "retail")
    incidents_mod.log_incident(storage, "agent_6", "LOW", "safety", "Issue A")
    incidents_mod.log_incident(storage, "agent_7", "LOW", "safety", "Issue B")
    agent_6_incidents = storage.list_incidents(agent_id="agent_6")
    assert len(agent_6_incidents) == 1
    assert agent_6_incidents[0]["description"] == "Issue A"


def test_generate_safety_report_has_10_items():
    storage = _new_storage()
    storage.register_agent("agent_8", "Agent Eight", "telecom", handles_pii=True,
                            autonomy_level="autonomous")
    assess_and_store(storage, "agent_8", {
        "sector": "telecom", "handles_pii": True, "autonomy_level": "autonomous",
        "affected_population": "public",
    })
    report = generate_safety_report(storage, "agent_8")
    assert len(SAFETY_REPORT_ITEMS) == 10
    assert set(report.keys()) == set(SAFETY_REPORT_ITEMS)


def test_ethics_label_tiers_ordered():
    assert len(TIERS) == 5
    tier_numbers = [t[0] for t in TIERS]
    assert tier_numbers == sorted(tier_numbers)


def test_compute_ethics_label_committed():
    storage = _new_storage()
    agent_id = "agent_9"
    storage.register_agent(agent_id, "Agent Nine", "telecom", handles_pii=True,
                            autonomy_level="autonomous")
    assess_and_store(storage, agent_id, {
        "sector": "telecom", "handles_pii": True, "autonomy_level": "autonomous",
        "affected_population": "public",
    })
    storage.log_decision(agent_id, "refund", "in", "out", "CRITICAL", requires_human_review=True)
    incident = incidents_mod.log_incident(
        storage, agent_id, "CRITICAL", "legal", "Threshold exceeded"
    )
    incidents_mod.resolve_incident(
        storage, incident["id"], "Misconfigured threshold", "Threshold corrected"
    )
    generate_safety_report(storage, agent_id)
    label = compute_ethics_label(storage, agent_id)
    assert label["tier_name_en"] == "Committed", label
    assert label["tier_name_ar"] == "ملتزم"
    assert 75 <= label["score"] <= 89


def test_end_to_end_demo_flow():
    storage = _new_storage()
    agent_id = "agent_10"
    storage.register_agent(agent_id, "Agent Ten", "telecom", handles_pii=True,
                            autonomy_level="autonomous")
    risk = assess_and_store(storage, agent_id, {
        "sector": "telecom", "handles_pii": True, "autonomy_level": "autonomous",
        "affected_population": "public",
    })
    assert risk["overall"] == "CRITICAL"

    for i in range(12):
        storage.log_decision(agent_id, "decision", f"in_{i}", f"out_{i}", risk["overall"],
                              requires_human_review=(i % 4 == 0))
    assert len(storage.list_decisions(agent_id)) == 12

    med = incidents_mod.log_incident(storage, agent_id, "MEDIUM", "financial", "Rounding error")
    incidents_mod.resolve_incident(storage, med["id"], "Rounding order", "Fixed order")
    crit = incidents_mod.log_incident(storage, agent_id, "CRITICAL", "legal", "Limit exceeded")
    incidents_mod.resolve_incident(storage, crit["id"], "Bad config", "Config fixed")
    assert len(storage.list_incidents(agent_id=agent_id)) == 2
    assert crit["reported_to_regulator"] == 1

    report = generate_safety_report(storage, agent_id)
    assert len(report) == 10

    label = compute_ethics_label(storage, agent_id)
    assert label["tier"] >= 1


def test_assess_agent_risk_wrapper():
    storage = _new_storage()
    storage.register_agent("agent_11", "Agent Eleven", "telecom", handles_pii=True,
                            autonomy_level="autonomous")
    result = assess_agent_risk(
        store=storage,
        agent_id="agent_11",
        sector="telecom",
        has_pii=True,
        is_autonomous=True,
        has_human_oversight=False,
    )
    assert result["overall"] == "CRITICAL", result


def test_assess_agent_risk_oversight_deescalates():
    storage = _new_storage()
    storage.register_agent("agent_12", "Agent Twelve", "telecom", handles_pii=True,
                            autonomy_level="autonomous")
    without_oversight = assess_agent_risk(
        storage, "agent_12", sector="telecom", has_pii=True,
        is_autonomous=True, has_human_oversight=False,
    )
    with_oversight = assess_agent_risk(
        storage, "agent_12", sector="telecom", has_pii=True,
        is_autonomous=True, has_human_oversight=True,
    )
    rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
    assert rank[with_oversight["overall"]] < rank[without_oversight["overall"]]


def test_register_agent_with_pl_agent_id_link():
    storage = _new_storage()
    linked_uuid = "11111111-2222-3333-4444-555555555555"
    agent = storage.register_agent(
        "agent_13", "Agent Thirteen", "telecom", pl_agent_id=linked_uuid
    )
    assert agent["pl_agent_id"] == linked_uuid


def test_safety_report_period_days_filters_old_records():
    storage = _new_storage()
    agent_id = "agent_14"
    storage.register_agent(agent_id, "Agent Fourteen", "telecom", handles_pii=True,
                            autonomy_level="autonomous")
    assess_and_store(storage, agent_id, {
        "sector": "telecom", "handles_pii": True, "autonomy_level": "autonomous",
        "affected_population": "public",
    })

    recent = incidents_mod.log_incident(storage, agent_id, "MEDIUM", "financial", "Recent issue")
    stale = incidents_mod.log_incident(storage, agent_id, "MEDIUM", "financial", "Old issue")
    # Backdate the "stale" incident's detected_at well outside a 90-day window.
    storage.conn.execute(
        "UPDATE incidents SET detected_at = ? WHERE id = ?",
        ("2020-01-01T00:00:00Z", stale["id"]),
    )
    storage.conn.commit()

    full_report = generate_safety_report(storage, agent_id)
    assert full_report["incident_history_and_remediation"]["total_incidents"] == 2

    windowed_report = generate_safety_report(storage, agent_id, period_days=90)
    assert windowed_report["incident_history_and_remediation"]["total_incidents"] == 1
    assert windowed_report["system_description"]["report_period_days"] == 90


TESTS = [
    (name, fn) for name, fn in sorted(globals().items())
    if name.startswith("test_") and callable(fn)
]


def main():
    passed = 0
    failed = 0
    for name, fn in TESTS:
        try:
            fn()
            print(f"✅ {name}")
            passed += 1
        except Exception as e:
            print(f"❌ {name}: {e}")
            failed += 1

    total = passed + failed
    print(f"\nResults: {passed} passed, {failed} failed, {total} total")
    if failed == 0:
        print("\U0001F389 All tests pass!")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
