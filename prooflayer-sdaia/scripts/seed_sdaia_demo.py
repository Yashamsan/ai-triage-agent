#!/usr/bin/env python3
"""Demo: end-to-end SDAIA compliance workflow (illustrative scaffold).

Simulates a telecom billing agent that handles PII and acts autonomously:
risk classification -> decision logging -> incident reporting ->
safety report -> ethics label -> compliance dashboard.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from app.sdaia_storage import SDAIAStorage
from app.sdaia_risk import assess_and_store, RISK_CATEGORIES
from app import incidents as incidents_mod
from app.safety_reports import generate_safety_report, SAFETY_REPORT_ITEMS
from app.sdaia_labels import compute_ethics_label


def line(char="-", n=60):
    print(char * n)


def main():
    storage = SDAIAStorage(":memory:")

    print("ProofLayer SDAIA Compliance Demo (illustrative scaffold)")
    line("=")

    agent_id = "telecom_billing_agent"
    storage.register_agent(
        agent_id=agent_id,
        name="Telecom Billing Assistant",
        sector="telecom",
        description="Autonomous agent that resolves billing disputes and issues refunds.",
        handles_pii=True,
        autonomy_level="autonomous",
    )
    print(f"\nRegistered agent: {agent_id} (sector=telecom, handles_pii=True, autonomy=autonomous)")

    line()
    print("Step 1: Risk classification across 7 categories")
    line()
    factors = {
        "sector": "telecom",
        "handles_pii": True,
        "autonomy_level": "autonomous",
        "affected_population": "public",
    }
    risk_result = assess_and_store(storage, agent_id, factors)
    for cat in RISK_CATEGORIES:
        info = risk_result["categories"][cat]
        print(f"  {cat:15s} -> {info['level']:8s} ({info['rationale']})")
    print(f"\n  OVERALL RISK LEVEL: {risk_result['overall']}")

    line()
    print("Step 2: Simulating 12 decisions")
    line()
    decision_types = ["refund_issue", "dispute_resolution", "plan_change", "balance_adjustment"]
    for i in range(12):
        d_type = decision_types[i % len(decision_types)]
        requires_review = i % 4 == 0  # every 4th decision flagged for human review
        storage.log_decision(
            agent_id=agent_id,
            decision_type=d_type,
            input_summary=f"customer_request_{i+1}",
            output_summary=f"action_taken_{i+1}",
            risk_level=risk_result["overall"],
            requires_human_review=requires_review,
        )
    decisions = storage.list_decisions(agent_id)
    flagged = sum(1 for d in decisions if d["requires_human_review"])
    print(f"  Logged {len(decisions)} decisions ({flagged} flagged for human review)")

    line()
    print("Step 3: Incident reporting")
    line()
    medium_incident = incidents_mod.log_incident(
        storage,
        agent_id=agent_id,
        severity="MEDIUM",
        category="financial",
        description="Refund amount miscalculated due to currency rounding error.",
    )
    print(f"  Logged MEDIUM incident #{medium_incident['id']}")
    incidents_mod.resolve_incident(
        storage,
        medium_incident["id"],
        root_cause="Rounding applied before currency conversion instead of after.",
        corrective_action="Fixed calculation order; added regression test.",
    )
    print(f"  Resolved incident #{medium_incident['id']}")

    critical_incident = incidents_mod.log_incident(
        storage,
        agent_id=agent_id,
        severity="CRITICAL",
        category="legal",
        description="Agent issued a refund exceeding authorized autonomous limit without human review.",
    )
    print(f"  Logged CRITICAL incident #{critical_incident['id']}")
    if critical_incident["reported_to_regulator"]:
        print(f"    -> Auto-reported to regulator: {critical_incident['regulator_report_ref']}")
    incidents_mod.resolve_incident(
        storage,
        critical_incident["id"],
        root_cause="Autonomous decision threshold misconfigured for this customer tier.",
        corrective_action="Threshold corrected; added mandatory human review above tier limit.",
    )
    print(f"  Resolved incident #{critical_incident['id']}")

    all_incidents = storage.list_incidents(agent_id=agent_id)
    print(f"\n  Total incidents: {len(all_incidents)} "
          f"({sum(1 for i in all_incidents if i['severity']=='CRITICAL')} critical, "
          f"{sum(1 for i in all_incidents if i.get('resolved_at'))} resolved)")

    line()
    print("Step 4: Safety report (10 items)")
    line()
    report = generate_safety_report(storage, agent_id)
    for i, item in enumerate(SAFETY_REPORT_ITEMS, start=1):
        print(f"  {i:2d}. {item}")
    print(f"\n  Safety report generated with all {len(SAFETY_REPORT_ITEMS)} items.")

    line()
    print("Step 5: Ethics label")
    line()
    label = compute_ethics_label(storage, agent_id)
    print(f"  Score: {label['score']}/100")
    print(f"  Ethics label: {label['tier_name_ar']} ({label['tier_name_en']})")

    line()
    print("Compliance Dashboard")
    line("=")
    agent = storage.get_agent(agent_id)
    print(f"  Agent:            {agent['name']} ({agent_id})")
    print(f"  Sector:           {agent['sector']}")
    print(f"  Overall risk:     {risk_result['overall']}")
    print(f"  Decisions logged: {len(decisions)}")
    print(f"  Incidents:        {len(all_incidents)} total, "
          f"{sum(1 for i in all_incidents if i.get('resolved_at'))} resolved")
    print(f"  Ethics label:     {label['tier_name_ar']} ({label['tier_name_en']}) — tier {label['tier']}/5")
    line("=")


if __name__ == "__main__":
    main()
