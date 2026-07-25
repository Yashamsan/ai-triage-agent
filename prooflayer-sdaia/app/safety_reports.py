"""Safety report generator for the SDAIA compliance module (illustrative scaffold).

Produces a 10-item safety report per agent, assembled from stored risk
assessments, decisions, and incidents. The 10-item structure is modeled
loosely on common AI-safety-report checklists (system description, risk
rationale, data governance, fairness testing, human oversight, incident
history, performance metrics, explainability, security testing, and a
compliance attestation) — it is not a verified transcription of a
specific SDAIA document section.
"""
from __future__ import annotations

import time

from app.sdaia_storage import to_epoch

SAFETY_REPORT_ITEMS = [
    "system_description",
    "risk_classification_rationale",
    "data_governance_and_privacy",
    "bias_and_fairness_testing",
    "human_oversight_mechanisms",
    "incident_history_and_remediation",
    "performance_and_accuracy_metrics",
    "explainability_and_transparency",
    "security_and_robustness_testing",
    "compliance_attestation",
]


def generate_safety_report(storage, agent_id: str, period_days: int | None = None) -> dict:
    """Generate a safety report. If period_days is given (e.g. 90 for a
    quarterly cadence), decisions and incidents are restricted to that
    trailing window; otherwise the report covers the agent's full history."""
    agent = storage.get_agent(agent_id)
    if not agent:
        raise ValueError(f"unknown agent: {agent_id}")

    risk_assessments = storage.get_risk_assessments(agent_id)
    decisions = storage.list_decisions(agent_id)
    incidents = storage.list_incidents(agent_id=agent_id)

    cutoff_epoch = None
    if period_days is not None:
        cutoff_epoch = time.time() - (period_days * 86400)
        decisions = [d for d in decisions if to_epoch(d["created_at"]) >= cutoff_epoch]
        incidents = [i for i in incidents if to_epoch(i["detected_at"]) >= cutoff_epoch]

    overall_level = "LOW"
    if risk_assessments:
        rank = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}
        overall_level = max((r["level"] for r in risk_assessments), key=lambda l: rank[l])

    resolved = [i for i in incidents if i.get("resolved_at")]
    unresolved = [i for i in incidents if not i.get("resolved_at")]
    critical_reported = [
        i for i in incidents if i["severity"] == "CRITICAL" and i.get("reported_to_regulator")
    ]

    human_reviewed_decisions = [d for d in decisions if d.get("requires_human_review")]

    report = {
        "system_description": {
            "agent_id": agent_id,
            "name": agent["name"],
            "sector": agent["sector"],
            "description": agent.get("description") or "",
            "autonomy_level": agent["autonomy_level"],
            "handles_pii": bool(agent["handles_pii"]),
            "report_period_days": period_days,
        },
        "risk_classification_rationale": {
            "overall_level": overall_level,
            "categories": [
                {"category": r["category"], "level": r["level"], "rationale": r["rationale"]}
                for r in risk_assessments
            ],
        },
        "data_governance_and_privacy": {
            "handles_pii": bool(agent["handles_pii"]),
            "note": "PII handling requires access controls and retention limits."
            if agent["handles_pii"]
            else "No personal data processed by this agent.",
        },
        "bias_and_fairness_testing": {
            "decisions_sampled": len(decisions),
            "note": "Fairness testing should be run against decision logs per release cycle.",
        },
        "human_oversight_mechanisms": {
            "decisions_flagged_for_review": len(human_reviewed_decisions),
            "total_decisions": len(decisions),
        },
        "incident_history_and_remediation": {
            "total_incidents": len(incidents),
            "resolved": len(resolved),
            "unresolved": len(unresolved),
            "critical_reported_to_regulator": len(critical_reported),
        },
        "performance_and_accuracy_metrics": {
            "total_decisions_logged": len(decisions),
            "note": "Attach accuracy/precision/recall metrics from evaluation harness here.",
        },
        "explainability_and_transparency": {
            "note": "Each decision record includes an input/output summary for auditability.",
        },
        "security_and_robustness_testing": {
            "note": "Attach adversarial/robustness test results here.",
        },
        "compliance_attestation": {
            "all_items_present": True,
            "unresolved_critical_incidents": len(
                [i for i in unresolved if i["severity"] == "CRITICAL"]
            ),
        },
    }

    storage.save_safety_report(agent_id, report)
    return report
