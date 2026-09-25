"""Safety report generator for the SDAIA compliance module.

Produces the 10-item safety report required by the SDAIA "Responsible AI
Policy" §8.8.1-8.8.10 before publishing a high-risk system (or any of its
releases). Each item is populated with real computed evidence where this
schema actually tracks it (risk categories, incident history, human-review
flags); where the policy asks for something this schema has no signal for
(per-language test coverage, modality-specific adversarial results,
jailbreak/security testing, model-version diffs), the item is returned with
`requires_manual_attestation: True` and no fabricated content — a report
that silently claims a section is satisfied when nothing was actually
verified is worse than one that says so plainly, since someone reading the
dashboard would otherwise take "present" to mean "tested."
"""
from __future__ import annotations

import time

from app.sdaia_storage import to_epoch

# §8.8.1-8.8.10, in document order.
SAFETY_REPORT_ITEMS = [
    "safety_barriers_and_tools",                    # §8.8.1
    "multilingual_testing_coverage",                  # §8.8.2
    "misuse_and_adversarial_scenario_results",          # §8.8.3
    "legal_and_human_principles_violation_testing",       # §8.8.4
    "expert_informed_incident_scenarios",                  # §8.8.5
    "jailbreak_and_vulnerability_testing",                   # §8.8.6
    "pre_publish_model_modifications",                        # §8.8.7
    "expected_risk_types_and_mitigation",                       # §8.8.8
    "post_deployment_continuous_protection",                     # §8.8.9
    "periodic_report_updates",                                    # §8.8.10
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
    prior_reports = storage.get_latest_safety_report(agent_id)

    if period_days is not None:
        cutoff_epoch = time.time() - (period_days * 86400)
        decisions = [d for d in decisions if to_epoch(d["created_at"]) >= cutoff_epoch]
        incidents = [i for i in incidents if to_epoch(i["detected_at"]) >= cutoff_epoch]

    resolved = [i for i in incidents if i.get("resolved_at")]
    human_reviewed_decisions = [d for d in decisions if d.get("requires_human_review")]
    legal_incidents = [i for i in incidents if i["category"] == "legal"]

    report = {
        "safety_barriers_and_tools": {
            "autonomy_level": agent["autonomy_level"],
            "decisions_with_human_review": len(human_reviewed_decisions),
            "total_decisions": len(decisions),
            "requires_manual_attestation": True,
            "attestation_note": (
                "Human-in-the-loop signal above is computed from decision logs; "
                "the actual barrier/tool inventory (filters, model-alignment "
                "techniques, refusal policies, risk monitors) is not tracked "
                "in this schema and must be attached separately."
            ),
        },
        "multilingual_testing_coverage": {
            "requires_manual_attestation": True,
            "attestation_note": "Per-language test coverage is not tracked in this schema.",
        },
        "misuse_and_adversarial_scenario_results": {
            "incidents_on_record": len(incidents),
            "requires_manual_attestation": True,
            "attestation_note": (
                "Real incidents are listed as partial evidence but are not a "
                "substitute for controlled misuse/adversarial testing across "
                "text, code, image, audio, and video modalities."
            ),
        },
        "legal_and_human_principles_violation_testing": {
            "legal_category_incidents": len(legal_incidents),
            "requires_manual_attestation": True,
            "attestation_note": (
                "Legal-category incident count is partial evidence only; "
                "structured legal/human-principles violation test results "
                "are not tracked in this schema."
            ),
        },
        "expert_informed_incident_scenarios": {
            "incidents_with_documented_root_cause": len(
                [i for i in incidents if i.get("root_cause")]
            ),
            "requires_manual_attestation": True,
            "attestation_note": (
                "Resolved incidents with a root cause are listed as partial "
                "evidence; expert-constructed realistic incident scenarios "
                "(legal, health, social, security specialists) are not "
                "tracked in this schema."
            ),
        },
        "jailbreak_and_vulnerability_testing": {
            "requires_manual_attestation": True,
            "attestation_note": "Jailbreak/security-vulnerability test results are not tracked in this schema.",
        },
        "pre_publish_model_modifications": {
            "requires_manual_attestation": True,
            "attestation_note": "Model version/modification history is not tracked in this schema.",
        },
        "expected_risk_types_and_mitigation": {
            "risk_categories": [
                {"category": r["category"], "level": r["level"], "rationale": r["rationale"]}
                for r in risk_assessments
            ],
            "mitigation_actions_on_record": [
                i["corrective_action"] for i in resolved if i.get("corrective_action")
            ],
        },
        "post_deployment_continuous_protection": {
            "open_incidents": len([i for i in incidents if not i.get("resolved_at")]),
            "requires_manual_attestation": True,
            "attestation_note": (
                "Open-incident count is partial evidence; real-time monitoring "
                "configuration for inputs/outputs during operation is not "
                "tracked in this schema."
            ),
        },
        "periodic_report_updates": {
            "report_period_days": period_days,
            "prior_report_on_file": prior_reports is not None,
            "prior_report_generated_at": prior_reports["generated_at"] if prior_reports else None,
        },
    }

    storage.save_safety_report(agent_id, report)
    return report
