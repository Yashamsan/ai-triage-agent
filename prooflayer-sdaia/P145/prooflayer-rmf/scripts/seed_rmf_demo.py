"""Seed the RMF module with a realistic 9-agent fleet.

Usage:  python -m scripts.seed_rmf_demo
Run from the prooflayer-rmf/ directory so the default SQLite path resolves.

Each agent gets: a context (Stage 1), 2-4 risks across the taxonomy
(Stage 2), assessments on the 4x4 matrix (Stage 3), treatment decisions
including at least one documented "accept" (Stage 4), and a review (Stage 5).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import sdaia_rmf, storage  # noqa: E402

AGENTS = [
    {"agent_id": "triage-agent-en", "description": "EN triage & routing", "human_role": "escalation approval"},
    {"agent_id": "triage-agent-ar", "description": "AR triage & routing (Gulf Arabic)", "human_role": "escalation approval"},
    {"agent_id": "billing-agent", "description": "Invoice & payment queries", "human_role": "refund approval"},
    {"agent_id": "sip-support-agent", "description": "SIP/data-circuit technical support", "human_role": "dispatch authorization"},
    {"agent_id": "fraud-screening-agent", "description": "Fraud flagging (ticket + end call)", "human_role": "case review"},
    {"agent_id": "portal-help-agent", "description": "Business portal navigation", "human_role": "none"},
    {"agent_id": "retention-agent", "description": "Contract renewal & retention offers", "human_role": "offer approval"},
    {"agent_id": "cloud-sales-agent", "description": "Cloud/DC solution inquiries", "human_role": "pricing approval"},
    {"agent_id": "voice-ivr-agent", "description": "IVR front-end (Clarity)", "human_role": "call-back routing"},
]

RISKS = [
    # (category, description, source, intent, timing, likelihood, impact)
    ("privacy_security", "Customer PII exposed in AR transcript logs", "third-party", "accidental", "operation", 3, 3),
    ("bias_discrimination_abuse", "Routing bias against Saudi-dialect queries", "internal", "accidental", "development", 2, 2),
    ("malicious_use", "Prompt injection via ticket text fields", "external", "malicious", "operation", 2, 4),
    ("misinformation", "Hallucinated SLA entitlement in chat replies", "internal", "accidental", "operation", 3, 3),
    ("human_machine_interaction", "Customer cannot reach human when agent loops", "internal", "accidental", "deployment", 3, 2),
    ("social_economic_environmental", "Retention offers misapplied to enterprise contracts", "internal", "accidental", "operation", 2, 3),
    ("safety_limitations", "Voice IVR misroutes emergency/outage calls", "internal", "accidental", "deployment", 1, 4),
    ("privacy_security", "Vendor MCP tool returns PHI in tool output", "third-party", "accidental", "operation", 2, 3),
]


def main() -> None:
    storage.init_db()
    for agent in AGENTS:
        aid = agent["agent_id"]
        print(f"\n=== {aid} ===")
        sdaia_rmf.register_context(
            aid,
            description=agent["description"],
            automation_level="high",
            human_role=agent["human_role"],
            lifecycle_stage="operation",
            change_plan="quarterly model update",
        )
        for cat, desc, src, intent, timing, lik, imp in RISKS:
            r = sdaia_rmf.register_risk(
                aid, cat, desc, source=src, intent=intent, timing=timing,
                identified_by="seed_rmf_demo",
            )
            sdaia_rmf.assess_risk(
                r["risk_id"], lik, imp,
                evidence="risk register baseline (seeded)",
                assessed_by="seed_rmf_demo",
            )
            # First risk per agent -> documented acceptance (Stage 4 showcase)
            strategy = "accept" if (cat, desc) == RISKS[0] else "mitigate"
            sdaia_rmf.treat_risk(
                r["risk_id"], strategy,
                rationale=f"Residual risk within approved tolerance for {cat}; "
                          f"controls in place; documented per SDAIA-P145 stage 4.",
                controls="monitoring + human approval gate",
                residual_likelihood=1, residual_impact=imp,
                approver="ai-governance-lead",
                approval_mechanism="system sign-off workflow",
            )
        sdaia_rmf.add_review(
            aid, review_type="periodic",
            findings="No drift observed; 1 minor routing anomaly under review",
            reviewed_by="seed_rmf_demo",
        )
    print("\n✅ Seeded 9 agents. Matrix:", sdaia_rmf.fleet_matrix())


if __name__ == "__main__":
    main()
