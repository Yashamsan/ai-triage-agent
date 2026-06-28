"""Multi-agent orchestration demo — 7 specialist agents.

Demonstrates:
  - 7 specialist agents (technical, billing, complaints, sales, loyalty, retention, fraud)
  - RouterAgent two-stage dispatch (Tier 1 / Tier 2 / Tier 3)
  - Cross-agent session memory (retention sees billing dispute history)
  - Fraud risk scoring with cross-agent payment failure amplification
  - ProofLayer governance recording (single CISO dashboard)

Run:
    docker compose up -d
    python scripts/demo_multi_agent.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import (
    BillingAgent,
    ComplaintsAgent,
    FraudDetectionAgent,
    LoyaltyAgent,
    MemoryBridge,
    ProofLayerGateway,
    RetentionAgent,
    RouterAgent,
    SalesAgent,
    TechnicalAgent,
    load_config,
)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _hr(label: str = "") -> None:
    width = 72
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * (width - pad - len(label) - 2)}")
    else:
        print("─" * width)


def _print_result(idx: str, msg: str, result: dict) -> None:
    tier_label = {
        1: "Tier 1 — Direct",
        2: "Tier 2 — Reflect",
        3: "Tier 3 — Escalate",
    }.get(result.get("tier", 0), "n/a")

    print(f"\n[{idx}] {msg[:65]!r}")
    print(f"     Intent      : {result.get('intent') or 'n/a'}")
    print(f"     Confidence  : {result.get('confidence', 0.0):.0%}")
    print(f"     Routing     : {tier_label}  →  {result.get('target_agent') or 'escalation'}")
    if result.get("reflection_notes"):
        print(f"     Reflection  : {result['reflection_notes'][:70]}")
    agent_reply = (result.get("response") or "").strip()[:130]
    print(f"     Response    : {agent_reply}")
    route_id = (result.get("route_decision_id") or "n/a")
    agent_id = (result.get("agent_decision_id") or "n/a")
    print(f"     ProofLayer  : route={route_id[:8]}  agent={agent_id[:8]}")


# ── Demo ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _hr("ProofLayer v3  —  Multi-Agent Orchestration Demo")
    print("  7 specialist agents  |  2-stage routing  |  cross-agent governance")

    # ── Bootstrap ──────────────────────────────────────────────────────────
    cfg     = load_config()
    gateway = ProofLayerGateway(cfg)
    memory  = MemoryBridge()
    router  = RouterAgent(cfg, gateway=gateway, memory=memory)

    agents = [
        TechnicalAgent(cfg, gateway),
        BillingAgent(cfg, gateway),
        ComplaintsAgent(cfg, gateway),
        SalesAgent(cfg, gateway),
        LoyaltyAgent(cfg, gateway),
        RetentionAgent(cfg, gateway),
        FraudDetectionAgent(cfg, gateway),
    ]
    router.load_agents(agents)
    print(f"\n  {len(agents)} specialist agents loaded | {len(router.routing_table)} routing rules")

    # ── Request 1: Technical — P1 outage ──────────────────────────────────
    _hr("Routing 8 requests across 7 specialist agents")

    r1 = router.route_sync(
        "The production platform has been completely down for 3 hours. This is a P1 emergency.",
        session_id="demo-session-001",
        customer_id="CUST-5501",
    )
    _print_result("1", "Production platform P1 outage (3 hours)", r1)
    time.sleep(0.2)

    # ── Request 2: Billing — duplicate charge ──────────────────────────────

    r2 = router.route_sync(
        "I was charged twice for my November invoice — SAR 149 appears twice on my statement.",
        session_id="demo-session-002",
        customer_id="CUST-7823",
    )
    _print_result("2", "Double charge SAR 149 — November invoice", r2)
    time.sleep(0.2)

    # ── Request 3: Fraud — international wire blocked ──────────────────────

    r3 = router.route_sync(
        "Why was my transaction TX-88241 for SAR 12,450 blocked? "
        "This was a property deposit — international wire to a UAE developer.",
        session_id="demo-session-003",
        customer_id="CUST-3310",
    )
    _print_result("3", "Blocked international wire SAR 12,450 (property deposit)", r3)
    time.sleep(0.2)

    # ── Request 4: Sales — promo inquiry ──────────────────────────────────

    r4 = router.route_sync(
        "I heard there's a promo code SAVE20 — can I apply it to my Professional plan?",
        session_id="demo-session-004",
        customer_id="CUST-6612",
    )
    _print_result("4", "Promo code SAVE20 inquiry (Professional plan)", r4)
    time.sleep(0.2)

    # ── Request 5: Loyalty — points balance ───────────────────────────────

    r5 = router.route_sync(
        "How many loyalty points do I have and when do they expire?",
        session_id="demo-session-005",
        customer_id="CUST-2290",
    )
    _print_result("5", "Loyalty points balance + expiry inquiry", r5)
    time.sleep(0.2)

    # ── Request 6: Complaints — regulatory ────────────────────────────────

    r6 = router.route_sync(
        "I am escalating this to CITC as a formal regulatory complaint. "
        "Your service has been unacceptable for three months.",
        session_id="demo-session-006",
        customer_id="CUST-1105",
    )
    _print_result("6", "CITC regulatory complaint (3 months unacceptable service)", r6)
    time.sleep(0.2)

    # ── Request 7+8: Cross-agent demo — same session ──────────────────────
    _hr("Cross-agent authority demo (session demo-cs-001)")

    # 7a: BillingAgent handles payment failure first
    print("\n[7a] First message: payment failure (billing-agent)")
    r7a = router.route_sync(
        "My payment keeps failing — this is the third time this month. "
        "I was double-charged last week too.",
        session_id="demo-cs-001",
        customer_id="CUST-4421",
    )
    _print_result("7a", "Recurring payment failures + double charge", r7a)
    time.sleep(0.2)

    # 7b: RetentionAgent sees BillingAgent context → amplifies churn model
    print("\n[7b] Second message: cancellation (retention-agent sees billing history)")
    r7b = router.route_sync(
        "I've had enough — I want to cancel my account.",
        session_id="demo-cs-001",   # same session → retention sees billing failures
        customer_id="CUST-4421",
    )
    _print_result("7b", "Cancellation (same session — billing failures known)", r7b)

    # ── Cross-agent session summary ────────────────────────────────────────
    _hr("Cross-Agent Session Summary")

    summary = memory.get_session_summary("demo-cs-001")
    print(f"\n  Session         : {summary['session_id']}")
    print(f"  Customer        : {summary['customer_id']}")
    print(f"  Duration        : {summary['duration_s']}s")
    print(f"  Agents involved : {', '.join(summary['agents_involved'])}")
    print("\n  Decision trail:")
    for d in summary["decisions"]:
        did = (d["decision_id"] or "n/a")[:14]
        print(
            f"    [{d['agent']:<26}] {d['intent']:<30} "
            f"{d['confidence']:.0%}  id={did}"
        )

    # ── All sessions summary ───────────────────────────────────────────────
    _hr("All Sessions")
    all_sessions = memory.all_sessions()
    print(f"\n  Total sessions : {len(all_sessions)}")
    for s in all_sessions:
        print(
            f"  {s['session_id']:<22}  "
            f"agents={len(s['agents_involved'])}  "
            f"decisions={s['total_decisions']}  "
            f"duration={s['duration_s']}s"
        )

    # ── ProofLayer governance dashboard ───────────────────────────────────
    _hr("ProofLayer Governance Snapshot")
    try:
        report = gateway.governance_report()
        ov = report.get("overview", {})
        print(f"\n  Total decisions  : {ov.get('total_decisions', 'n/a')}")
        print(f"  PII flagged      : {ov.get('pii_decision_count', 'n/a')}")
        print(f"  Trace steps      : {ov.get('trace_step_count', 'n/a')}")
        print(f"  Open exceptions  : {report.get('open_exceptions', 'n/a')}")
        print(f"  Cross-agent edges: {report.get('cross_agent_edges', 'n/a')}")
    except Exception as exc:
        print(f"\n  [ProofLayer unreachable — docker compose up] {exc}")

    _hr("Demo complete")
    print(
        "\n  Admin UI   : http://localhost:8000/ui/admin.html\n"
        "  Seed data  : python scripts/seed_prooflayer_demo.py\n"
        "  All decisions, trace steps, and cross-agent edges are recorded\n"
        "  in ProofLayer for the CISO governance dashboard.\n"
    )


if __name__ == "__main__":
    main()
