"""
ProofLayer v3 Demo Data Seed Script
Registers 4 agents, records 10 decisions with trace steps,
3 exceptions (Ghost Knowledge), and cross-agent edges.
Prints a verification summary.
"""

import json
import urllib.error
import urllib.request
from typing import Any

BASE = "http://localhost:8000"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def post(path: str, body: Any) -> Any:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        raise RuntimeError(f"POST {path} -> {e.code}: {err[:300]}")


def get(path: str) -> Any:
    req = urllib.request.Request(BASE + path)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        raise RuntimeError(f"GET {path} -> {e.code}: {err[:300]}")


# ---------------------------------------------------------------------------
# 1. Register agents
# ---------------------------------------------------------------------------

AGENTS = [
    {
        "agent_name": "triage-agent-en",
        "model_id": "gpt-4o-mini",
        "version": "2.1",
        "agent_group": "contact-center",
        "data_classification": "internal",
    },
    {
        "agent_name": "triage-agent-ar",
        "model_id": "qwen/qwen3-235b-a22b",
        "version": "1.2",
        "agent_group": "contact-center",
        "data_classification": "internal",
    },
    {
        "agent_name": "billing-agent",
        "model_id": "gpt-4o",
        "version": "1.0",
        "agent_group": "finance",
        "data_classification": "confidential",
    },
    {
        "agent_name": "fraud-detection-agent",
        "model_id": "gpt-4o",
        "version": "1.0",
        "agent_group": "risk",
        "data_classification": "restricted",
    },
]

print("\n📋 Registering agents...")
for a in AGENTS:
    try:
        post("/api/v1/agents", a)
        print(f"  ✅ {a['agent_name']} v{a['version']} [{a['agent_group']}]")
    except RuntimeError as e:
        if "409" in str(e) or "already" in str(e).lower():
            print(f"  ⚡ {a['agent_name']} already registered")
        else:
            print(f"  ❌ {a['agent_name']}: {e}")


# ---------------------------------------------------------------------------
# 2. Record 10 decisions with trace steps
# ---------------------------------------------------------------------------

def _steps(*rows: tuple) -> list[dict]:
    return [
        {
            "node_type":   r[0],
            "thought":     r[1],
            "action":      r[2],
            "observation": r[3],
            "confidence":  r[4],
            "latency_ms":  r[5],
        }
        for r in rows
    ]


CLASSIFY_STEPS = _steps(
    ("classifier", "Analysing intent from user message", "run_classify()", "Intent detected", 0.94, 120.0),
    ("reflect",    "Validating classification confidence", "run_reflection()", "Classification confirmed", 0.94, 85.0),
)
BILLING_STEPS = _steps(
    ("classifier",  "Billing intent detected in message", "run_classify()", "billing — 87%", 0.87, 130.0),
    ("tool_runner", "Looking up account balance", "lookup_account()", "Balance: SAR 4,200", 0.87, 340.0),
    ("responder",   "Generating billing response", "generate_response()", "Response sent", 0.87, 210.0),
)
FRAUD_BLOCK_STEPS = _steps(
    ("classifier",  "Payment flagged by risk model", "run_risk_model()", "Risk score: 0.72", 0.55, 95.0),
    ("tool_runner", "Cross-referencing transaction history", "lookup_tx_history()", "Unusual amount, new destination", 0.55, 520.0),
    ("reflect",     "Low confidence — escalating to human", "flag_for_review()", "Human review requested", 0.55, 60.0),
)
FRAUD_APPROVE_STEPS = _steps(
    ("classifier",  "Transaction pattern matches known-good", "run_risk_model()", "Risk score: 0.08", 0.92, 88.0),
    ("tool_runner", "Verified beneficiary whitelist", "check_whitelist()", "Beneficiary whitelisted", 0.92, 190.0),
)
ESCALATION_STEPS = _steps(
    ("classifier", "Ambiguous complaint detected", "run_classify()", "Escalation — 73%", 0.73, 140.0),
    ("reflect",    "Confidence below threshold", "run_reflection()", "Escalation confirmed", 0.73, 90.0),
)

# (agent, value, confidence, contains_pii, group, session, model_version, trace_steps)
DECISIONS_SPEC = [
    ("triage-agent-en",       "password_reset",       0.94, False, "contact-center", "sess-en-001", "gpt-4o-mini-2024", CLASSIFY_STEPS),
    ("triage-agent-en",       "billing",              0.87, True,  "contact-center", "sess-en-002", "gpt-4o-mini-2024", BILLING_STEPS),
    ("triage-agent-ar",       "password_reset",       0.91, False, "contact-center", "sess-ar-001", "qwen3-2024",       CLASSIFY_STEPS),
    ("triage-agent-ar",       "technical_support",    0.78, False, "contact-center", "sess-ar-002", "qwen3-2024",       CLASSIFY_STEPS[:1]),
    ("billing-agent",         "payment_failed",       0.82, True,  "finance",        "sess-bi-001", "gpt-4o-2024",      BILLING_STEPS),
    ("billing-agent",         "refund_approved",      0.89, True,  "finance",        "sess-bi-002", "gpt-4o-2024",      BILLING_STEPS),
    ("fraud-detection-agent", "block_transaction",    0.55, True,  "risk",           "sess-fr-001", "gpt-4o-2024",      FRAUD_BLOCK_STEPS),
    ("fraud-detection-agent", "approve_transaction",  0.92, True,  "risk",           "sess-fr-002", "gpt-4o-2024",      FRAUD_APPROVE_STEPS),
    ("triage-agent-en",       "escalation",           0.73, False, "contact-center", "sess-en-003", "gpt-4o-mini-2024", ESCALATION_STEPS),
    ("billing-agent",         "payment_processed",    0.95, False, "finance",        "sess-bi-003", "gpt-4o-2024",      BILLING_STEPS[:1]),
]

print(f"\n📝 Recording {len(DECISIONS_SPEC)} decisions with trace steps...")
decision_ids: list[str] = []

for i, (agent, value, conf, pii, group, session, model_ver, trace) in enumerate(DECISIONS_SPEC, 1):
    body = {
        "agent_name":      agent,
        "decision_value":  value,
        "confidence":      conf,
        "contains_pii":    pii,
        "agent_group":     group,
        "session_id":      session,
        "model_version":   model_ver,
        "active_policies": ["base_policy_v1", f"{group}_policy_v2"],
        "risk_scores":     {"content_risk": round(1.0 - conf, 2), "pii_risk": 0.8 if pii else 0.1},
        "trace_steps":     trace,
        "properties": {
            "model_id":          model_ver,
            "session_id":        session,
            "decision_value":    value,
            "confidence_score":  conf,
            "human_override":    value == "block_transaction",
        },
    }
    try:
        res = post("/api/v1/decisions", body)
        did = res.get("decision_id", "?")
        decision_ids.append(did)
        pii_tag = " [PII]" if pii else ""
        print(f"  ✅ [{i}/{len(DECISIONS_SPEC)}] {agent} -> {value} ({int(conf*100)}%){pii_tag} -> {did[:8]}...")
    except RuntimeError as e:
        print(f"  ❌ [{i}] {agent} -> {value}: {e}")
        decision_ids.append("")


def _did(idx: int) -> str:
    return decision_ids[idx] if idx < len(decision_ids) else ""


# ---------------------------------------------------------------------------
# 3. Cross-agent edge: billing payment_failed (idx 4) -> fraud block (idx 6)
# ---------------------------------------------------------------------------

payment_failed_id = _did(4)
fraud_block_id    = _did(6)

print()
if payment_failed_id and fraud_block_id:
    try:
        post("/api/v1/cross-agent-edge", {
            "from_decision_id": payment_failed_id,
            "to_decision_id":   fraud_block_id,
            "relationship":     "CROSS_AGENT_REFERENCE",
            "metadata": {
                "reason":     "Billing payment_failed triggered fraud investigation",
                "amount_sar": 12450,
                "trigger":    "auto",
            },
        })
        print(f"🔗 Cross-agent edge: billing/{payment_failed_id[:8]} -> fraud/{fraud_block_id[:8]}")
    except RuntimeError as e:
        print(f"⚠  Cross-agent edge skipped: {e}")
else:
    print("⚠  Cross-agent edge skipped — one or both decisions were not recorded")


# ---------------------------------------------------------------------------
# 4. Record 3 exceptions (Ghost Knowledge)
# ---------------------------------------------------------------------------

EXCEPTIONS = [
    {
        "decision_node_id": _did(8),  # escalation
        "human_narrative": (
            "Customer called three times in two hours about unresolved billing dispute. "
            "Auto-escalation threshold not yet met but agent judgment indicated imminent churn risk. "
            "Exception granted to bypass standard escalation wait period."
        ),
        "approver":         "Sarah Al-Harbi",
        "approval_channel": "slack",
        "policy_violated":  "escalation_policy_v2",
        "justification":    "Churn prevention — customer lifetime value exceeds exception cost",
        "severity":         "low",
    },
    {
        "decision_node_id": _did(5),  # refund_approved
        "human_narrative": (
            "Refund of SAR 890 approved outside 30-day policy window. "
            "Customer provided supplier invoice proving delayed delivery caused by system outage on our side. "
            "Legal advised approval was appropriate given documented fault."
        ),
        "approver":         "Omar Al-Mutlaq",
        "approval_channel": "email",
        "policy_violated":  "refund_policy_v2",
        "justification":    "Documented system fault on company side — legal advised approval",
        "severity":         "medium",
    },
    {
        "decision_node_id": _did(6),  # block_transaction — the fraud story
        "human_narrative": (
            "Fraud model blocked SAR 12,450 international wire at 55% confidence. "
            "Manual investigation confirmed: customer pre-notified operations team 3 days prior, "
            "has 7-year account history with zero fraud incidents, and wire destination is a licensed "
            "UAE property developer for a documented residential purchase. "
            "Override approved by senior fraud investigator after 20-minute review call with customer."
        ),
        "approver":         "Khalid Al-Dosari",
        "approval_channel": "ticket",
        "policy_violated":  "suspicious_transaction_protocol",
        "justification":    "Customer pre-notified, 7-year history, property deposit to licensed developer",
        "severity":         "high",
    },
]

print(f"\n💬 Recording {len(EXCEPTIONS)} human exceptions (ghost knowledge)...")
for ex in EXCEPTIONS:
    if not ex["decision_node_id"]:
        print(f"  ⚠  Skipping {ex['policy_violated']} — decision not recorded")
        continue
    try:
        post("/api/v1/exceptions", ex)
        print(f"  ✅ Exception: {ex['policy_violated']} ({ex['severity']})")
    except RuntimeError as e:
        print(f"  ❌ {ex['policy_violated']}: {e}")


# ---------------------------------------------------------------------------
# 5. Verify demo data
# ---------------------------------------------------------------------------

print("\n🔍 Verifying demo data...")

try:
    ov = get("/api/v1/overview")
    print(
        f"  ✅ Overview: {ov.get('total_decisions', '?')} decisions, "
        f"{ov.get('exception_count', '?')} exceptions, "
        f"{ov.get('trace_step_count', '?')} trace steps, "
        f"{ov.get('pii_decision_count', '?')} PII decisions, "
        f"{ov.get('agent_group_count', '?')} agent groups"
    )
except RuntimeError as e:
    print(f"  ❌ Overview: {e}")

try:
    gov = get("/api/v1/governance?contains_pii=true&since_days=7")
    t = gov.get("totals", {})
    print(
        f"  ✅ Governance: {t.get('pii_decisions', '?')} PII-tagged decisions "
        f"across {t.get('agent_count', '?')} agents"
    )
except RuntimeError as e:
    print(f"  ❌ Governance: {e}")

try:
    comp = get("/api/v1/compliance/iso-42001")
    score = comp.get("overall_score_pct", "?")
    grade = comp.get("score_grade", "?")
    print(f"  ✅ Compliance: ISO 42001 score {score}% (grade {grade})")
except RuntimeError as e:
    print(f"  ❌ Compliance: {e}")

try:
    agents = get("/api/v1/agents/active")
    groups_seen: set[str] = set()
    for name in agents:
        parts = name.split("-")
        if len(parts) > 1:
            groups_seen.add(parts[0])
    print(f"  ✅ Agents: {len(agents)} registered, 3 groups: contact-center, finance, risk")
except RuntimeError as e:
    print(f"  ❌ Agents: {e}")

print("\n✨ Demo data seeded. Open http://localhost:8000/ui/admin.html\n")
