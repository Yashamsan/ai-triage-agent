"""ProofLayer v3 Demo Data Seed Script.

Registers 6 agents, records 15 decisions with trace steps,
5 exceptions (Ghost Knowledge), and 3 cross-agent edges.
Prints a full verification summary.

Run after docker compose up:
    python scripts/seed_prooflayer_demo.py
"""

import json
import urllib.error
import urllib.request
from typing import Any

BASE = "http://localhost:8000"


# ── HTTP helpers ──────────────────────────────────────────────────────────────

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
        raise RuntimeError(f"POST {path} -> {e.code}: {err[:400]}")


def get(path: str) -> Any:
    req = urllib.request.Request(BASE + path)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        err = e.read().decode(errors="replace")
        raise RuntimeError(f"GET {path} -> {e.code}: {err[:400]}")


# ── 1. Register agents ────────────────────────────────────────────────────────

AGENTS = [
    {
        "agent_name":          "triage-agent-en",
        "model_id":            "deepseek/deepseek-chat",
        "version":             "2.1",
        "agent_group":         "contact-center",
        "data_classification": "internal",
    },
    {
        "agent_name":          "triage-agent-ar",
        "model_id":            "openrouter/qwen/qwen3-235b-a22b",
        "version":             "1.2",
        "agent_group":         "contact-center",
        "data_classification": "internal",
    },
    {
        "agent_name":          "billing-agent",
        "model_id":            "deepseek/deepseek-chat",
        "version":             "1.0",
        "agent_group":         "finance",
        "data_classification": "confidential",
    },
    {
        "agent_name":          "fraud-detection-agent",
        "model_id":            "deepseek/deepseek-chat",
        "version":             "1.0",
        "agent_group":         "risk",
        "data_classification": "restricted",
    },
    {
        "agent_name":          "technical-agent",
        "model_id":            "deepseek/deepseek-chat",
        "version":             "1.0",
        "agent_group":         "support",
        "data_classification": "internal",
    },
    {
        "agent_name":          "retention-agent",
        "model_id":            "deepseek/deepseek-chat",
        "version":             "1.0",
        "agent_group":         "retention",
        "data_classification": "confidential",
    },
]

print("\n=== Step 1: Register Agents ===")
for a in AGENTS:
    try:
        post("/api/v1/agents", a)
        print(f"  [ok] {a['agent_name']} v{a['version']} [{a['agent_group']}]")
    except RuntimeError as e:
        if "409" in str(e) or "already" in str(e).lower():
            print(f"  [--] {a['agent_name']} already registered")
        else:
            print(f"  [!!] {a['agent_name']}: {e}")


# ── 2. Trace step builders ────────────────────────────────────────────────────

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


STEPS_TRIAGE = _steps(
    ("classifier", "Analyse intent from user message",         "run_classify()",    "Intent detected",              0.94, 120.0),
    ("reflect",    "Validate classification confidence",        "run_reflection()",  "Classification confirmed",      0.94,  85.0),
    ("tool_runner","Look up FAQ for matched intent",           "faq_lookup()",      "FAQ article found",             0.94, 210.0),
    ("responder",  "Generate final response",                   "generate_response()","Response dispatched",          0.94,  95.0),
)
STEPS_TRIAGE_AR = _steps(
    ("classifier-ar","Classify Arabic intent (Qwen3)",          "classify_ar()",     "Intent: greeting (97%)",        0.97, 145.0),
    ("reflect-ar",   "Bilingual LLM-as-Judge validation",       "reflect_ar()",      "Classification confirmed",      0.97,  92.0),
    ("tool_runner",  "FAQ lookup (Arabic vector search)",        "faq_lookup_ar()",   "FAQ article found (AR)",        0.97, 230.0),
    ("responder-ar", "Generate Arabic response",                 "generate_response_ar()","Response dispatched",       0.97, 105.0),
)
STEPS_BILLING = _steps(
    ("classifier",  "Billing intent detected",                  "run_classify()",    "billing_error — 93%",          0.93, 130.0),
    ("tool_runner", "Look up account transaction history",      "lookup_account()",  "Duplicate charge confirmed",    0.93, 340.0),
    ("reflect",     "Verify charge details against policy",     "run_reflection()",  "Duplicate verified",           0.93,  88.0),
    ("responder",   "Initiate refund and generate response",    "process_refund()",  "Refund initiated SAR 149",      0.93, 210.0),
)
STEPS_BILLING_REFUND = _steps(
    ("classifier",  "Refund intent classified",                 "run_classify()",    "refund_request — 90%",         0.90, 118.0),
    ("tool_runner", "Verify eligibility within 30-day window",  "check_refund_eligibility()", "Eligible",            0.90, 290.0),
    ("responder",   "Process refund and notify customer",       "process_refund()",  "Refund SAR 890 processed",     0.90, 180.0),
)
STEPS_FRAUD_BLOCK = _steps(
    ("classifier",  "Transaction risk model triggered",         "run_risk_model()",  "Risk score: 0.72",             0.55,  95.0),
    ("tool_runner", "Cross-reference transaction history",      "lookup_tx_history()","Unusual: new destination",     0.55, 520.0),
    ("reflect",     "Low confidence — flag for human review",   "flag_for_review()", "Human review requested",        0.55,  60.0),
)
STEPS_FRAUD_APPROVE = _steps(
    ("classifier",  "Transaction matches known-good pattern",   "run_risk_model()",  "Risk score: 0.08",             0.92,  88.0),
    ("tool_runner", "Verify beneficiary whitelist",             "check_whitelist()", "Beneficiary whitelisted",       0.92, 190.0),
    ("responder",   "Approve transaction",                      "approve_tx()",      "Transaction approved",          0.92,  75.0),
)
STEPS_TECHNICAL = _steps(
    ("tech-classifier", "Detect service outage signal",         "detect_issue()",    "service_outage — 93%",         0.93,  98.0),
    ("tool_runner",     "Query status page and incident log",   "check_status()",    "Active incident confirmed",     0.93, 280.0),
    ("responder",       "Generate outage notice",               "notify_customer()", "ETA communicated: 45 min",      0.93, 120.0),
)
STEPS_TECHNICAL_API = _steps(
    ("tech-classifier", "API error classified",                 "detect_issue()",    "api_error — 89%",              0.89, 102.0),
    ("tool_runner",     "Check API health endpoint",            "check_api_health()","p99 latency: 4800ms (breach)", 0.89, 310.0),
    ("reflect",         "Severity assessment",                  "assess_severity()", "SEV-2 auto-opened",             0.89,  55.0),
    ("responder",       "Create incident ticket",               "create_ticket()",   "Incident #INC-38291 opened",    0.89, 145.0),
)
STEPS_RETENTION = _steps(
    ("retention-classifier","Cancellation signal detected",     "classify_churn()",  "cancellation_request — 94%",   0.94, 110.0),
    ("tool_runner",         "Look up customer tenure & LTV",    "lookup_customer()", "3.2 yr | LTV: SAR 8,400",      0.94, 350.0),
    ("reflect",             "Churn risk & save-offer scoring",  "score_save_offer()","Save probability: 72%",         0.94, 180.0),
    ("responder",           "Generate personalised save offer", "generate_offer()",  "3-month free offer dispatched", 0.94, 220.0),
)
STEPS_RETENTION_WIN = _steps(
    ("retention-classifier","Win-back signal detected",         "classify_churn()",  "win_back — 91%",               0.91, 105.0),
    ("tool_runner",         "Retrieve previous plan details",   "lookup_account()",  "Previous: Pro plan",            0.91, 240.0),
    ("responder",           "Reinstate account with offer",     "reinstate_account()","2 months free applied",        0.91, 160.0),
)
STEPS_ESCALATION = _steps(
    ("classifier", "Ambiguous complaint detected",              "run_classify()",    "escalation — 73%",             0.73, 140.0),
    ("reflect",    "Confidence below tier threshold",           "run_reflection()",  "Human escalation confirmed",    0.73,  90.0),
)

# ── 3. Record decisions ───────────────────────────────────────────────────────
# Columns: agent, decision_value, confidence, contains_pii, group, session, model_id, trace_steps

DECISIONS_SPEC: list[tuple] = [
    # 0  triage-en: password reset
    ("triage-agent-en",       "password_reset",        0.94, False, "contact-center", "sess-en-001", "deepseek/deepseek-chat", STEPS_TRIAGE),
    # 1  triage-en: billing inquiry (passes to billing-agent cross-edge)
    ("triage-agent-en",       "billing_inquiry",       0.87, True,  "contact-center", "sess-en-002", "deepseek/deepseek-chat", STEPS_TRIAGE[:2]),
    # 2  triage-ar: Arabic greeting
    ("triage-agent-ar",       "ar_greeting",           0.97, False, "contact-center", "sess-ar-001", "qwen/qwen3-235b-a22b",   STEPS_TRIAGE_AR),
    # 3  triage-ar: Arabic technical issue
    ("triage-agent-ar",       "ar_technical_issue",    0.84, False, "contact-center", "sess-ar-002", "qwen/qwen3-235b-a22b",   STEPS_TRIAGE_AR[:2]),
    # 4  billing: billing error (duplicate charge)
    ("billing-agent",         "billing_error",         0.93, True,  "finance",        "sess-bi-001", "deepseek/deepseek-chat", STEPS_BILLING),
    # 5  billing: refund approved (exception scenario)
    ("billing-agent",         "refund_approved",       0.90, True,  "finance",        "sess-bi-002", "deepseek/deepseek-chat", STEPS_BILLING_REFUND),
    # 6  billing: payment failed
    ("billing-agent",         "payment_failed",        0.82, True,  "finance",        "sess-bi-003", "deepseek/deepseek-chat", STEPS_BILLING[:2]),
    # 7  fraud: block transaction (low confidence → exception)
    ("fraud-detection-agent", "block_transaction",     0.55, True,  "risk",           "sess-fr-001", "deepseek/deepseek-chat", STEPS_FRAUD_BLOCK),
    # 8  fraud: approve transaction
    ("fraud-detection-agent", "approve_transaction",   0.92, True,  "risk",           "sess-fr-002", "deepseek/deepseek-chat", STEPS_FRAUD_APPROVE),
    # 9  fraud: account compromise
    ("fraud-detection-agent", "account_compromise",    0.95, True,  "risk",           "sess-fr-003", "deepseek/deepseek-chat", STEPS_FRAUD_BLOCK[:2]),
    # 10 technical: service outage
    ("technical-agent",       "service_outage",        0.93, False, "support",        "sess-tc-001", "deepseek/deepseek-chat", STEPS_TECHNICAL),
    # 11 technical: API error
    ("technical-agent",       "api_error",             0.89, False, "support",        "sess-tc-002", "deepseek/deepseek-chat", STEPS_TECHNICAL_API),
    # 12 retention: cancellation (cross-agent: knows billing-agent resolved the double-charge)
    ("retention-agent",       "cancellation_request",  0.94, True,  "retention",      "sess-bi-001", "deepseek/deepseek-chat", STEPS_RETENTION),
    # 13 retention: win-back
    ("retention-agent",       "win_back",              0.91, True,  "retention",      "sess-rt-001", "deepseek/deepseek-chat", STEPS_RETENTION_WIN),
    # 14 triage-en: escalation (ambiguous, low-confidence)
    ("triage-agent-en",       "escalation",            0.73, False, "contact-center", "sess-en-003", "deepseek/deepseek-chat", STEPS_ESCALATION),
]

print(f"\n=== Step 2: Record {len(DECISIONS_SPEC)} Decisions with Trace Steps ===")
decision_ids: list[str] = []

for i, (agent, value, conf, pii, group, session, model_id, trace) in enumerate(DECISIONS_SPEC):
    body = {
        "agent_name":      agent,
        "decision_value":  value,
        "confidence":      conf,
        "contains_pii":    pii,
        "agent_group":     group,
        "session_id":      session,
        "model_version":   model_id,
        "active_policies": ["base_policy_v1", f"{group}_policy_v2"],
        "risk_scores":     {"content_risk": round(1.0 - conf, 2), "pii_risk": 0.8 if pii else 0.1},
        "trace_steps":     trace,
        "properties": {
            "model_id":         model_id,
            "session_id":       session,
            "decision_value":   value,
            "confidence_score": conf,
            "human_override":   value in ("block_transaction", "escalation"),
        },
    }
    try:
        res = post("/api/v1/decisions", body)
        did = res.get("decision_id", "?")
        decision_ids.append(did)
        pii_tag = " [PII]" if pii else ""
        steps_n = len(trace)
        print(f"  [{i:02d}] {agent:<26} -> {value:<28} {int(conf*100)}%{pii_tag} | {steps_n} steps | {did[:8]}...")
    except RuntimeError as e:
        print(f"  [!!] [{i:02d}] {agent} -> {value}: {e}")
        decision_ids.append("")


def _did(idx: int) -> str:
    return decision_ids[idx] if idx < len(decision_ids) else ""


# ── 4. Cross-agent edges ──────────────────────────────────────────────────────
# Edge 1: billing (billing_error, idx 4) → fraud (block_transaction, idx 7)
#         Billing payment_failed triggered fraud investigation
# Edge 2: triage-en (billing_inquiry, idx 1) → billing (billing_error, idx 4)
#         Router handed off from first contact to specialist
# Edge 3: billing-agent (billing_error, idx 4) → retention (cancellation, idx 12)
#         Same session: billing resolved → retention saw context

CROSS_EDGES = [
    {
        "from_idx":     4,
        "to_idx":       7,
        "relationship": "TRIGGERED_FRAUD_REVIEW",
        "metadata": {
            "reason":     "Billing billing_error triggered fraud investigation",
            "amount_sar": 12450,
            "trigger":    "auto",
        },
        "description": "billing/billing_error -> fraud/block_transaction",
    },
    {
        "from_idx":     1,
        "to_idx":       4,
        "relationship": "ROUTER_HANDOFF",
        "metadata": {
            "reason":     "Triage classified billing_inquiry, dispatched to specialist",
            "tier":       1,
        },
        "description": "triage-en/billing_inquiry -> billing/billing_error",
    },
    {
        "from_idx":     4,
        "to_idx":       12,
        "relationship": "CROSS_AGENT_CONTEXT_SHARED",
        "metadata": {
            "reason":     "billing-agent resolved duplicate charge in same session; retention used context",
            "session_id": "sess-bi-001",
        },
        "description": "billing/billing_error -> retention/cancellation_request",
    },
]

print(f"\n=== Step 3: Record {len(CROSS_EDGES)} Cross-Agent Edges ===")
for edge in CROSS_EDGES:
    from_id = _did(edge["from_idx"])
    to_id   = _did(edge["to_idx"])
    if not from_id or not to_id:
        print(f"  [--] Skipped ({edge['description']}) — decision IDs missing")
        continue
    try:
        post("/api/v1/cross-agent-edge", {
            "from_decision_id": from_id,
            "to_decision_id":   to_id,
            "relationship":     edge["relationship"],
            "metadata":         edge["metadata"],
        })
        print(f"  [ok] {edge['relationship']:<36} {edge['description']}")
    except RuntimeError as e:
        print(f"  [!!] {edge['description']}: {e}")


# ── 5. Record exceptions (Ghost Knowledge) ────────────────────────────────────

EXCEPTIONS = [
    {
        "decision_id": _did(14),   # escalation (sess-en-003)
        "human_narrative": (
            "Customer called three times in two hours about an unresolved billing dispute. "
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
        "decision_id": _did(5),    # refund_approved (sess-bi-002)
        "human_narrative": (
            "Refund of SAR 890 approved outside the 30-day policy window. "
            "Customer provided supplier invoice proving delayed delivery caused by a system outage "
            "on our side. Legal advised approval was appropriate given the documented company fault."
        ),
        "approver":         "Omar Al-Mutlaq",
        "approval_channel": "email",
        "policy_violated":  "refund_policy_v2",
        "justification":    "Documented system fault on company side — legal advised approval",
        "severity":         "medium",
    },
    {
        "decision_id": _did(7),    # block_transaction (55% confidence)
        "human_narrative": (
            "Fraud model blocked SAR 12,450 international wire at 55% confidence. "
            "Manual investigation confirmed: customer pre-notified operations 3 days prior, "
            "has 7-year account history with zero fraud incidents, and wire destination is a "
            "licensed UAE property developer for a documented residential purchase. "
            "Override approved by senior fraud investigator after a 20-minute review call."
        ),
        "approver":         "Khalid Al-Dosari",
        "approval_channel": "ticket",
        "policy_violated":  "suspicious_transaction_protocol",
        "justification":    "Customer pre-notified; 7-year history; property deposit to licensed developer",
        "severity":         "high",
    },
    {
        "decision_id": _did(9),    # account_compromise
        "human_narrative": (
            "Account compromise flag on a known corporate shared-login account. "
            "The 'unusual IP' was the customer's new satellite office in Riyadh. "
            "CISO confirmed this IP range should be whitelisted; agent decision correct "
            "but whitelisting was not yet applied at time of incident."
        ),
        "approver":         "Nora Al-Rashid",
        "approval_channel": "slack",
        "policy_violated":  "account_compromise_protocol",
        "justification":    "Known corporate shared login — satellite office IP not yet whitelisted",
        "severity":         "medium",
    },
    {
        "decision_id": _did(12),   # cancellation_request (retention)
        "human_narrative": (
            "Retention agent offered standard 3-month free extension to a high-LTV customer "
            "threatening cancellation. Customer accepted but requested the billing duplicate "
            "refund (resolved in the same session by billing-agent) to be expedited to 1 business day "
            "instead of the standard 5-7 days. Exception approved to retain a customer with "
            "SAR 8,400 LTV and 3.2-year tenure."
        ),
        "approver":         "Fatima Al-Zahra",
        "approval_channel": "phone",
        "policy_violated":  "refund_sla_policy_v1",
        "justification":    "High LTV retention; billing error on our side; expedited refund appropriate",
        "severity":         "low",
    },
]

print(f"\n=== Step 4: Record {len(EXCEPTIONS)} Human Exceptions (Ghost Knowledge) ===")
for ex in EXCEPTIONS:
    if not ex["decision_id"]:
        print(f"  [--] Skipped {ex['policy_violated']} — decision ID missing")
        continue
    try:
        post("/api/v1/exceptions", ex)
        print(f"  [ok] {ex['policy_violated']:<45} severity={ex['severity']}")
    except RuntimeError as e:
        print(f"  [!!] {ex['policy_violated']}: {e}")


# ── 6. Verify ─────────────────────────────────────────────────────────────────

print("\n=== Step 5: Verification ===")

try:
    ov = get("/api/v1/overview")
    print(
        f"  [ok] Overview: {ov.get('total_decisions', '?')} decisions | "
        f"{ov.get('exception_count', '?')} exceptions | "
        f"{ov.get('trace_step_count', '?')} trace steps | "
        f"{ov.get('pii_decision_count', '?')} PII | "
        f"{ov.get('agent_group_count', '?')} groups"
    )
except RuntimeError as e:
    print(f"  [!!] Overview: {e}")

try:
    gov = get("/api/v1/governance?contains_pii=true&since_days=7")
    t = gov.get("totals", {})
    print(
        f"  [ok] Governance: {t.get('pii_decisions', '?')} PII decisions "
        f"across {t.get('agent_count', '?')} agents"
    )
except RuntimeError as e:
    print(f"  [!!] Governance: {e}")

try:
    comp = get("/api/v1/compliance/iso-42001")
    score = comp.get("overall_score_pct", "?")
    grade = comp.get("score_grade", "?")
    print(f"  [ok] ISO 42001 compliance score: {score}% (grade {grade})")
except RuntimeError as e:
    print(f"  [!!] Compliance: {e}")

try:
    agents_active = get("/api/v1/agents/active")
    groups: set[str] = set()
    for a in agents_active:
        for registered in AGENTS:
            if registered["agent_name"] == a:
                groups.add(registered["agent_group"])
    print(f"  [ok] Agents: {len(agents_active)} active | groups: {', '.join(sorted(groups)) or 'n/a'}")
except RuntimeError as e:
    print(f"  [!!] Agents: {e}")

# Per-agent decision counts
print("\n  Decision breakdown by agent:")
for a in AGENTS:
    count = sum(1 for spec in DECISIONS_SPEC if spec[0] == a["agent_name"])
    print(f"    {a['agent_name']:<30} {count} decisions")

print(f"\n  Total decisions seeded : {len(DECISIONS_SPEC)}")
print(f"  Total exceptions       : {len(EXCEPTIONS)}")
print(f"  Cross-agent edges      : {len(CROSS_EDGES)}")
print(f"  Trace steps (approx)   : {sum(len(s[7]) for s in DECISIONS_SPEC)}")
print("\n  Admin dashboard -> http://localhost:8000/ui/admin.html")
print( "  Multi-agent demo  -> python scripts/demo_multi_agent.py\n")
