"""Multi-agent orchestration demo.

Demonstrates the full architecture:
  - 6 specialist agents (triage-en, triage-ar, billing, fraud, technical, retention)
  - RouterAgent with two-stage dispatch (Tier 1 / Tier 2 / Tier 3)
  - Cross-agent session memory (retention sees billing history)
  - ProofLayer governance recording (single CISO dashboard)

Run:
    docker compose up -d          # needs ProofLayer DB running
    python scripts/demo_multi_agent.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orchestrator import (
    BaseSpecialistAgent,
    MemoryBridge,
    ProofLayerGateway,
    RouterAgent,
    load_config,
)
from orchestrator.agent_base import AgentState


# ── Specialist Agent Implementations ─────────────────────────────────────────
# Each agent inherits BaseSpecialistAgent and overrides two nodes:
#   _classify_node → sets intent / confidence / contains_pii
#   _respond_node  → sets response_text
#
# These use keyword-based classification for a fast, deterministic demo.
# In production, replace _classify_node with an LLM call (same interface).


class TriageAgentEN(BaseSpecialistAgent):
    """English-language first-contact triage."""

    version  = "2.1"
    model_id = "deepseek/deepseek-chat"

    @property
    def name(self) -> str: return "triage-agent-en"

    @property
    def group(self) -> str: return "contact-center"

    def _classify_node(self, state: AgentState) -> dict:
        m = state["message"].lower()
        if any(w in m for w in ["hello", "hi", "hey", "good"]):
            intent, conf = "greeting", 0.97
        elif any(w in m for w in ["password", "login", "locked", "access"]):
            intent, conf = "password_reset", 0.93
        elif "bill" in m or "charge" in m or "invoice" in m:
            intent, conf = "billing_inquiry", 0.85
        elif any(w in m for w in ["broken", "not working", "error", "issue"]):
            intent, conf = "technical_issue", 0.88
        elif any(w in m for w in ["down", "outage", "status"]):
            intent, conf = "service_status", 0.90
        else:
            intent, conf = "general_inquiry", 0.72
        step = self._trace_step(
            "classifier", thought="EN keyword classifier",
            action="keyword_match()", observation=f"→ {intent}",
            confidence=conf, latency_ms=3.0,
        )
        return {"intent": intent, "confidence": conf, "contains_pii": False,
                "trace_steps": [step]}

    def _respond_node(self, state: AgentState) -> dict:
        _RESPONSES = {
            "greeting":        "Hello! Welcome to support. How can I help you today?",
            "password_reset":  "I'll send a password reset link to your registered email right away.",
            "billing_inquiry": "Let me pull up your billing details.",
            "technical_issue": "I can see the issue. Let me create a ticket and escalate to our tech team.",
            "service_status":  "Checking our status page... all systems are currently operational.",
            "general_inquiry": "Thanks for reaching out. A specialist will assist you shortly.",
        }
        text = _RESPONSES.get(state["intent"], "How can I help you today?")
        ctx  = state.get("context", "")
        if ctx:
            text += "\n\n*[Context from previous agents noted]*"
        step = self._trace_step(
            "responder", thought="Select response template",
            action="template_response()", observation="Response ready",
            confidence=state["confidence"], latency_ms=2.0,
        )
        return {
            "response_text": f"**Support Assistant**\n\n{text}\n\n---\nConfidence: {state['confidence']:.0%}",
            "trace_steps":   state.get("trace_steps", []) + [step],
        }


class TriageAgentAR(BaseSpecialistAgent):
    """Arabic-language first-contact triage (keyword demo; uses Qwen in production)."""

    version  = "1.2"
    model_id = "openrouter/qwen/qwen3-235b-a22b"

    @property
    def name(self) -> str: return "triage-agent-ar"

    @property
    def group(self) -> str: return "contact-center"

    def _classify_node(self, state: AgentState) -> dict:
        m = state["message"]
        if any(w in m for w in ["مرحبا", "أهلاً", "السلام", "صباح", "مساء"]):
            intent, conf = "ar_greeting", 0.97
        elif any(w in m for w in ["كلمة المرور", "تسجيل", "حساب"]):
            intent, conf = "ar_password_reset", 0.91
        elif any(w in m for w in ["فاتورة", "دفع", "رسوم"]):
            intent, conf = "ar_billing_inquiry", 0.87
        elif any(w in m for w in ["مشكلة", "خطأ", "لا يعمل"]):
            intent, conf = "ar_technical_issue", 0.85
        else:
            intent, conf = "ar_general_inquiry", 0.70
        step = self._trace_step(
            "classifier-ar", thought="Arabic keyword classifier",
            action="keyword_match_ar()", observation=f"→ {intent}",
            confidence=conf, latency_ms=4.0,
        )
        return {"intent": intent, "confidence": conf, "contains_pii": False,
                "trace_steps": [step]}

    def _respond_node(self, state: AgentState) -> dict:
        _RESPONSES = {
            "ar_greeting":       "أهلاً وسهلاً! كيف يمكنني مساعدتك اليوم؟",
            "ar_password_reset": "سأرسل رابط إعادة تعيين كلمة المرور إلى بريدك الإلكتروني المسجل.",
            "ar_billing_inquiry":"دعني أطلع على تفاصيل فاتورتك.",
            "ar_technical_issue":"سأرفع تذكرة دعم فني لحل هذه المشكلة.",
            "ar_general_inquiry":"شكراً للتواصل معنا. سيتواصل معك متخصص قريباً.",
        }
        text = _RESPONSES.get(state["intent"], "كيف يمكنني مساعدتك؟")
        step = self._trace_step(
            "responder-ar", thought="Arabic response template",
            action="template_response_ar()", observation="Response ready",
            confidence=state["confidence"], latency_ms=2.0,
        )
        return {
            "response_text": f"**مساعد الدعم**\n\n{text}\n\n---\nالثقة: {state['confidence']:.0%}",
            "trace_steps":   state.get("trace_steps", []) + [step],
        }


class BillingAgent(BaseSpecialistAgent):
    """Handles payment, invoice, and subscription issues."""

    version  = "1.0"
    model_id = "deepseek/deepseek-chat"

    @property
    def name(self) -> str: return "billing-agent"

    @property
    def group(self) -> str: return "finance"

    def _classify_node(self, state: AgentState) -> dict:
        m = state["message"].lower()
        if any(w in m for w in ["refund", "money back"]):
            intent, conf = "refund_request", 0.91
        elif any(w in m for w in ["twice", "double", "duplicate", "overcharged"]):
            intent, conf = "billing_error", 0.93
        elif any(w in m for w in ["dispute", "wrong charge", "incorrect"]):
            intent, conf = "invoice_dispute", 0.88
        elif any(w in m for w in ["cancel subscription", "change plan", "upgrade", "downgrade"]):
            intent, conf = "subscription_change", 0.85
        elif any(w in m for w in ["declined", "failed", "not charged"]):
            intent, conf = "payment_failed", 0.90
        elif any(w in m for w in ["how much", "price", "cost", "plan"]):
            intent, conf = "pricing_inquiry", 0.82
        else:
            intent, conf = "billing_inquiry", 0.74
        step = self._trace_step(
            "classifier", thought="Billing intent classifier",
            action="keyword_match()", observation=f"→ {intent} | PII=true",
            confidence=conf, latency_ms=5.0,
        )
        return {"intent": intent, "confidence": conf, "contains_pii": True,
                "trace_steps": [step]}

    def _respond_node(self, state: AgentState) -> dict:
        _RESPONSES = {
            "refund_request":     "Your refund has been submitted (ref #RF-{ts}). Processing takes 5–7 business days.",
            "billing_error":      "I can see the duplicate charge. A refund for the duplicate amount has been initiated.",
            "invoice_dispute":    "I've flagged invoice #{ts} for review. Our billing team will contact you within 24 hours.",
            "subscription_change":"Your subscription has been updated as requested.",
            "payment_failed":     "Your payment method has been flagged. Please update your card details in Account Settings.",
            "pricing_inquiry":    "Our plans: Basic £9.99/mo, Pro £29.99/mo, Enterprise £99.99/mo. All include 24/7 support.",
            "billing_inquiry":    "I've pulled up your billing summary. Your next invoice is due on the 1st.",
        }
        ts   = str(int(time.time()))[-6:]
        text = _RESPONSES.get(state["intent"], "Let me look into your billing question.").format(ts=ts)

        # Acknowledge cross-agent context (e.g. prior triage interaction)
        ctx = state.get("context", "")
        if ctx:
            text += "\n\n*[Cross-agent context applied — see session log]*"

        step = self._trace_step(
            "responder", thought="Generate billing resolution",
            action="resolve_billing()", observation="Resolution prepared",
            confidence=state["confidence"], latency_ms=8.0,
        )
        return {
            "response_text": (
                f"**Billing Support**\n\n{text}\n\n"
                f"---\nConfidence: {state['confidence']:.0%} | Data class: CONFIDENTIAL"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }


class FraudDetectionAgent(BaseSpecialistAgent):
    """Detects and responds to fraud / account security incidents."""

    version  = "1.0"
    model_id = "deepseek/deepseek-chat"

    @property
    def name(self) -> str: return "fraud-detection-agent"

    @property
    def group(self) -> str: return "risk"

    def _classify_node(self, state: AgentState) -> dict:
        m = state["message"].lower()
        if any(w in m for w in ["fraud", "scam", "defraud"]):
            intent, conf = "fraud_report", 0.95
        elif any(w in m for w in ["hacked", "compromised", "someone else"]):
            intent, conf = "account_compromise", 0.94
        elif any(w in m for w in ["suspicious", "didn't make", "not me", "recognize"]):
            intent, conf = "suspicious_transaction", 0.91
        elif any(w in m for w in ["unusual", "strange", "odd"]):
            intent, conf = "unusual_activity", 0.86
        elif any(w in m for w in ["block", "freeze", "lock"]):
            intent, conf = "block_request", 0.92
        elif any(w in m for w in ["verify", "confirm", "identity"]):
            intent, conf = "verification_request", 0.83
        else:
            intent, conf = "suspicious_transaction", 0.72
        step = self._trace_step(
            "fraud-classifier", thought="Risk signal detection",
            action="detect_risk_signals()", observation=f"→ {intent} | RESTRICTED",
            confidence=conf, latency_ms=12.0,
        )
        return {"intent": intent, "confidence": conf, "contains_pii": True,
                "trace_steps": [step]}

    def _respond_node(self, state: AgentState) -> dict:
        _RESPONSES = {
            "fraud_report":         "URGENT: A fraud case has been opened (case #FD-{ts}). Your account has been temporarily secured. Our fraud team will call you within 2 hours.",
            "account_compromise":   "Your account has been locked immediately. We've revoked all active sessions. A secure reset link has been sent to your verified email.",
            "suspicious_transaction":"The transaction has been flagged for review (ref #{ts}). If you didn't authorise it, we've placed a hold and will reverse it within 24 hours.",
            "unusual_activity":     "We've noted the unusual activity and tightened security on your account. Please verify your recent logins in Security Settings.",
            "block_request":        "Your card ending in **** has been blocked immediately. A replacement will arrive in 3–5 business days.",
            "verification_request": "Identity verification initiated. Please check your email for a one-time verification link valid for 15 minutes.",
        }
        ts   = str(int(time.time()))[-6:]
        text = _RESPONSES.get(state["intent"], "Security alert logged. Our risk team is reviewing.").format(ts=ts)
        step = self._trace_step(
            "risk-responder", thought="Generate security response",
            action="execute_risk_action()", observation="Action taken and logged",
            confidence=state["confidence"], latency_ms=15.0,
        )
        return {
            "response_text": (
                f"**Fraud & Security**\n\n{text}\n\n"
                f"---\nConfidence: {state['confidence']:.0%} | Data class: RESTRICTED"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }


class TechnicalAgent(BaseSpecialistAgent):
    """Resolves connectivity, configuration, and API issues."""

    version  = "1.0"
    model_id = "deepseek/deepseek-chat"

    @property
    def name(self) -> str: return "technical-agent"

    @property
    def group(self) -> str: return "support"

    def _classify_node(self, state: AgentState) -> dict:
        m = state["message"].lower()
        if any(w in m for w in ["down", "outage", "offline", "not available"]):
            intent, conf = "service_outage", 0.93
        elif any(w in m for w in ["internet", "connectivity", "connection", "wifi"]):
            intent, conf = "connectivity_issue", 0.91
        elif any(w in m for w in ["api", "endpoint", "integration", "webhook"]):
            intent, conf = "api_error", 0.89
        elif any(w in m for w in ["slow", "laggy", "performance", "timeout"]):
            intent, conf = "performance_issue", 0.87
        elif any(w in m for w in ["configure", "setup", "settings", "config"]):
            intent, conf = "device_config", 0.84
        elif any(w in m for w in ["feature", "request", "suggestion", "improve"]):
            intent, conf = "feature_request", 0.80
        else:
            intent, conf = "technical_issue", 0.75
        step = self._trace_step(
            "tech-classifier", thought="Technical issue classifier",
            action="keyword_match()", observation=f"→ {intent}",
            confidence=conf, latency_ms=4.0,
        )
        return {"intent": intent, "confidence": conf, "contains_pii": False,
                "trace_steps": [step]}

    def _respond_node(self, state: AgentState) -> dict:
        _RESPONSES = {
            "service_outage":    "We're aware of an ongoing service disruption affecting some users. Our engineering team is working on a fix. ETA: 45 minutes. Track updates at status.example.com.",
            "connectivity_issue":"Connectivity troubleshooting: 1) Restart router. 2) Try a different network. 3) If mobile, toggle Aeroplane mode. Ticket #{ts} created if the issue persists.",
            "api_error":         "API incident logged (ref #{ts}). Check our developer docs for rate limits. If you're getting 5xx errors, our API is degraded — we'll notify via webhook.",
            "performance_issue": "Performance diagnostics running. Current p99 latency is 280ms (normal <100ms). Incident #{ts} opened with SLA priority.",
            "device_config":     "Configuration guide sent to your email. Key steps: 1) Download the agent. 2) Set API_KEY env var. 3) Run `agent --verify`.",
            "feature_request":   "Feature request logged (FR-{ts}). Our product team reviews requests weekly. You'll be notified when it enters the roadmap.",
            "technical_issue":   "Ticket #{ts} created. A technical specialist will contact you within 4 hours.",
        }
        ts   = str(int(time.time()))[-6:]
        text = _RESPONSES.get(state["intent"], "Technical ticket created.").format(ts=ts)
        step = self._trace_step(
            "tech-responder", thought="Generate technical resolution",
            action="create_ticket_and_respond()", observation="Ticket created",
            confidence=state["confidence"], latency_ms=10.0,
        )
        return {
            "response_text": (
                f"**Technical Support**\n\n{text}\n\n"
                f"---\nConfidence: {state['confidence']:.0%}"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }


class RetentionAgent(BaseSpecialistAgent):
    """Handles cancellations, downgrades, and win-back scenarios.

    Cross-agent authority: if billing-agent resolved a billing issue in this
    session, retention knows and can address it as part of the save offer.
    """

    version  = "1.0"
    model_id = "deepseek/deepseek-chat"

    @property
    def name(self) -> str: return "retention-agent"

    @property
    def group(self) -> str: return "retention"

    def _classify_node(self, state: AgentState) -> dict:
        m = state["message"].lower()
        if any(w in m for w in ["cancel", "cancellation", "close account", "stop"]):
            intent, conf = "cancellation_request", 0.94
        elif any(w in m for w in ["downgrade", "basic plan", "cheaper"]):
            intent, conf = "downgrade_request", 0.90
        elif any(w in m for w in ["competitor", "better deal", "switching to", "going to"]):
            intent, conf = "competitor_mention", 0.88
        elif any(w in m for w in ["unhappy", "disappointed", "frustrated", "poor"]):
            intent, conf = "dissatisfaction", 0.85
        elif any(w in m for w in ["come back", "return", "reconsider"]):
            intent, conf = "win_back", 0.91
        elif any(w in m for w in ["loyalty", "reward", "points", "how long"]):
            intent, conf = "loyalty_inquiry", 0.87
        else:
            intent, conf = "cancellation_request", 0.78
        step = self._trace_step(
            "retention-classifier", thought="Churn risk classification",
            action="classify_churn_signal()", observation=f"→ {intent}",
            confidence=conf, latency_ms=6.0,
        )
        return {"intent": intent, "confidence": conf, "contains_pii": True,
                "trace_steps": [step]}

    def _respond_node(self, state: AgentState) -> dict:
        # Check cross-agent context — billing resolution changes the save offer
        ctx              = state.get("context", "")
        billing_resolved = "billing-agent" in ctx and "billing_error" in ctx

        _BASE = {
            "cancellation_request": (
                "We're sorry to hear you'd like to cancel. "
                "As a valued customer, we'd like to offer you 3 months FREE "
                "before processing the cancellation. Can we discuss what's driving this?"
            ),
            "downgrade_request": (
                "I've noted your request to downgrade. Before we do that, "
                "I'd like to offer a 20% discount on your current plan for 6 months. "
                "Would that help?"
            ),
            "competitor_mention": (
                "I hear you. Before you leave, let me show you what we've improved "
                "recently — and I can offer a custom pricing package to match or beat "
                "any competitor quote."
            ),
            "dissatisfaction": (
                "Your feedback matters. I've escalated your concerns to our Customer "
                "Success Manager who will call you within 24 hours with a personalised resolution plan."
            ),
            "win_back": (
                "Welcome back! I've reinstated your account with all previous settings. "
                "As a returning customer you receive 2 months free."
            ),
            "loyalty_inquiry": (
                "You've been with us for 2+ years! You have 1,250 loyalty points "
                "worth £25 credit. Apply them in Account → Rewards."
            ),
        }
        text = _BASE.get(state["intent"], "Let me see what we can do to keep you with us.")

        if billing_resolved and state["intent"] == "cancellation_request":
            text = (
                "I can see our billing team just resolved a duplicate charge for you — "
                "I hope that helped! " + text
            )
        step = self._trace_step(
            "retention-responder", thought="Build personalised save offer",
            action="generate_save_offer()", observation="Offer ready",
            confidence=state["confidence"], latency_ms=12.0,
        )
        return {
            "response_text": (
                f"**Retention Specialist**\n\n{text}\n\n"
                f"---\nConfidence: {state['confidence']:.0%} | Data class: CONFIDENTIAL"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }


# ── Demo runner ───────────────────────────────────────────────────────────────

def _hr(label: str = "") -> None:
    width = 72
    if label:
        pad = (width - len(label) - 2) // 2
        print(f"\n{'─' * pad} {label} {'─' * (width - pad - len(label) - 2)}")
    else:
        print("─" * width)


def _print_result(idx: int, msg: str, result: dict) -> None:
    tier_label = {1: "Tier 1 — Direct", 2: "Tier 2 — Reflect", 3: "Tier 3 — Escalate"}.get(
        result.get("tier", 0), "Unknown"
    )
    print(f"\n[{idx}] {msg[:60]!r}")
    print(f"     Intent      : {result.get('intent')}")
    print(f"     Confidence  : {result.get('confidence', 0):.0%}")
    print(f"     Routing     : {tier_label} → {result.get('target_agent')}")
    if result.get("reflection_notes"):
        print(f"     Reflection  : {result['reflection_notes'][:80]}")
    print(f"     Agent reply : {result.get('response', '')[:120].strip()}")
    print(f"     Decision ID : {result.get('agent_decision_id') or result.get('route_decision_id') or 'n/a'}")


def main() -> None:
    _hr("ProofLayer v3 — Multi-Agent Orchestration Demo")

    # ── Bootstrap ──────────────────────────────────────────────────────────
    cfg     = load_config()
    gateway = ProofLayerGateway(cfg)
    memory  = MemoryBridge()
    router  = RouterAgent(cfg, gateway=gateway, memory=memory)

    agents = [
        TriageAgentEN(cfg, gateway),
        TriageAgentAR(cfg, gateway),
        BillingAgent(cfg, gateway),
        FraudDetectionAgent(cfg, gateway),
        TechnicalAgent(cfg, gateway),
        RetentionAgent(cfg, gateway),
    ]
    router.load_agents(agents)

    # ── Demo requests ──────────────────────────────────────────────────────
    _hr("Routing 6 requests across 6 agents")

    # 1. Technical outage — routed to technical-agent (Tier 1)
    r1 = router.route_sync(
        "The platform has been completely down for 3 hours. This is a P1 outage.",
        session_id="demo-session-001",
        customer_id="CUST-5501",
    )
    _print_result(1, "Platform down — P1 outage", r1)
    time.sleep(0.2)

    # 2. Billing error — routed to billing-agent (Tier 1)
    r2 = router.route_sync(
        "I was charged twice for my November invoice. The amount was £149 and I see two entries.",
        session_id="demo-session-002",
        customer_id="CUST-7823",
    )
    _print_result(2, "Double charge on November invoice", r2)
    time.sleep(0.2)

    # 3. Fraud report — routed to fraud-detection-agent (Tier 1)
    r3 = router.route_sync(
        "There is an unauthorised transaction on my account for $340 to an unknown vendor.",
        session_id="demo-session-003",
        customer_id="CUST-3310",
    )
    _print_result(3, "Unauthorised transaction reported", r3)
    time.sleep(0.2)

    # 4. Arabic greeting — routed to triage-agent-ar (Tier 1)
    r4 = router.route_sync(
        "مرحبا، أريد المساعدة من فضلك",
        session_id="demo-session-004",
        customer_id="CUST-9901",
        language="ar",
    )
    _print_result(4, "Arabic greeting (مرحبا)", r4)
    time.sleep(0.2)

    # 5. Cross-agent demo: billing then cancellation (same session)
    #    billing-agent resolves double charge → retention-agent sees the context
    _hr("Cross-agent authority demo (session demo-session-005)")

    print("\n[5a] First message: billing error (billing-agent)")
    r5a = router.route_sync(
        "I was double-charged this month and I'm really frustrated.",
        session_id="demo-session-005",
        customer_id="CUST-4421",
    )
    _print_result(5, "Double-charge + frustration", r5a)
    time.sleep(0.2)

    print("\n[5b] Second message: cancellation (retention-agent sees billing context)")
    r5b = router.route_sync(
        "I've had enough. I want to cancel my account.",
        session_id="demo-session-005",  # same session → retention sees billing history
        customer_id="CUST-4421",
    )
    _print_result(6, "Cancellation request (same session)", r5b)
    time.sleep(0.2)

    # 6. Password reset — routed to triage-agent-en (Tier 1)
    r6 = router.route_sync(
        "I can't log into my account. My password doesn't work.",
        session_id="demo-session-006",
        customer_id="CUST-1122",
    )
    _print_result(7, "Password login issue", r6)

    # ── Cross-agent governance output ──────────────────────────────────────
    _hr("Cross-Agent Session Summary")

    summary = memory.get_session_summary("demo-session-005")
    print(f"\n  Session       : {summary['session_id']}")
    print(f"  Customer      : {summary['customer_id']}")
    print(f"  Language      : {summary['language']}")
    print(f"  Duration      : {summary['duration_s']}s")
    print(f"  Agents involved ({len(summary['agents_involved'])}):")
    for ag in summary["agents_involved"]:
        print(f"    - {ag}")
    print("\n  Decision trail:")
    for d in summary["decisions"]:
        did = (d["decision_id"] or "n/a")[:16]
        print(
            f"    [{d['agent']:<25}] {d['intent']:<28} "
            f"{d['confidence']:.0%}  id={did}"
        )

    # ── ProofLayer governance dashboard ───────────────────────────────────
    _hr("ProofLayer Governance Snapshot")
    try:
        report = gateway.governance_report()
        ov = report.get("overview", {})
        print(f"\n  Total decisions  : {ov.get('total_decisions', 'n/a')}")
        print(f"  PII flagged      : {ov.get('pii_flagged', 'n/a')}")
        print(f"  Open exceptions  : {report.get('open_exceptions', 'n/a')}")
        print(f"  Cross-agent edges: {report.get('cross_agent_edges', 'n/a')}")
    except Exception as exc:
        print(f"\n  [ProofLayer unreachable — start the API container] {exc}")

    _hr("Demo complete")
    print(
        "\n  Admin dashboard : http://localhost:8000/ui/admin.html\n"
        "  All decisions, trace steps, and cross-agent edges are recorded\n"
        "  in the ProofLayer graph for the CISO governance dashboard.\n"
    )


if __name__ == "__main__":
    main()
