"""Retention specialist agent.

Handles: cancellation requests, downgrade requests, competitor mentions,
customer dissatisfaction, win-back campaigns, and loyalty inquiries.

Cross-agent authority: RetentionAgent queries ALL agents in the session
to build churn probability before generating a save offer. A customer
with unresolved billing + technical issues has a higher churn probability
and receives a more aggressive save offer.

Policy:
  - High-value (LTV > SAR 5,000): up to 3 months free + 30% discount
  - Mid-value  (LTV > SAR 1,000): up to 1 month free + 20% discount
  - Standard:                      10% discount offer
  - Competitor comparison: match or beat with approval flag
  - Win-back (lapsed): 2 months free, no condition
"""
from __future__ import annotations

import time

from orchestrator.agent_base import AgentState, BaseSpecialistAgent

# ── Churn probability signals ──────────────────────────────────────────────

_CHURN_AMPLIFIERS: dict[str, float] = {
    "billing_error":         0.15,
    "payment_failed":        0.10,
    "invoice_dispute":       0.12,
    "service_outage":        0.08,
    "api_error":             0.06,
    "regulatory_complaint":  0.20,
    "safety_concern":        0.25,
    "agent_complaint":       0.18,
    "service_grievance":     0.15,
    "dissatisfaction":       0.12,
    "competitor_mention":    0.18,
}

# ── Save offer tiers ───────────────────────────────────────────────────────

_SAVE_OFFERS: dict[str, dict] = {
    "high_value":   {"discount": 0.30, "free_months": 3,  "threshold_sar": 5_000},
    "medium_value": {"discount": 0.20, "free_months": 1,  "threshold_sar": 1_000},
    "standard":     {"discount": 0.10, "free_months": 0,  "threshold_sar": 0},
}


def _churn_probability(base: float, ctx: str) -> float:
    """Amplify churn probability based on cross-agent signals."""
    prob = base
    for signal, weight in _CHURN_AMPLIFIERS.items():
        if signal in ctx:
            prob = min(0.99, prob + weight)
    return round(prob, 3)


def _customer_segment(session_id: str) -> str:
    seed = sum(ord(c) for c in session_id)
    ltv  = 300 + (seed % 9_000)
    if ltv > 5_000:
        return "high_value"
    if ltv > 1_000:
        return "medium_value"
    return "standard"


class RetentionAgent(BaseSpecialistAgent):
    """Prevents churn by generating personalised save offers with cross-agent context."""

    version             = "1.0"
    model_id            = "deepseek/deepseek-chat"
    contains_pii        = True
    data_classification = "confidential"

    _POLICIES: dict = {
        "save_offers": {seg: {"discount_pct": int(v["discount"] * 100),
                               "free_months":   v["free_months"],
                               "ltv_threshold_sar": v["threshold_sar"]}
                        for seg, v in _SAVE_OFFERS.items()},
        "cross_agent_churn_amplification": True,
        "competitor_match_requires_pricing_approval": True,
        "win_back_free_months": 2,
    }

    @property
    def name(self) -> str: return "retention-agent"

    @property
    def group(self) -> str: return "retention"

    # ── Keyword taxonomy ──────────────────────────────────────────────────

    _KEYWORDS: dict[str, list[str]] = {
        "cancellation_request": [
            "cancel", "cancellation", "close my account", "stop service",
            "terminate", "end my subscription", "quit", "delete my account",
        ],
        "downgrade_request": [
            "downgrade", "cheaper plan", "basic plan", "reduce plan",
            "lower tier", "cost too much", "too expensive", "smaller plan",
        ],
        "competitor_mention": [
            "competitor", "switching to", "going to", "better deal",
            "cheaper alternative", "other provider", "different company",
        ],
        "dissatisfaction": [
            "unhappy", "disappointed", "frustrated", "terrible", "awful",
            "not satisfied", "poor service", "let down", "expected more",
        ],
        "win_back": [
            "come back", "return", "reactivate", "reinstate", "reconsider",
            "sign up again", "re-subscribe", "missed your service",
        ],
        "loyalty_inquiry": [
            "how long", "been a customer", "since when", "loyalty reward",
            "long-term customer", "tenure", "years with you",
        ],
    }

    # ── Churn analysis ────────────────────────────────────────────────────

    def _analyse_churn(
        self, intent: str, message: str, ctx: str
    ) -> tuple[float, str, dict]:
        base_probs = {
            "cancellation_request": 0.75,
            "downgrade_request":    0.45,
            "competitor_mention":   0.60,
            "dissatisfaction":      0.50,
            "win_back":             0.15,
            "loyalty_inquiry":      0.10,
        }
        base  = base_probs.get(intent, 0.30)
        prob  = _churn_probability(base, ctx)
        seg   = _customer_segment(ctx[:20] if ctx else "default")
        offer = _SAVE_OFFERS[seg]
        return prob, seg, offer

    def _unresolved_issues_from_ctx(self, ctx: str) -> list[str]:
        issues = []
        if "billing_error" in ctx or "payment_failed" in ctx:
            issues.append("billing dispute")
        if "service_outage" in ctx or "api_error" in ctx or "connectivity_issue" in ctx:
            issues.append("technical issue")
        if "agent_complaint" in ctx or "service_grievance" in ctx:
            issues.append("complaint on record")
        return issues

    # ── Classification node ───────────────────────────────────────────────

    def _classify_node(self, state: AgentState) -> dict:
        m      = state["message"].lower()
        intent = "cancellation_request"
        conf   = 0.72

        for candidate, keywords in self._KEYWORDS.items():
            hits = sum(1 for k in keywords if k in m)
            if hits:
                intent = candidate
                conf   = min(0.97, 0.78 + hits * 0.05)
                break

        # Cross-agent context raises confidence (we know more about this customer)
        ctx = state.get("context", "")
        if ctx:
            prob, seg, _ = self._analyse_churn(intent, state["message"], ctx)
            conf         = min(0.99, conf + 0.04)

        step = self._trace_step(
            node_type="retention-classifier",
            thought=f"Churn signal: {intent} | cross-agent ctx={'yes' if ctx else 'no'}",
            action="classify_churn_signal()",
            observation=f"intent={intent} conf={conf:.0%}",
            confidence=conf, latency_ms=11.0,
        )
        return {
            "intent":       intent,
            "confidence":   conf,
            "contains_pii": True,
            "trace_steps":  [step],
        }

    # ── Response node ─────────────────────────────────────────────────────

    def _respond_node(self, state: AgentState) -> dict:
        ref    = str(int(time.time()))[-6:]
        intent = state["intent"]
        ctx    = state.get("context", "")

        churn_prob, segment, offer = self._analyse_churn(intent, state["message"], ctx)
        unresolved = self._unresolved_issues_from_ctx(ctx)
        notes: list[str] = []

        # ── Acknowledge cross-agent context ──────────────────────────────
        preamble = ""
        if unresolved:
            preamble = (
                f"I can see you've had some recent issues with us "
                f"({', '.join(unresolved)}). I sincerely apologise for those. "
            )

        # ── Intent-specific save offer ────────────────────────────────────
        if intent == "cancellation_request":
            disc_pct  = int(offer["discount"] * 100)
            free_mo   = offer["free_months"]
            offer_str = f"{disc_pct}% discount for 12 months"
            if free_mo > 0:
                offer_str = f"{free_mo} month{'s' if free_mo > 1 else ''} FREE + {disc_pct}% discount"
            text = (
                f"{preamble}"
                f"Before we process your cancellation (ref SAVE-{ref}), we'd like to offer:\n\n"
                f"  **{offer_str}** — no contract extension required\n\n"
                f"Can I apply this to your account now? Our churn analysis shows "
                f"{churn_prob:.0%} churn probability — we really want to make this right."
            )
            if segment == "high_value":
                notes.append(f"High-value customer: {offer['free_months']} months free applied (requires finance sign-off).")

        elif intent == "downgrade_request":
            text = (
                f"{preamble}"
                f"Downgrade request DR-{ref} noted. Before we downgrade, I can offer:\n"
                f"  • 20% discount on your current plan for 6 months\n"
                f"  • Or a flexible mid-tier option at SAR 99/mo (15 users, 50 GB)\n\n"
                f"Which would you prefer? The discount keeps all your current features."
            )

        elif intent == "competitor_mention":
            text = (
                f"{preamble}"
                f"I understand you're considering alternatives (ref COMP-{ref}). "
                f"I can offer a custom pricing package to match or beat any competitor quote. "
                f"Please share the competitor's offer details and I'll escalate to our pricing team "
                f"for a personalised match within 24 hours."
            )
            notes.append("Competitor quote requested — pricing team approval required for match/beat.")

        elif intent == "dissatisfaction":
            text = (
                f"{preamble}"
                f"Your feedback is very important to us (ref DISS-{ref}). "
                f"A Customer Success Manager has been assigned to your account and will "
                f"contact you within 4 hours with a personalised resolution plan. "
                f"As an immediate gesture, I've applied a SAR 50 account credit."
            )

        elif intent == "win_back":
            text = (
                f"Welcome back! We're so glad to hear from you (ref WB-{ref}).\n\n"
                f"Returning customer offer:\n"
                f"  • 2 months FREE on any plan (no strings)\n"
                f"  • All your previous data and settings restored\n"
                f"  • Upgraded to your previous plan at old pricing\n\n"
                f"Shall I reactivate your account now?"
            )

        else:  # loyalty_inquiry
            text = (
                f"Thank you for being a loyal customer (ref LYL-{ref})! "
                f"As a long-term member, you qualify for our loyalty rewards:\n"
                f"  • 15% permanent discount on your current plan\n"
                f"  • Priority support queue\n"
                f"  • Dedicated account manager (for 3+ year customers)\n\n"
                f"Shall I apply the loyalty discount to your account?"
            )

        if ctx and not notes:
            notes.append(f"Cross-agent context reviewed: churn probability {churn_prob:.0%} | segment: {segment}")

        full_text = text
        if notes:
            full_text += "\n\n" + "\n".join(f"  • {n}" for n in notes)

        step = self._trace_step(
            node_type="retention-responder",
            thought=(
                f"Save offer | segment={segment} | churn_prob={churn_prob:.0%} "
                f"| unresolved={unresolved}"
            ),
            action="generate_save_offer()",
            observation=f"Offer dispatched | churn_prob={churn_prob:.0%}",
            confidence=state["confidence"], latency_ms=20.0,
        )
        return {
            "response_text": (
                f"**Retention Specialist**\n\n{full_text}\n\n"
                f"---\nChurn Probability: {churn_prob:.0%}  |  "
                f"Segment: {segment.replace('_', ' ').title()}  |  "
                f"Confidence: {state['confidence']:.0%}  |  Ref: SAVE-{ref}"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }
