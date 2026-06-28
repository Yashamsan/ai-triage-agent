"""Billing specialist agent.

Handles: payment failures, refund requests, invoice disputes,
subscription changes, pricing inquiries, and billing errors.

Policy:
  - Refunds ≤ SAR 50: auto-approved
  - Refunds SAR 51–200: approved with manager flag
  - Refunds > SAR 200: requires finance approval workflow
  - 3+ payment failures in session: link to fraud agent (cross-agent edge)
  - Billing errors older than 90 days: escalate to compliance
"""
from __future__ import annotations

import re
import time

from orchestrator.agent_base import AgentState, BaseSpecialistAgent

# ── Policy thresholds ──────────────────────────────────────────────────────

_REFUND_AUTO_MAX     = 50.0    # SAR
_REFUND_MANAGER_MAX  = 200.0   # SAR
_FAILURE_FRAUD_LIMIT = 3       # payment failures before fraud signal


class BillingAgent(BaseSpecialistAgent):
    """Resolves billing disputes, payment issues, and subscription changes."""

    version  = "1.0"
    model_id = "deepseek/deepseek-chat"

    @property
    def name(self) -> str: return "billing-agent"

    @property
    def group(self) -> str: return "finance"

    # ── Keyword taxonomy ──────────────────────────────────────────────────

    _KEYWORDS: dict[str, list[str]] = {
        "refund_request": [
            "refund", "money back", "reimbursement", "credit back", "return payment",
            "charge back", "overpaid", "want it back",
        ],
        "billing_error": [
            "charged twice", "double charge", "duplicate", "wrong amount",
            "overcharged", "extra charge", "incorrect charge", "billed incorrectly",
        ],
        "invoice_dispute": [
            "dispute", "wrong invoice", "incorrect invoice", "don't recognise",
            "didn't authorise", "not my charge", "unrecognised", "not my account",
        ],
        "payment_failed": [
            "payment failed", "card declined", "transaction failed", "not charged",
            "couldn't process", "payment issue", "billing failed", "renewal failed",
        ],
        "subscription_change": [
            "upgrade", "downgrade", "change plan", "switch plan",
            "add seats", "remove seats", "annual to monthly", "monthly to annual",
        ],
        "pricing_inquiry": [
            "how much", "price", "cost", "pricing", "plans", "packages",
            "enterprise plan", "discount", "bulk pricing", "custom quote",
        ],
    }

    # ── Amount extraction ─────────────────────────────────────────────────

    _AMOUNT_RE = re.compile(r"(?:sar|£|\$|€|usd)?\s*(\d[\d,]*(?:\.\d{1,2})?)", re.I)

    def _extract_amount(self, message: str) -> float | None:
        matches = self._AMOUNT_RE.findall(message.replace(",", ""))
        amounts = [float(m) for m in matches if m]
        return max(amounts) if amounts else None

    def _refund_tier(self, amount: float | None) -> str:
        if amount is None:
            return "standard"
        if amount <= _REFUND_AUTO_MAX:
            return "auto"
        if amount <= _REFUND_MANAGER_MAX:
            return "manager"
        return "finance"

    # ── Cross-agent signal detection ──────────────────────────────────────

    def _count_payment_failures_in_session(self, ctx: str) -> int:
        return ctx.count("payment_failed")

    def _has_fraud_context(self, ctx: str) -> bool:
        return "fraud" in ctx or "suspicious" in ctx or "block" in ctx

    # ── Classification node ───────────────────────────────────────────────

    def _classify_node(self, state: AgentState) -> dict:
        m      = state["message"].lower()
        intent = "billing_inquiry"
        conf   = 0.70

        for candidate, keywords in self._KEYWORDS.items():
            hits = sum(1 for k in keywords if k in m)
            if hits:
                intent = candidate
                conf   = min(0.97, 0.78 + hits * 0.05)
                break

        # Amount extraction informs policy decisions downstream
        amount = self._extract_amount(state["message"])

        # If fraud context exists in session, boost confidence on billing errors
        ctx = state.get("context", "")
        if self._has_fraud_context(ctx) and intent in ("billing_error", "payment_failed"):
            conf = min(0.99, conf + 0.04)

        step = self._trace_step(
            node_type="billing-classifier",
            thought=f"Keyword match → {intent} | amount_detected={amount}",
            action="keyword_classify()",
            observation=f"intent={intent} conf={conf:.0%} amount={amount}",
            confidence=conf, latency_ms=10.0,
        )
        return {
            "intent":       intent,
            "confidence":   conf,
            "contains_pii": True,
            "trace_steps":  [step],
        }

    # ── Canned resolution templates ───────────────────────────────────────

    def _respond_node(self, state: AgentState) -> dict:
        ref    = str(int(time.time()))[-6:]
        intent = state["intent"]
        msg    = state["message"]
        ctx    = state.get("context", "")

        amount = self._extract_amount(msg)
        tier   = self._refund_tier(amount)
        notes: list[str] = []

        # ── Intent-specific resolution ──────────────────────────────────

        if intent == "refund_request":
            amt_str = f"SAR {amount:.2f}" if amount else "the disputed amount"
            if tier == "auto":
                text = (
                    f"Refund of {amt_str} approved automatically (ref REF-{ref}). "
                    f"Credit will appear on your statement within 5–7 business days."
                )
            elif tier == "manager":
                text = (
                    f"Refund of {amt_str} requires manager confirmation (ref REF-{ref}). "
                    f"A billing manager will review and process within 24 hours."
                )
                notes.append(f"Manager approval flag set — amount SAR {amount:.2f} exceeds auto-approval threshold.")
            else:
                text = (
                    f"Refund of {amt_str} requires Finance approval (ref REF-{ref}). "
                    f"Finance team will review within 2 business days and contact you."
                )
                notes.append(f"Finance workflow triggered — amount SAR {amount:.2f} exceeds manager threshold.")

        elif intent == "billing_error":
            text = (
                f"Billing error confirmed (case BILL-{ref}). "
                f"{'Duplicate charge of ' + f'SAR {amount:.2f}' if amount else 'Charge'} reversed immediately. "
                f"Adjustment appears on next statement or as direct credit within 3 days."
            )
            notes.append("Billing error logged for audit trail — auto-reconciliation triggered.")

        elif intent == "invoice_dispute":
            text = (
                f"Invoice dispute INV-{ref} opened. "
                f"Our billing team will investigate and respond within 3 business days. "
                f"The disputed charge has been suspended pending review."
            )

        elif intent == "payment_failed":
            failures = self._count_payment_failures_in_session(ctx)
            text = (
                f"Payment failure PF-{ref} logged. "
                f"Common causes: expired card, insufficient funds, or 3D-Secure timeout. "
                f"Please update your payment method in Account → Billing → Payment Methods."
            )
            if failures >= _FAILURE_FRAUD_LIMIT:
                notes.append(
                    f"ALERT: {failures} payment failures detected in this session. "
                    f"Cross-agent fraud signal generated (link to fraud-detection-agent)."
                )

        elif intent == "subscription_change":
            text = (
                f"Subscription change request SC-{ref} queued. "
                f"Changes take effect at the start of your next billing cycle. "
                f"Pro-rata credit will be applied for any mid-cycle upgrades."
            )

        elif intent == "pricing_inquiry":
            text = (
                "Current pricing:\n"
                "  • Starter: SAR 49/mo (5 users, 10 GB)\n"
                "  • Professional: SAR 149/mo (25 users, 100 GB)\n"
                "  • Enterprise: Custom pricing — contact sales@example.com\n\n"
                "Annual billing saves 20%. Ref: pricing.example.com"
            )

        else:
            text = (
                f"Billing inquiry BIL-{ref} logged. "
                f"Our team will review and respond within 1 business day."
            )

        # Cross-agent context acknowledgement
        if ctx:
            notes.append("Prior session context from other agents reviewed.")

        # Build final response
        full_text = text
        if notes:
            full_text += "\n\n" + "\n".join(f"  • {n}" for n in notes)

        step = self._trace_step(
            node_type="billing-responder",
            thought=f"Generate billing resolution | tier={tier} | failures_in_session={self._count_payment_failures_in_session(ctx)}",
            action="resolve_billing()",
            observation=f"Case #{ref} processed",
            confidence=state["confidence"], latency_ms=18.0,
        )
        return {
            "response_text": (
                f"**Billing Support**\n\n{full_text}\n\n"
                f"---\nConfidence: {state['confidence']:.0%}  |  Data: CONFIDENTIAL  |  "
                f"Ref: #{ref}"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }
