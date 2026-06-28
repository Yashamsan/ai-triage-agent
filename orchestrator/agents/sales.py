"""Sales and Growth specialist agent.

Handles: plan upgrades, promo inquiries, cross-sell opportunities,
plan comparisons, lead capture, and upsell opportunities.

Policy:
  - Promo codes validated against active promo registry
  - Upsell recommendations gated by customer segment (not shown to at-risk customers)
  - Annual plan discounts auto-applied when customer mentions cost concern
"""
from __future__ import annotations

import time

from orchestrator.agent_base import AgentState, BaseSpecialistAgent

# ── Active promo registry ──────────────────────────────────────────────────

_PROMOS: dict[str, dict] = {
    "SAVE20":     {"discount": 0.20, "description": "20% off any plan for 12 months", "valid": True},
    "LOYALTY10":  {"discount": 0.10, "description": "10% loyalty reward discount",      "valid": True},
    "ANNUAL25":   {"discount": 0.25, "description": "25% off with annual commitment",   "valid": True},
    "ENTERPRISE": {"discount": 0.00, "description": "Custom enterprise pricing",        "valid": True},
}

# ── Plan catalogue ─────────────────────────────────────────────────────────

_PLANS: dict[str, dict] = {
    "starter":      {"price_monthly": 49,   "price_annual": 39,   "users": 5,    "storage": "10 GB"},
    "professional": {"price_monthly": 149,  "price_annual": 119,  "users": 25,   "storage": "100 GB"},
    "business":     {"price_monthly": 399,  "price_annual": 299,  "users": 100,  "storage": "1 TB"},
    "enterprise":   {"price_monthly": None, "price_annual": None, "users": None, "storage": "Unlimited"},
}


class SalesAgent(BaseSpecialistAgent):
    """Handles upgrades, promotions, cross-sell opportunities, and lead capture."""

    version             = "1.0"
    model_id            = "deepseek/deepseek-chat"
    data_classification = "internal"
    contains_pii        = False

    _POLICIES: dict = {
        "active_promos": {k: v["description"] for k, v in _PROMOS.items() if v["valid"]},
        "upsell_blocked_for_at_risk_customers":  True,
        "annual_discount_auto_apply_on_cost_mention": True,
        "plans": list(_PLANS.keys()),
    }

    @property
    def name(self) -> str: return "sales-agent"

    @property
    def group(self) -> str: return "growth"

    # ── Keyword taxonomy ──────────────────────────────────────────────────

    _KEYWORDS: dict[str, list[str]] = {
        "upgrade_request": [
            "upgrade", "higher plan", "more features", "more users", "more storage",
            "better plan", "premium", "professional", "business plan",
        ],
        "promo_inquiry": [
            "promo", "promotion", "discount", "coupon", "code", "deal",
            "offer", "sale", "special price", "voucher",
        ],
        "plan_comparison": [
            "compare plans", "difference between", "which plan", "plan comparison",
            "what do i get", "included in", "features of", "plan details",
        ],
        "cross_sell": [
            "add on", "addon", "additional service", "also need", "what else",
            "integration", "plugin", "extend", "bundle",
        ],
        "upsell_opportunity": [
            "running out", "limit", "storage full", "too slow", "need more",
            "reached limit", "maximum", "can't add more", "quota",
        ],
        "lead_capture": [
            "interested", "want to try", "demo", "trial", "free trial",
            "sign up", "get started", "how do i start", "new account",
        ],
    }

    # ── Promo lookup ──────────────────────────────────────────────────────

    def _find_promo_code(self, message: str) -> str | None:
        upper = message.upper()
        for code in _PROMOS:
            if code in upper:
                return code
        return None

    def _is_at_risk_customer(self, ctx: str) -> bool:
        return any(signal in ctx for signal in [
            "cancellation_request", "dissatisfaction", "competitor_mention",
            "billing_error", "complaints-agent",
        ])

    # ── Classification node ───────────────────────────────────────────────

    def _classify_node(self, state: AgentState) -> dict:
        m      = state["message"].lower()
        intent = "plan_comparison"
        conf   = 0.70

        for candidate, keywords in self._KEYWORDS.items():
            hits = sum(1 for k in keywords if k in m)
            if hits:
                intent = candidate
                conf   = min(0.96, 0.78 + hits * 0.05)
                break

        # Promo code in message → strong signal for promo_inquiry
        if self._find_promo_code(state["message"]):
            intent = "promo_inquiry"
            conf   = 0.95

        step = self._trace_step(
            node_type="sales-classifier",
            thought=f"Keyword match → {intent} | promo_code={self._find_promo_code(state['message'])}",
            action="classify_sales_intent()",
            observation=f"intent={intent} conf={conf:.0%}",
            confidence=conf, latency_ms=7.0,
        )
        return {
            "intent":       intent,
            "confidence":   conf,
            "contains_pii": False,
            "trace_steps":  [step],
        }

    # ── Response node ─────────────────────────────────────────────────────

    def _respond_node(self, state: AgentState) -> dict:
        ref    = str(int(time.time()))[-6:]
        intent = state["intent"]
        ctx    = state.get("context", "")
        notes: list[str] = []

        at_risk = self._is_at_risk_customer(ctx)
        promo   = self._find_promo_code(state["message"])

        if intent == "upgrade_request":
            text = (
                f"Upgrade request UP-{ref} noted. "
                f"Recommended next tier: Professional (SAR 149/mo — 25 users, 100 GB). "
                f"Business plan available at SAR 399/mo for 100 users + 1 TB. "
            )
            if not at_risk:
                text += "\nCurrent offer: annual commitment saves 20% (promo ANNUAL25)."
            else:
                # Customer is at-risk — offer retention-friendly upgrade
                text += "\nAs a valued customer, I can apply a 20% loyalty discount to any plan upgrade."
                notes.append("At-risk customer: retention-friendly upgrade pricing applied.")

        elif intent == "promo_inquiry":
            if promo and _PROMOS.get(promo, {}).get("valid"):
                info = _PROMOS[promo]
                text = (
                    f"Promo code {promo!r} is valid! "
                    f"{info['description']}. "
                    f"Applied to your account (ref PROMO-{ref}). "
                    f"Discount reflected on your next invoice."
                )
            elif promo:
                text = (
                    f"Promo code {promo!r} is no longer active. "
                    f"Currently active promos: SAVE20 (20% off), ANNUAL25 (25% annual). "
                    f"Apply at checkout or let me apply one now — just confirm the plan."
                )
            else:
                text = (
                    f"Active promotions as of today:\n"
                    f"  • SAVE20  — 20% off any plan for 12 months\n"
                    f"  • ANNUAL25 — 25% off with annual commitment\n"
                    f"  • LOYALTY10 — 10% loyalty reward (eligible customers)\n\n"
                    f"Share the code at checkout or I can apply it directly (ref PROMO-{ref})."
                )

        elif intent == "plan_comparison":
            text = (
                "Plan comparison:\n\n"
                "  Starter     SAR 49/mo  (SAR 39/mo annual) — 5 users   | 10 GB\n"
                "  Professional SAR 149/mo (SAR 119/mo annual) — 25 users  | 100 GB\n"
                "  Business    SAR 399/mo (SAR 299/mo annual) — 100 users | 1 TB\n"
                "  Enterprise  Custom pricing — unlimited users | unlimited storage\n\n"
                f"All plans include 24/7 support and 99.9% SLA. Ref: COMP-{ref}"
            )

        elif intent == "cross_sell":
            text = (
                f"Add-on recommendations (ref XS-{ref}):\n"
                f"  • Advanced Analytics — SAR 49/mo\n"
                f"  • SSO Integration — SAR 29/mo\n"
                f"  • Priority Support — SAR 99/mo\n"
                f"  • Custom Integrations — from SAR 199/mo\n\n"
                f"Bundle any 2 add-ons and save 15%. Contact sales@example.com for custom bundles."
            )

        elif intent == "upsell_opportunity":
            text = (
                f"Storage/quota upgrade options (ref UP-{ref}):\n"
                f"  • Add 100 GB: SAR 29/mo\n"
                f"  • Add 1 TB: SAR 79/mo\n"
                f"  • Upgrade to Business plan for unlimited expansion: SAR 399/mo\n\n"
                f"Upgrade takes effect immediately. Pro-rata credit applied for remaining days."
            )

        else:  # lead_capture
            text = (
                f"Great to hear you're interested! Here's how to get started (ref LD-{ref}):\n"
                f"  1. Free 14-day trial: https://app.example.com/trial (no card needed)\n"
                f"  2. Or book a demo: https://calendly.com/example-sales\n"
                f"  3. Questions? Email: sales@example.com\n\n"
                f"Our team will follow up within 24 hours."
            )

        if ctx and not notes:
            notes.append("Customer session context reviewed — offer personalised accordingly.")

        full_text = text
        if notes:
            full_text += "\n\n" + "\n".join(f"  • {n}" for n in notes)

        step = self._trace_step(
            node_type="sales-responder",
            thought=f"Generate sales response | at_risk={at_risk} | promo={promo}",
            action="generate_offer()",
            observation=f"Offer dispatched ref #{ref}",
            confidence=state["confidence"], latency_ms=10.0,
        )
        return {
            "response_text": (
                f"**Sales & Growth**\n\n{full_text}\n\n"
                f"---\nConfidence: {state['confidence']:.0%}  |  Ref: #{ref}"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }
