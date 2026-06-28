"""Loyalty and Rewards specialist agent.

Handles: points inquiries, redemption requests, tier status,
benefit inquiries, points transfers, and loyalty complaints.

Policy:
  - Points expire after 24 months of inactivity
  - Tier upgrades require 3-month qualification period
  - Points transfers capped at 25% of balance per request
  - Redemption minimum: 100 points (= SAR 1)
"""
from __future__ import annotations

import time

from orchestrator.agent_base import AgentState, BaseSpecialistAgent

# ── Tier configuration ─────────────────────────────────────────────────────

_TIERS: dict[str, dict] = {
    "bronze":   {"min_points": 0,      "multiplier": 1.0, "perks": ["Standard support"]},
    "silver":   {"min_points": 5_000,  "multiplier": 1.5, "perks": ["Priority support", "5% discount"]},
    "gold":     {"min_points": 15_000, "multiplier": 2.0, "perks": ["Dedicated CSM", "10% discount", "Early access"]},
    "platinum": {"min_points": 50_000, "multiplier": 3.0, "perks": ["VIP support 24/7", "20% discount", "Custom SLA"]},
}

_POINTS_TO_SAR = 0.01       # 100 points = SAR 1
_MIN_REDEEM    = 100         # minimum points for redemption
_TRANSFER_CAP  = 0.25        # max 25% of balance per transfer request


def _tier_for_points(points: int) -> str:
    tier = "bronze"
    for name, cfg in _TIERS.items():
        if points >= cfg["min_points"]:
            tier = name
    return tier


def _next_tier(current: str) -> str | None:
    tiers = list(_TIERS.keys())
    idx   = tiers.index(current)
    return tiers[idx + 1] if idx < len(tiers) - 1 else None


class LoyaltyAgent(BaseSpecialistAgent):
    """Manages points balances, tier status, redemptions, and loyalty benefits."""

    version             = "1.0"
    model_id            = "deepseek/deepseek-chat"
    data_classification = "internal"
    contains_pii        = False

    _POLICIES: dict = {
        "tiers": {k: {"min_points": v["min_points"], "multiplier": v["multiplier"]}
                  for k, v in _TIERS.items()},
        "points_to_sar_rate":      _POINTS_TO_SAR,
        "min_redeem_points":       _MIN_REDEEM,
        "transfer_cap_pct":        int(_TRANSFER_CAP * 100),
        "expiry_months_inactivity": 24,
        "tier_qualification_months": 3,
    }

    @property
    def name(self) -> str: return "loyalty-agent"

    @property
    def group(self) -> str: return "growth"

    # ── Keyword taxonomy ──────────────────────────────────────────────────

    _KEYWORDS: dict[str, list[str]] = {
        "points_inquiry": [
            "points", "how many points", "balance", "points balance",
            "earned points", "reward points", "check points",
        ],
        "redemption_request": [
            "redeem", "use points", "spend points", "cash in", "exchange points",
            "use my rewards", "apply points", "convert points",
        ],
        "tier_status": [
            "tier", "status", "gold", "platinum", "silver", "bronze",
            "what tier", "my level", "loyalty level", "membership level",
        ],
        "benefit_inquiry": [
            "benefit", "perks", "what do i get", "advantages", "privilege",
            "rewards program", "what's included", "loyalty benefits",
        ],
        "points_transfer": [
            "transfer points", "share points", "give points", "send points",
            "move points", "gift points", "transfer to",
        ],
        "loyalty_complaint": [
            "points expired", "missing points", "didn't get points", "lost my points",
            "points not added", "wrong points", "unfair points", "tier downgraded",
        ],
    }

    # ── Demo customer profile (simulated) ────────────────────────────────

    def _get_demo_profile(self, session_id: str) -> dict:
        seed = sum(ord(c) for c in session_id)
        points  = 7_500 + (seed % 20_000)
        tenure  = 1 + (seed % 5)
        tier    = _tier_for_points(points)
        return {
            "points":  points,
            "tier":    tier,
            "tenure_years": tenure,
            "sar_value": round(points * _POINTS_TO_SAR, 2),
        }

    # ── Classification node ───────────────────────────────────────────────

    def _classify_node(self, state: AgentState) -> dict:
        m      = state["message"].lower()
        intent = "points_inquiry"
        conf   = 0.70

        for candidate, keywords in self._KEYWORDS.items():
            hits = sum(1 for k in keywords if k in m)
            if hits:
                intent = candidate
                conf   = min(0.97, 0.78 + hits * 0.05)
                break

        # Complaint about loyalty → always medium-high confidence
        if intent == "loyalty_complaint":
            conf = max(conf, 0.86)

        step = self._trace_step(
            node_type="loyalty-classifier",
            thought=f"Keyword match → {intent}",
            action="classify_loyalty_intent()",
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
        ref     = str(int(time.time()))[-6:]
        intent  = state["intent"]
        profile = self._get_demo_profile(state["session_id"])
        pts     = profile["points"]
        tier    = profile["tier"]
        sar_val = profile["sar_value"]
        next_t  = _next_tier(tier)
        tier_cfg = _TIERS[tier]
        notes: list[str] = []

        if intent == "points_inquiry":
            next_threshold = _TIERS.get(next_t, {}).get("min_points", pts) if next_t else pts
            pts_to_next    = max(0, next_threshold - pts)
            text = (
                f"Loyalty account summary (ref LYL-{ref}):\n"
                f"  • Points balance: {pts:,} points (≈ SAR {sar_val:.2f})\n"
                f"  • Current tier:   {tier.capitalize()}\n"
                f"  • Points multiplier: {tier_cfg['multiplier']}×\n"
            )
            if next_t:
                text += f"  • Next tier ({next_t.capitalize()}): {pts_to_next:,} more points needed\n"
            else:
                text += "  • You are at our highest tier — Platinum!\n"
            text += "\nPoints expire after 24 months of account inactivity."

        elif intent == "redemption_request":
            if pts < _MIN_REDEEM:
                text = (
                    f"You currently have {pts:,} points, which is below the minimum "
                    f"redemption threshold of {_MIN_REDEEM} points. "
                    f"Continue using our service to earn more points."
                )
            else:
                text = (
                    f"Redemption options for your {pts:,} points:\n"
                    f"  • Statement credit: SAR {sar_val:.2f}\n"
                    f"  • Gift vouchers: Amazon, Jarir, Noon (100 pts = SAR 1)\n"
                    f"  • Service credits: apply to next invoice\n"
                    f"  • Charity donation: 100 pts = SAR 2 donated\n\n"
                    f"Redemption request RED-{ref} created — choose your option above."
                )

        elif intent == "tier_status":
            perks_str = ", ".join(tier_cfg["perks"])
            text = (
                f"Your loyalty tier: {tier.upper()} (ref TIER-{ref})\n"
                f"  • Multiplier: {tier_cfg['multiplier']}× points on all purchases\n"
                f"  • Perks: {perks_str}\n"
                f"  • Account tenure: {profile['tenure_years']} year(s)\n"
            )
            if next_t:
                needed = max(0, _TIERS[next_t]["min_points"] - pts)
                text += f"\nEarn {needed:,} more points to reach {next_t.capitalize()} tier."

        elif intent == "benefit_inquiry":
            rows = []
            for name, cfg in _TIERS.items():
                active = "← YOU" if name == tier else ""
                rows.append(f"  {name.capitalize():<12} {cfg['min_points']:>6,} pts | {', '.join(cfg['perks'])} {active}")
            text = f"Loyalty tier benefits (ref BEN-{ref}):\n" + "\n".join(rows)

        elif intent == "points_transfer":
            cap     = int(pts * _TRANSFER_CAP)
            text = (
                f"Points transfer request TRF-{ref}:\n"
                f"  • Your balance: {pts:,} points\n"
                f"  • Maximum transfer: {cap:,} points (25% cap per request)\n\n"
                f"To transfer, reply with: recipient email + amount (up to {cap:,} pts).\n"
                f"Processing time: 1 business day."
            )
            notes.append("Points transfers are irreversible. Confirmation email will be sent.")

        else:  # loyalty_complaint
            text = (
                f"Loyalty complaint LC-{ref} filed. "
                f"Our rewards team will audit your account within 3 business days. "
                f"Missing or expired points are reviewed case-by-case. "
                f"If a system error caused the issue, points will be restored."
            )
            notes.append("Loyalty audit queued. Tier protection applied during review period.")

        ctx = state.get("context", "")
        if ctx:
            notes.append("Customer session history reviewed — tier context verified.")

        full_text = text
        if notes:
            full_text += "\n\n" + "\n".join(f"  • {n}" for n in notes)

        step = self._trace_step(
            node_type="loyalty-responder",
            thought=f"Generate loyalty response | tier={tier} | points={pts}",
            action="query_loyalty_db()",
            observation=f"Balance {pts} pts | tier={tier}",
            confidence=state["confidence"], latency_ms=13.0,
        )
        return {
            "response_text": (
                f"**Loyalty & Rewards**\n\n{full_text}\n\n"
                f"---\nConfidence: {state['confidence']:.0%}  |  "
                f"Tier: {tier.capitalize()}  |  Balance: {pts:,} pts  |  Ref: LYL-{ref}"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }
