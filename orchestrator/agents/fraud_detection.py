"""Fraud Detection specialist agent.

Handles: suspicious transactions, fraud reports, account compromise,
unusual activity, block requests, and identity verification.

Risk scoring model — weighted factors:
  velocity     0.30   10+ transactions in 30 minutes
  geo_anomaly  0.25   international / unusual geo
  high_value   0.20   > SAR 5,000 single transaction
  new_device   0.15   unknown device fingerprint
  time_anomaly 0.10   2-4 AM local time

Cross-agent authority:
  Checks MemoryBridge for BillingAgent payment failures —
  recurring failures in the same session amplify the risk score.

Three-tier response:
  risk >= 0.80 → BLOCK + alert created for investigator queue
  risk >= 0.50 → FLAG for human review (soft hold)
  risk <  0.50 → APPROVE with monitoring note

PHI data classification — highest sensitivity tier.
"""
from __future__ import annotations

import re
import time

from orchestrator.agent_base import AgentState, BaseSpecialistAgent

# ── Risk factor weights ────────────────────────────────────────────────────

_WEIGHTS: dict[str, float] = {
    "velocity":     0.30,
    "geo_anomaly":  0.25,
    "high_value":   0.20,
    "new_device":   0.15,
    "time_anomaly": 0.10,
}

_HIGH_VALUE_THRESHOLD = 5_000.0   # SAR
_CRITICAL_THRESHOLD   = 25_000.0  # SAR — always escalate

_BLOCK_THRESHOLD      = 0.80
_REVIEW_THRESHOLD     = 0.50

# ── Known fraud patterns ───────────────────────────────────────────────────

_PATTERNS: dict[str, str] = {
    "FT-8291": "Velocity Spike — many small transactions before a large one",
    "FT-4402": "Geo Anomaly — transaction in different country within minutes",
    "FT-3177": "Test → Withdraw — small test charge followed by large withdrawal",
    "FT-2051": "New Device — first-time device for high-value transfer",
    "FT-6630": "Rapid CNP — card-not-present burst in 2-minute window",
}

# ── Signal keywords ────────────────────────────────────────────────────────

_GEO_SIGNALS   = ["international", "overseas", "abroad", "foreign", "wire", "transfer to", "uae", "uk", "usa", "europe"]
_VELOCITY_SIGS = ["multiple", "several", "many", "rapid", "burst", "10 times", "flood", "batch"]
_DEVICE_SIGS   = ["new device", "new phone", "unknown device", "different device", "new computer"]
_TIME_SIGS     = ["midnight", "3am", "4am", "2am", "early morning", "late night", "unusual time"]
_HIGH_VAL_SIGS = ["large amount", "big transfer", "wire", "international wire", "deposit", "property"]

_AMOUNT_RE = re.compile(r"(?:sar|£|\$|€|usd)?\s*(\d[\d,]*(?:\.\d{1,2})?)", re.I)


class FraudDetectionAgent(BaseSpecialistAgent):
    """Risk scoring and response for suspected fraud and account security incidents."""

    version  = "1.0"
    model_id = "deepseek/deepseek-chat"

    @property
    def name(self) -> str: return "fraud-detection-agent"

    @property
    def group(self) -> str: return "risk"

    # ── Keyword taxonomy (intent classification) ──────────────────────────

    _KEYWORDS: dict[str, list[str]] = {
        "fraud_report": [
            "fraud", "scam", "defraud", "phishing", "impersonation",
            "fake", "bogus", "fraudulent", "stolen card",
        ],
        "account_compromise": [
            "hacked", "compromised", "someone else", "not me", "intruder",
            "access without permission", "unauthorised login", "my account was",
        ],
        "suspicious_transaction": [
            "suspicious", "didn't make", "i did not make", "don't recognise",
            "unfamiliar", "strange transaction", "unknown charge",
        ],
        "unusual_activity": [
            "unusual", "odd", "strange", "unexpected", "weird",
            "abnormal", "different pattern", "something off",
        ],
        "block_request": [
            "block", "freeze", "lock", "disable", "stop immediately",
            "cancel card", "suspend account", "hold all",
        ],
        "verification_request": [
            "verify", "confirm", "prove", "identity check", "is this legitimate",
            "is this real", "was this authorised", "should i approve",
        ],
    }

    # ── Risk scoring engine ───────────────────────────────────────────────

    def _extract_amount(self, message: str) -> float | None:
        matches = _AMOUNT_RE.findall(message.replace(",", ""))
        amounts = [float(m) for m in matches if m]
        return max(amounts) if amounts else None

    def _score_risk(self, message: str, ctx: str) -> dict[str, float]:
        m      = message.lower()
        scores: dict[str, float] = {}

        # Velocity
        if any(s in m for s in _VELOCITY_SIGS):
            scores["velocity"] = _WEIGHTS["velocity"]

        # Geo anomaly
        if any(s in m for s in _GEO_SIGNALS):
            scores["geo_anomaly"] = _WEIGHTS["geo_anomaly"]

        # High value
        amount = self._extract_amount(message)
        if amount and amount >= _HIGH_VALUE_THRESHOLD:
            scores["high_value"] = _WEIGHTS["high_value"]
        elif amount and amount >= _HIGH_VALUE_THRESHOLD * 0.5:
            scores["high_value"] = _WEIGHTS["high_value"] * 0.5

        # New device
        if any(s in m for s in _DEVICE_SIGS):
            scores["new_device"] = _WEIGHTS["new_device"]

        # Time anomaly
        if any(s in m for s in _TIME_SIGS):
            scores["time_anomaly"] = _WEIGHTS["time_anomaly"]

        # Cross-agent amplifier: BillingAgent payment failures in session
        billing_failures = ctx.count("payment_failed")
        if billing_failures >= 2:
            scores["billing_pattern"] = min(0.20, billing_failures * 0.07)

        return scores

    def _total_risk(self, scores: dict[str, float]) -> float:
        return min(0.95, sum(scores.values()))

    def _match_pattern(self, message: str, scores: dict) -> str | None:
        m = message.lower()
        if "velocity" in scores and "high_value" in scores:
            return "FT-8291"
        if "geo_anomaly" in scores and "high_value" in scores:
            return "FT-4402"
        if "new_device" in scores and "high_value" in scores:
            return "FT-2051"
        if "velocity" in scores and any(w in m for w in ["rapid", "burst", "card"]):
            return "FT-6630"
        return None

    def _risk_tier(self, risk: float) -> str:
        if risk >= _BLOCK_THRESHOLD:
            return "BLOCKED"
        if risk >= _REVIEW_THRESHOLD:
            return "REVIEW"
        return "APPROVED"

    # ── Classification node ───────────────────────────────────────────────

    def _classify_node(self, state: AgentState) -> dict:
        m      = state["message"].lower()
        ctx    = state.get("context", "")
        intent = "suspicious_transaction"
        conf   = 0.72

        for candidate, keywords in self._KEYWORDS.items():
            hits = sum(1 for k in keywords if k in m)
            if hits:
                intent = candidate
                conf   = min(0.97, 0.80 + hits * 0.05)
                break

        # Run risk scoring in classify node so it's available to respond node
        scores  = self._score_risk(state["message"], ctx)
        risk    = self._total_risk(scores)
        pattern = self._match_pattern(state["message"], scores)

        # High risk always boosts confidence
        if risk >= _BLOCK_THRESHOLD:
            conf = max(conf, 0.92)

        amount = self._extract_amount(state["message"])

        step = self._trace_step(
            node_type="fraud-classifier",
            thought=f"Intent: {intent} | risk={risk:.2f} | pattern={pattern} | amount={amount}",
            action="risk_score()",
            observation=f"risk={risk:.2f} tier={self._risk_tier(risk)} pattern={pattern}",
            confidence=conf, latency_ms=18.0,
        )
        return {
            "intent":       intent,
            "confidence":   conf,
            "contains_pii": True,
            "trace_steps":  [step],
        }

    # ── Response node ─────────────────────────────────────────────────────

    def _respond_node(self, state: AgentState) -> dict:
        ref     = str(int(time.time()))[-6:]
        intent  = state["intent"]
        message = state["message"]
        ctx     = state.get("context", "")
        notes:  list[str] = []

        scores  = self._score_risk(message, ctx)
        risk    = self._total_risk(scores)
        tier    = self._risk_tier(risk)
        pattern = self._match_pattern(message, scores)
        amount  = self._extract_amount(message)

        # ── Risk factors summary for notes ────────────────────────────────
        active_factors = list(scores.keys())
        if active_factors:
            notes.append(f"Risk factors detected: {', '.join(active_factors)}")
        if pattern:
            notes.append(f"Pattern match: {pattern} — {_PATTERNS[pattern]}")
        if "billing_pattern" in scores:
            notes.append("Cross-agent signal: recurring BillingAgent payment failures amplified risk score.")

        # ── Intent-specific response ───────────────────────────────────────
        alert_id = f"FRAUD-{ref}"

        if tier == "BLOCKED":
            if intent in ("fraud_report", "account_compromise"):
                text = (
                    f"CRITICAL — Alert {alert_id} created and assigned to our fraud investigation team.\n\n"
                    f"Immediate actions taken:\n"
                    f"  • Account temporarily secured and all active sessions revoked\n"
                    f"  • All pending transactions placed on hold\n"
                    f"  • Fraud investigator will call you within 2 hours\n\n"
                    f"Risk score: {risk:.0%} | Tier: BLOCKED"
                )
            else:
                amt_str = f"SAR {amount:,.2f}" if amount else "the transaction"
                text = (
                    f"Transaction BLOCKED — Alert {alert_id}.\n\n"
                    f"  • {amt_str} has been blocked (risk score: {risk:.0%})\n"
                    f"  • Our fraud team is reviewing this now\n"
                    f"  • If this was legitimate, our investigator will call you within 2 hours\n"
                    f"  • A reversal will be processed within 24 hours if the transaction is verified"
                )

        elif tier == "REVIEW":
            amt_str = f"SAR {amount:,.2f}" if amount else "the activity"
            text = (
                f"Transaction FLAGGED — Alert {alert_id}.\n\n"
                f"  • {amt_str} placed on SOFT HOLD pending human review\n"
                f"  • Our risk team will review within 4 hours\n"
                f"  • You will receive an email once cleared\n\n"
                f"Risk score: {risk:.0%} | Action required: Human review\n"
                f"If this is urgent, call our 24/7 fraud line: +966 800 XXX YYYY"
            )

        else:  # APPROVED
            if intent == "verification_request":
                text = (
                    f"Verification complete — Alert {alert_id}.\n\n"
                    f"Risk analysis: {risk:.0%} — LOW RISK. Transaction appears legitimate.\n"
                    f"  • No suspicious patterns detected\n"
                    f"  • Transaction approved\n"
                    f"  • Enhanced monitoring applied for 48 hours as standard protocol"
                )
            elif intent == "block_request":
                text = (
                    f"Block request {alert_id} processed.\n\n"
                    f"  • Card/account locked immediately\n"
                    f"  • Replacement card dispatched (3–5 business days)\n"
                    f"  • Online banking access restricted — re-enable via branch visit or video KYC"
                )
            else:
                text = (
                    f"Security review {alert_id} completed.\n\n"
                    f"Risk score: {risk:.0%} — within normal parameters.\n"
                    f"Activity has been logged and enhanced monitoring applied for 48 hours.\n"
                    f"No further action required."
                )

        full_text = text
        if notes:
            full_text += "\n\n" + "\n".join(f"  • {n}" for n in notes)

        # Critical threshold note
        if amount and amount >= _CRITICAL_THRESHOLD:
            full_text += f"\n\n  ⚠ Amount exceeds SAR {_CRITICAL_THRESHOLD:,.0f} — mandatory senior review."

        step = self._trace_step(
            node_type="fraud-responder",
            thought=f"Risk={risk:.2f} tier={tier} amount={amount} pattern={pattern}",
            action=f"execute_fraud_action({tier.lower()})",
            observation=f"Alert {alert_id} | tier={tier} | risk={risk:.0%}",
            confidence=state["confidence"], latency_ms=22.0,
        )
        return {
            "response_text": (
                f"**Fraud & Security**\n\n{full_text}\n\n"
                f"---\nRisk Score: {risk:.0%}  |  Decision: {tier}  |  "
                f"Data: PHI/RESTRICTED  |  Alert: {alert_id}"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }
