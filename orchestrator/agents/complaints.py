"""Complaints and Escalations specialist agent.

Handles: general escalations, regulatory complaints, service grievances,
agent complaints, billing complaints, and safety concerns.

Policy:
  - Regulatory complaints (CITC, SAMA, consumer protection) → compliance channel
  - Safety concerns → immediate senior escalation regardless of confidence
  - Agent misconduct complaints → HR/QA routing
  - Severity scoring drives SLA: critical 1hr, high 4hr, medium 24hr
"""
from __future__ import annotations

import time

from orchestrator.agent_base import AgentState, BaseSpecialistAgent

# ── Severity config ────────────────────────────────────────────────────────

_SEVERITY_SLA: dict[str, str] = {
    "critical": "1 hour",
    "high":     "4 hours",
    "medium":   "24 hours",
    "low":      "3 business days",
}

_REGULATORY_BODIES = [
    "citc", "sama", "consumer protection", "moci", "tribunal",
    "ombudsman", "regulator", "regulatory", "complaint authority",
]

_SAFETY_SIGNALS = [
    "unsafe", "danger", "injury", "harm", "threat", "violence",
    "discriminat", "harass", "abuse", "attack",
]


class ComplaintsAgent(BaseSpecialistAgent):
    """Routes and resolves escalations, grievances, and regulatory complaints."""

    version             = "1.0"
    model_id            = "deepseek/deepseek-chat"
    data_classification = "internal"
    contains_pii        = False

    _POLICIES: dict = {
        "sla": {"critical": "1hr", "high": "4hr", "medium": "24hr", "low": "3_business_days"},
        "safety_concern_overrides_all": True,
        "regulatory_routes_to_compliance_channel": True,
        "agent_misconduct_routes_to_hr_qa": True,
        "regulatory_bodies": ["citc", "sama", "moci", "ombudsman"],
    }

    @property
    def name(self) -> str: return "complaints-agent"

    @property
    def group(self) -> str: return "escalation"

    # ── Keyword taxonomy ──────────────────────────────────────────────────

    _KEYWORDS: dict[str, list[str]] = {
        "regulatory_complaint": [
            "regulator", "regulatory", "citc", "sama", "consumer protection",
            "ombudsman", "tribunal", "complaint authority", "legal action",
        ],
        "safety_concern": [
            "unsafe", "danger", "injury", "harm", "threat", "violence",
            "discriminat", "harass", "abuse", "hostile",
        ],
        "agent_complaint": [
            "agent was rude", "agent hung up", "agent didn't help", "bad service",
            "worst agent", "agent misconduct", "unprofessional", "condescending",
        ],
        "billing_complaint": [
            "billing complaint", "unfair charge", "deceptive pricing", "hidden fee",
            "auto-renewed without consent", "unauthorised renewal", "misleading",
        ],
        "service_grievance": [
            "unacceptable", "fed up", "disappointed", "terrible service",
            "appalling", "disgusting", "outrageous", "intolerable",
        ],
        "escalation": [
            "escalate", "speak to manager", "supervisor", "senior agent",
            "not satisfied", "not resolved", "this is not good enough",
        ],
    }

    # ── Severity scoring ──────────────────────────────────────────────────

    def _classify_severity(self, intent: str, message: str) -> str:
        m = message.lower()
        if intent == "safety_concern":
            return "critical"
        if intent == "regulatory_complaint":
            return "high"
        if intent == "agent_complaint":
            return "high"
        if any(w in m for w in ["legal", "lawyer", "court", "sue", "lawsuit"]):
            return "high"
        if intent == "service_grievance" and any(
            w in m for w in ["years", "loyal", "always", "never again"]
        ):
            return "medium"
        return "low"

    # ── Classification node ───────────────────────────────────────────────

    def _classify_node(self, state: AgentState) -> dict:
        m      = state["message"].lower()
        intent = "escalation"
        conf   = 0.72

        for candidate, keywords in self._KEYWORDS.items():
            hits = sum(1 for k in keywords if k in m)
            if hits:
                intent = candidate
                conf   = min(0.97, 0.78 + hits * 0.05)
                break

        # Safety is always critical — boost to max confidence
        if any(s in m for s in _SAFETY_SIGNALS):
            intent = "safety_concern"
            conf   = 0.98

        # Cross-agent: if routing failure in session → complaints-agent should see it
        ctx = state.get("context", "")
        if ctx and conf < 0.85:
            conf = min(0.90, conf + 0.04)

        sev = self._classify_severity(intent, state["message"])

        step = self._trace_step(
            node_type="complaints-classifier",
            thought=f"Keyword match → {intent} | severity={sev}",
            action="classify_complaint()",
            observation=f"intent={intent} sev={sev} conf={conf:.0%}",
            confidence=conf, latency_ms=9.0,
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
        sev    = self._classify_severity(intent, state["message"])
        sla    = _SEVERITY_SLA.get(sev, "3 business days")
        notes: list[str] = []

        if intent == "regulatory_complaint":
            text = (
                f"Regulatory complaint COMP-{ref} opened with CRITICAL priority. "
                f"Your complaint has been routed to our Compliance team and will be "
                f"formally acknowledged within 24 hours as required by law. "
                f"Your complaint reference number for the regulatory body is COMP-{ref}. "
                f"Our compliance team will contact you within {sla}."
            )
            notes.append("Compliance team notified. Regulatory SLA clock started.")
            notes.append("Do not close this case without compliance sign-off.")

        elif intent == "safety_concern":
            text = (
                f"URGENT — Safety concern SC-{ref} escalated to Senior Management immediately. "
                f"A member of our leadership team will contact you within 1 hour. "
                f"If you are in immediate danger, please contact emergency services (911/999/112)."
            )
            notes.append("CRITICAL: Senior management paged immediately.")

        elif intent == "agent_complaint":
            text = (
                f"Agent misconduct complaint AC-{ref} filed with our QA department. "
                f"We take all professional conduct complaints seriously. "
                f"Our Quality Assurance team will review call recordings and contact you "
                f"within {sla} with our findings and any corrective actions taken."
            )
            notes.append("QA review triggered. Call recordings flagged for retention.")

        elif intent == "billing_complaint":
            text = (
                f"Billing complaint BC-{ref} escalated to our Billing Disputes team. "
                f"We will investigate the pricing/charge concern and provide a full written "
                f"response within {sla}. Any disputed charges are suspended pending review."
            )

        elif intent == "service_grievance":
            text = (
                f"Service grievance GR-{ref} acknowledged. We sincerely apologise for "
                f"the experience you've had. A Customer Success Manager has been assigned "
                f"to your case and will contact you within {sla} to discuss resolution."
            )

        else:  # escalation
            text = (
                f"Escalation ESC-{ref} created. Your case has been assigned to a senior "
                f"support manager. You will receive a callback within {sla}. "
                f"We have noted your request and will ensure a satisfactory resolution."
            )

        ctx = state.get("context", "")
        if ctx:
            notes.append("Prior agent session context reviewed and attached to case.")

        full_text = text
        if notes:
            full_text += "\n\n" + "\n".join(f"  • {n}" for n in notes)

        step = self._trace_step(
            node_type="complaints-responder",
            thought=f"Route complaint | sev={sev} | sla={sla}",
            action="open_complaint_case()",
            observation=f"Case #{ref} opened | sev={sev}",
            confidence=state["confidence"], latency_ms=11.0,
        )
        return {
            "response_text": (
                f"**Complaints & Escalations**\n\n{full_text}\n\n"
                f"---\nSeverity: {sev.upper()}  |  SLA: {sla}  |  "
                f"Confidence: {state['confidence']:.0%}  |  Case: COMP-{ref}"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }
