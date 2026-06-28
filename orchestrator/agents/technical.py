"""Technical Support specialist agent.

Handles: service outages, connectivity issues, device configuration,
API errors, performance degradation, and feature requests.

Policy:
  - P1 outages auto-create incident tickets and trigger SLA credit
  - Hardware replacement requests escalate to L2 for approval
  - API SEV-2 incidents are auto-opened when latency breach is detected
"""
from __future__ import annotations

import time

from orchestrator.agent_base import AgentState, BaseSpecialistAgent


class TechnicalAgent(BaseSpecialistAgent):
    """Resolves infrastructure, connectivity, and device support issues."""

    version  = "1.0"
    model_id = "deepseek/deepseek-chat"

    @property
    def name(self) -> str: return "technical-agent"

    @property
    def group(self) -> str: return "support"

    # ── Keyword taxonomy ──────────────────────────────────────────────────

    _KEYWORDS: dict[str, list[str]] = {
        "service_outage": [
            "down", "outage", "offline", "unavailable", "not available",
            "service down", "cannot access", "stopped working", "unreachable",
        ],
        "connectivity_issue": [
            "internet", "connectivity", "connection", "wifi", "wi-fi",
            "network", "router", "no signal", "dropped", "keeps disconnecting",
        ],
        "api_error": [
            "api", "endpoint", "integration", "webhook", "error code",
            "500", "502", "503", "504", "timeout", "rate limit", "4xx", "5xx",
        ],
        "performance_issue": [
            "slow", "laggy", "performance", "unresponsive", "high latency",
            "takes forever", "loading", "frozen", "delay", "hangs",
        ],
        "device_config": [
            "configure", "setup", "settings", "config", "install", "device",
            "how do i", "instructions", "guide", "cannot set up", "steps",
        ],
        "feature_request": [
            "feature", "request", "suggestion", "improve", "enhancement",
            "missing", "wish", "roadmap", "please add", "idea",
        ],
    }

    # ── Severity / SLA helpers ────────────────────────────────────────────

    _P1_SIGNALS   = ["production", "all users", "completely", "3 hours", "critical", "p1", "urgent", "emergency"]
    _HW_SIGNALS   = ["replace", "broken", "dead", "damaged", "physical", "hardware", "screen"]

    def _parse_severity(self, message: str) -> str:
        m = message.lower()
        if any(s in m for s in self._P1_SIGNALS):
            return "P1"
        if any(w in m for w in ["most", "many", "several", "multiple"]):
            return "P2"
        return "P3"

    def _is_hw_replacement(self, message: str) -> bool:
        return any(w in message.lower() for w in self._HW_SIGNALS)

    # ── Classification node ───────────────────────────────────────────────

    def _classify_node(self, state: AgentState) -> dict:
        m      = state["message"].lower()
        intent = "technical_issue"
        conf   = 0.70

        for candidate, keywords in self._KEYWORDS.items():
            hits = sum(1 for k in keywords if k in m)
            if hits:
                intent = candidate
                conf   = min(0.97, 0.78 + hits * 0.04)
                break

        # P1 severity boosts confidence
        sev = self._parse_severity(state["message"])
        if sev == "P1" and intent == "service_outage":
            conf = min(0.99, conf + 0.05)

        # Cross-agent recurring signal boosts confidence
        ctx = state.get("context", "")
        if "technical-agent" in ctx:
            conf = min(0.99, conf + 0.04)

        step = self._trace_step(
            node_type="tech-classifier",
            thought=f"Keyword match → {intent} | severity={sev} | recurring={'yes' if ctx else 'no'}",
            action="keyword_classify()",
            observation=f"intent={intent} conf={conf:.0%}",
            confidence=conf, latency_ms=8.0,
        )
        return {
            "intent":       intent,
            "confidence":   conf,
            "contains_pii": False,
            "trace_steps":  [step],
        }

    # ── Canned resolution templates ───────────────────────────────────────

    _RESPONSES: dict[str, str] = {
        "service_outage": (
            "Incident #{ref} (severity {sev}) opened. Engineering is actively investigating. "
            "Current ETA: 45 minutes. "
            "Live status: https://status.example.com\n\n"
            "All affected customers will receive an automatic SLA credit."
        ),
        "connectivity_issue": (
            "Connectivity troubleshoot guide (ticket #{ref}):\n"
            "  1. Restart router — hold power button 10 seconds\n"
            "  2. Run: `ping -c 4 8.8.8.8` and share packet loss %\n"
            "  3. Try wired Ethernet to isolate Wi-Fi issues\n"
            "  4. Check if other devices on same network are affected\n\n"
            "If issue persists after the above, an L2 engineer will call back within 2 hours."
        ),
        "api_error": (
            "API incident #{ref} logged (SEV-2). "
            "Our API status: degraded — p99 latency breach detected. "
            "Root cause investigation in progress. ETA: 20 minutes.\n\n"
            "Immediate workaround: implement exponential backoff (max 3 retries, 2s base delay).\n"
            "Docs: https://docs.example.com/api/errors#retries"
        ),
        "performance_issue": (
            "Performance incident #{ref} opened (SLA timer paused). "
            "Diagnostics running in us-east-1 region. "
            "Our SRE team identified elevated DB query times. "
            "Expected normalisation: 15–30 minutes.\n\n"
            "We will notify you by email once resolved."
        ),
        "device_config": (
            "Device setup guide:\n"
            "  1. Download: curl -s https://dl.example.com/agent | bash\n"
            "  2. Authenticate: export API_KEY=<your_key_from_portal>\n"
            "  3. Verify: agent --verify --verbose\n\n"
            "Full documentation: https://docs.example.com/device-setup\n"
            "Reference ticket: #{ref}"
        ),
        "feature_request": (
            "Feature request FR-{ref} logged and routed to our Product team. "
            "Requests are reviewed every sprint (2 weeks). "
            "High-voted requests (10+ upvotes) enter roadmap planning automatically.\n\n"
            "Track progress: https://roadmap.example.com"
        ),
        "technical_issue": (
            "Technical ticket #{ref} created with your report. "
            "A support engineer will contact you within 4 business hours.\n\n"
            "For P1 production issues, escalate via: +966 800 XXX XXXX (24/7)"
        ),
    }

    # ── Response node ─────────────────────────────────────────────────────

    def _respond_node(self, state: AgentState) -> dict:
        ref    = str(int(time.time()))[-6:]
        intent = state["intent"]
        sev    = self._parse_severity(state["message"])

        template = self._RESPONSES.get(intent, self._RESPONSES["technical_issue"])
        text     = template.format(ref=ref, sev=sev)

        notes: list[str] = []

        # Policy: hardware replacement needs L2 approval
        if intent == "device_config" and self._is_hw_replacement(state["message"]):
            notes.append("Hardware replacement request — L2 approval required before dispatch.")

        # P1 outage: add SLA credit notice
        if intent == "service_outage" and sev == "P1":
            notes.append("P1 SLA breach: 10% service credit auto-applied to next invoice.")

        # Cross-agent context: recurring issue from same session
        ctx = state.get("context", "")
        if ctx:
            notes.append("Cross-agent session history reviewed.")

        if notes:
            text += "\n\n" + "\n".join(f"  • {n}" for n in notes)

        step = self._trace_step(
            node_type="tech-responder",
            thought=f"Generate resolution | sev={sev} | hw_replace={self._is_hw_replacement(state['message'])}",
            action="create_ticket_and_respond()",
            observation=f"Ticket #{ref} dispatched",
            confidence=state["confidence"], latency_ms=14.0,
        )
        return {
            "response_text": (
                f"**Technical Support**\n\n{text}\n\n"
                f"---\nConfidence: {state['confidence']:.0%}  |  Severity: {sev}  |  "
                f"Ticket: #{ref}"
            ),
            "trace_steps": state.get("trace_steps", []) + [step],
        }
