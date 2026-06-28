"""Cross-agent session memory bridge.

Tracks which specialist agents have handled a given session and what
they concluded. The router injects this context into each agent so
they can make authority-aware decisions (e.g. retention-agent knows
billing-agent already resolved a double-charge before seeing a cancel).

Backed by shared.memory (module-level singleton, same store both agents use).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


# ── Per-turn agent decision record ────────────────────────────────────────────

@dataclass
class AgentDecision:
    agent_name: str
    agent_group: str
    intent: str
    confidence: float
    decision_id: str | None
    timestamp: float = field(default_factory=time.time)

    def to_context_line(self) -> str:
        ts = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        did = f" [{self.decision_id[:8]}]" if self.decision_id else ""
        return (
            f"[{ts}] {self.agent_name} ({self.agent_group})"
            f" → intent={self.intent} confidence={self.confidence:.0%}{did}"
        )


# ── Cross-agent session ───────────────────────────────────────────────────────

@dataclass
class CrossAgentSession:
    session_id:  str
    customer_id: str | None
    language:    str
    decisions:   list[AgentDecision] = field(default_factory=list)
    created_at:  float = field(default_factory=time.time)

    def agents_involved(self) -> list[str]:
        seen: list[str] = []
        for d in self.decisions:
            if d.agent_name not in seen:
                seen.append(d.agent_name)
        return seen

    def has_agent(self, agent_name: str) -> bool:
        return any(d.agent_name == agent_name for d in self.decisions)

    def last_decision_by(self, agent_name: str) -> AgentDecision | None:
        for d in reversed(self.decisions):
            if d.agent_name == agent_name:
                return d
        return None

    def context_for_agent(self, requesting_agent: str) -> str:
        """Build a context block describing what previous agents did.

        Excludes the requesting agent's own prior decisions so it doesn't
        bias reflection — the agent sees what OTHERS concluded, not itself.
        """
        prior = [d for d in self.decisions if d.agent_name != requesting_agent]
        if not prior:
            return ""
        lines = [
            "## Cross-Agent Session Context",
            f"Session: {self.session_id}  language: {self.language}",
            "",
            "Prior agent decisions:",
        ]
        for d in prior:
            lines.append(f"  {d.to_context_line()}")
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        return {
            "session_id":      self.session_id,
            "customer_id":     self.customer_id,
            "language":        self.language,
            "agents_involved": self.agents_involved(),
            "total_decisions": len(self.decisions),
            "duration_s":      round(time.time() - self.created_at, 2),
            "decisions": [
                {
                    "agent":       d.agent_name,
                    "group":       d.agent_group,
                    "intent":      d.intent,
                    "confidence":  round(d.confidence, 3),
                    "decision_id": d.decision_id,
                }
                for d in self.decisions
            ],
        }


# ── MemoryBridge ──────────────────────────────────────────────────────────────

class MemoryBridge:
    """Singleton-safe cross-agent session store.

    Usage:
        bridge = MemoryBridge()
        bridge.start_session("s1", customer_id="C42", language="en")
        bridge.add_agent_decision("s1", "billing-agent", "finance", "billing_error", 0.92, "uuid...")
        ctx = bridge.get_cross_agent_context("s1", "retention-agent")
        print(ctx)  # "Prior agent decisions: [billing-agent → ...]"
    """

    def __init__(self) -> None:
        self._sessions: dict[str, CrossAgentSession] = {}

    # ── Session lifecycle ─────────────────────────────────────────────────

    def start_session(
        self,
        session_id: str,
        customer_id: str | None = None,
        language: str = "en",
    ) -> CrossAgentSession:
        """Create or return an existing cross-agent session."""
        if session_id not in self._sessions:
            self._sessions[session_id] = CrossAgentSession(
                session_id=session_id,
                customer_id=customer_id,
                language=language,
            )
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> CrossAgentSession | None:
        return self._sessions.get(session_id)

    def session_exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    # ── Decision tracking ─────────────────────────────────────────────────

    def add_agent_decision(
        self,
        session_id: str,
        agent_name: str,
        agent_group: str,
        intent: str,
        confidence: float,
        decision_id: str | None = None,
    ) -> None:
        """Append an agent's decision to the session history."""
        sess = self._sessions.get(session_id)
        if sess is None:
            sess = self.start_session(session_id)
        sess.decisions.append(
            AgentDecision(
                agent_name=agent_name,
                agent_group=agent_group,
                intent=intent,
                confidence=confidence,
                decision_id=decision_id,
            )
        )

    # ── Context retrieval ─────────────────────────────────────────────────

    def get_cross_agent_context(
        self,
        session_id: str,
        requesting_agent: str,
    ) -> str:
        """Return a formatted context block for the requesting agent to see."""
        sess = self._sessions.get(session_id)
        if sess is None:
            return ""
        return sess.context_for_agent(requesting_agent)

    def has_agent_handled(self, session_id: str, agent_name: str) -> bool:
        sess = self._sessions.get(session_id)
        return sess is not None and sess.has_agent(agent_name)

    def get_agents_involved(self, session_id: str) -> list[str]:
        sess = self._sessions.get(session_id)
        return sess.agents_involved() if sess else []

    # ── Governance ────────────────────────────────────────────────────────

    def get_session_summary(self, session_id: str) -> dict[str, Any]:
        sess = self._sessions.get(session_id)
        if sess is None:
            return {"session_id": session_id, "error": "session not found"}
        return sess.summary()

    def all_sessions(self) -> list[dict[str, Any]]:
        return [s.summary() for s in self._sessions.values()]

    def clear_session(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)

    def __len__(self) -> int:
        return len(self._sessions)

    def __repr__(self) -> str:
        return f"<MemoryBridge sessions={len(self)}>"
