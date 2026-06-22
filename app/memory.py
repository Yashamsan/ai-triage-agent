"""Session memory: tracks conversation history per session_id."""

from dataclasses import dataclass, field
from typing import Optional
import time


@dataclass
class ConversationTurn:
    """A single exchange in a conversation."""
    role: str                 # "user" | "assistant"
    content: str
    timestamp: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


@dataclass
class SessionMemory:
    """Tracks a single conversation session."""
    session_id: str
    turns: list[ConversationTurn] = field(default_factory=list)
    current_intent: Optional[str] = None
    escalation_level: Optional[int] = None
    confidence: float = 0.0

    def add_turn(self, role: str, content: str, metadata: dict | None = None):
        self.turns.append(ConversationTurn(
            role=role, content=content,
            timestamp=time.time(), metadata=metadata or {}
        ))

    def recent_context(self, n: int = 3) -> str:
        if not self.turns:
            return ""
        lines = [f"{t.role.upper()}: {t.content}" for t in self.turns[-n:]]
        return "\n".join(lines)

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn_count": len(self.turns),
            "last_topic": self.turns[-1].content if self.turns else None,
            "current_intent": self.current_intent,
            "escalation_level": self.escalation_level,
            "confidence": self.confidence,
        }

    def to_context_block(self) -> str:
        if not self.turns:
            return "[No prior conversation]"
        parts = ["## Conversation History"]
        for i, t in enumerate(self.turns[-5:], 1):
            label = "User" if t.role == "user" else "Agent"
            parts.append(f"{i}. {label}: {t.content[:200]}")
        return "\n".join(parts)


_sessions: dict[str, SessionMemory] = {}


def get_session(session_id: str) -> SessionMemory:
    if session_id not in _sessions:
        _sessions[session_id] = SessionMemory(session_id=session_id)
    return _sessions[session_id]


def session_exists(session_id: str) -> bool:
    return session_id in _sessions and len(_sessions[session_id].turns) > 0


def clear_session(session_id: str):
    _sessions.pop(session_id, None)
