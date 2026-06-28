"""Abstract base for all specialist agents in the multi-agent orchestrator.

Subclass, override `_classify_node` and `_respond_node`, done.
The base builds a 3-node LangGraph:
    classify → respond → record
and records all decisions to ProofLayer via the injected gateway.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from langgraph.graph import END, StateGraph
from typing import TypedDict

if TYPE_CHECKING:
    from orchestrator.prooflayer_gateway import ProofLayerGateway


# ── Per-agent LangGraph state ─────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    message: str
    session_id: str
    customer_id: str | None
    context: str            # cross-agent context injected by the router
    # Classification output
    intent: str
    confidence: float
    contains_pii: bool
    # Tool / retrieval output
    tool_output: str
    # Response
    response_text: str
    # Governance
    decision_id: str | None
    trace_steps: list[dict]


# ── Return value for router.dispatch ─────────────────────────────────────────

@dataclass
class AgentResponse:
    intent: str
    confidence: float
    response: str
    decision_id: str | None = None
    needs_escalation: bool = False
    contains_pii: bool = False
    trace_steps: list[dict] = field(default_factory=list)


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseSpecialistAgent(ABC):
    """Common scaffold every specialist agent builds on.

    Subclasses MUST implement:
        name          → str (property)
        group         → str (property)
        _classify_node(state: AgentState) → dict
            Must return at minimum: intent, confidence, contains_pii
        _respond_node(state: AgentState) → dict
            Must return at minimum: response_text

    Subclasses MAY override:
        model_id      class attribute
        version       class attribute
        _build_graph  to add nodes (e.g. tool_runner, reflection)
    """

    version:  str = "1.0"
    model_id: str = "deepseek/deepseek-chat"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        gateway: "ProofLayerGateway | None" = None,
    ) -> None:
        self._config  = config or {}
        self._gateway = gateway
        self._graph   = self._build_graph()

    # ── Abstract interface ────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def group(self) -> str: ...

    @abstractmethod
    def _classify_node(self, state: AgentState) -> dict: ...

    @abstractmethod
    def _respond_node(self, state: AgentState) -> dict: ...

    # ── Default 3-node LangGraph ──────────────────────────────────────────

    def _build_graph(self):
        """Build default classify → respond → record graph.

        Override this to inject extra nodes (reflection, tool_runner, etc.)
        following the same pattern as app/agent_graph.py.
        """
        g = StateGraph(AgentState)
        g.add_node("classify", self._classify_node)
        g.add_node("respond",  self._respond_node)
        g.add_node("record",   self._record_node)
        g.set_entry_point("classify")
        g.add_edge("classify", "respond")
        g.add_edge("respond",  "record")
        g.add_edge("record",   END)
        return g.compile()

    # ── ProofLayer recording ──────────────────────────────────────────────

    def _record_node(self, state: AgentState) -> dict:
        """Record agent decision to ProofLayer (best-effort)."""
        if not self._gateway:
            return {"decision_id": None}
        try:
            result = self._gateway.record_decision(
                agent_name=self.name,
                agent_group=self.group,
                model_id=self.model_id,
                decision_value=state["intent"],
                confidence=state["confidence"],
                session_id=state["session_id"],
                contains_pii=state.get("contains_pii", False),
                trace_steps=state.get("trace_steps", []),
            )
            return {"decision_id": result.get("decision_id")}
        except Exception as exc:
            print(f"[{self.name}] ProofLayer record skipped: {exc}")
            return {"decision_id": None}

    # ── Public API ────────────────────────────────────────────────────────

    def process(
        self,
        message: str,
        session_id: str,
        customer_id: str | None = None,
        context: str = "",
    ) -> AgentResponse:
        """Synchronously run the agent graph and return a structured result.

        Called by the router's dispatch node (sync, runs in a thread when the
        router uses ainvoke, so asyncio.run inside here is safe if needed).
        """
        initial: AgentState = {
            "message":      message,
            "session_id":   session_id,
            "customer_id":  customer_id,
            "context":      context,
            "intent":       "unknown",
            "confidence":   0.0,
            "contains_pii": False,
            "tool_output":  "",
            "response_text":"",
            "decision_id":  None,
            "trace_steps":  [],
        }
        final = self._graph.invoke(initial)
        return AgentResponse(
            intent=final["intent"],
            confidence=final["confidence"],
            response=final["response_text"],
            decision_id=final.get("decision_id"),
            needs_escalation=final.get("confidence", 1.0) < 0.50,
            contains_pii=final.get("contains_pii", False),
            trace_steps=final.get("trace_steps", []),
        )

    # ── Helpers for subclasses ────────────────────────────────────────────

    def _trace_step(
        self,
        node_type: str,
        thought: str = "",
        action: str = "",
        observation: str = "",
        confidence: float | None = None,
        latency_ms: float | None = None,
    ) -> dict:
        return {
            "node_type":   node_type,
            "thought":     thought,
            "action":      action,
            "observation": observation,
            "confidence":  confidence,
            "latency_ms":  latency_ms,
        }

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} group={self.group}>"
