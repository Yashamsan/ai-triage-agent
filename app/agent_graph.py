"""LangGraph StateGraph for ai-triage-agent.

Nodes: classifier → reflect → tool_runner → store_memory → responder/escalation

MCP notes:
  app/mcp_server.py + app/mcp_client.py expose the same tools over MCP stdio
  for external integrations (IDE plugins, Claude Desktop).
  tool_runner_node calls run_tool() synchronously; LangGraph runs sync nodes
  in a thread pool when ainvoke() is used, so the event loop is never blocked.
"""

from typing import Literal, TypedDict

from langfuse import observe
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from app.classifier import classify
from app.memory import get_session
from app.precedent_store import find_precedent, store_trace
from app.reflection import reflect as reflection_check
from app.response_generator import generate_response
from app.tools import run_tool

checkpointer = InMemorySaver()


# ── State ─────────────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    # Input
    message: str
    session_id: str | None
    # Classifier output
    intent: str
    confidence: float
    needs_escalation: bool
    # Reflection output
    needs_revision: bool
    revised_intent: str | None
    revised_confidence: float
    critique: str | None
    # Tool output
    tool_output: str
    resolved: bool
    # Memory / precedent context
    context_history: str
    precedent_context: str
    # Final output
    response_text: str


# ── Nodes ─────────────────────────────────────────────────────────────────────

@observe(name="classifier-node")
def classifier_node(state: AgentState) -> dict:
    result = classify(state["message"])

    session = get_session(state.get("session_id") or "default")
    session.add_turn("user", state["message"])
    session.current_intent = result.intent
    session.confidence = result.confidence
    context_history = session.to_context_block()

    precedent_text = ""
    try:
        precedents = find_precedent(symptoms=state["message"], top_k=2)
        if precedents:
            lines = []
            for p in precedents:
                decision = p.get("human_correction") or p.get("decision", "unknown")
                reason = (p.get("reason") or "No reason recorded")[:100]
                lines.append(f"- Previous similar case → {decision} ({reason})")
            precedent_text = "## Relevant Precedents\n" + "\n".join(lines)
    except Exception:
        pass

    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "needs_escalation": result.needs_escalation,
        "context_history": context_history,
        "precedent_context": precedent_text,
    }


@observe(name="reflect-node")
def reflection_node(state: AgentState) -> dict:
    if state["intent"] in ("greeting", "unknown"):
        return {
            "needs_revision": False,
            "revised_intent": None,
            "revised_confidence": state["confidence"],
            "critique": None,
        }

    context_parts = []
    if state.get("precedent_context"):
        context_parts.append(state["precedent_context"])
    if state.get("context_history"):
        context_parts.append(state["context_history"])
    combined_context = "\n\n".join(context_parts)

    result = reflection_check(
        query=state["message"],
        classification=state["intent"],
        confidence=state["confidence"],
        context=combined_context,
    )

    if result and result.get("needs_revision"):
        revised = result.get("suggested_intent") or state["intent"]
        adj = result.get("confidence_adjustment", 0.0)
        return {
            "needs_revision": True,
            "revised_intent": revised,
            "revised_confidence": max(0.0, state["confidence"] + adj),
            "critique": result.get("critique", ""),
            "needs_escalation": (
                True if revised == "escalation" else state["needs_escalation"]
            ),
        }

    return {
        "needs_revision": False,
        "revised_intent": state["intent"],
        "revised_confidence": state["confidence"],
        "critique": None,
    }


@observe(name="tool-runner-node")
def tool_runner_node(state: AgentState) -> dict:
    effective_intent = (
        state["revised_intent"] if state.get("needs_revision") else state["intent"]
    )
    tool_result = run_tool(effective_intent, state["message"])
    return {
        "tool_output": tool_result.data if tool_result else "",
        "resolved": tool_result.resolved if tool_result else False,
    }


@observe(name="store-memory-node")
def store_memory_node(state: AgentState) -> dict:
    effective_intent = (
        state["revised_intent"] if state.get("needs_revision") else state["intent"]
    )
    trace = {
        "query": state["message"],
        "classification": effective_intent,
        "confidence": (
            state["revised_confidence"]
            if state.get("needs_revision")
            else state["confidence"]
        ),
        "intent": state["intent"],
        "needs_escalation": state["needs_escalation"],
        "resolved": state["resolved"],
        "needed_revision": state.get("needs_revision", False),
        "critique": state.get("critique"),
        "session_id": state.get("session_id"),
    }
    try:
        store_trace(trace)
    except Exception as e:
        print(f"[Memory] Failed to store precedent: {e}")

    session = get_session(state.get("session_id") or "default")
    session.add_turn("assistant", f"Classified as: {effective_intent}")
    session.escalation_level = 2 if state["needs_escalation"] else 1

    return {}


_INTENT_LABELS = {
    "greeting": "Support Assistant",
    "password_reset": "Password Reset",
    "billing": "Billing",
    "technical_support": "Technical Support",
    "escalation": "Escalation",
    "unknown": "Support Assistant",
}


@observe(name="responder-node")
def responder_node(state: AgentState) -> dict:
    effective_intent = (
        state["revised_intent"] if state.get("needs_revision") else state["intent"]
    )
    label = _INTENT_LABELS.get(effective_intent, effective_intent.replace("_", " ").title())

    generated = generate_response(
        message=state["message"],
        intent=effective_intent,
        tool_output=state.get("tool_output", ""),
        context_history=state.get("context_history", ""),
        precedent_context=state.get("precedent_context", ""),
        confidence=state["revised_confidence"],
        critique=state.get("critique") or "",
    )

    revision_note = ""
    if state.get("needs_revision") and state.get("critique"):
        revision_note = f"\n\n*Reflection note: {state['critique']}*"

    response = (
        f"**{label}**\n\n"
        f"{generated}\n\n"
        f"---\n"
        f"Confidence: {state['revised_confidence']:.0%}"
        f"{revision_note}"
    )
    return {"response_text": response}


@observe(name="escalation-node")
def escalation_node(state: AgentState) -> dict:
    effective_intent = (
        state["revised_intent"] if state.get("needs_revision") else state["intent"]
    )
    response = (
        f"**Escalation Required**\n\n"
        f"**Intent:** {effective_intent.replace('_', ' ').title()}\n"
        f"**Confidence:** {state['revised_confidence']:.0%}\n\n"
        f"{state['tool_output']}\n\n"
        f"---\nA senior support agent will follow up shortly."
    )
    if state.get("critique"):
        response += f"\n\n*Reflection note: {state['critique']}*"
    return {"response_text": response}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_classifier(state: AgentState) -> Literal["reflect"]:
    return "reflect"


def route_after_reflection(
    state: AgentState,
) -> Literal["tool_runner"]:
    return "tool_runner"


def route_after_tool(state: AgentState) -> Literal["store_memory"]:
    return "store_memory"


def route_after_memory(
    state: AgentState,
) -> Literal["responder", "escalation"]:
    # Only escalate when the classifier (or reflection) explicitly flagged it.
    # Unresolved but non-escalated means the agent gave a best-effort answer
    # (e.g. "unknown" guide message) — that goes to responder, not a human.
    if state["needs_escalation"]:
        return "escalation"
    return "responder"


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_triage_agent() -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("classifier", classifier_node)
    workflow.add_node("reflect", reflection_node)
    workflow.add_node("tool_runner", tool_runner_node)
    workflow.add_node("store_memory", store_memory_node)
    workflow.add_node("responder", responder_node)
    workflow.add_node("escalation", escalation_node)

    workflow.set_entry_point("classifier")

    workflow.add_conditional_edges(
        "classifier", route_after_classifier,
        {"reflect": "reflect"},
    )
    workflow.add_conditional_edges(
        "reflect", route_after_reflection,
        {"tool_runner": "tool_runner"},
    )
    workflow.add_conditional_edges(
        "tool_runner", route_after_tool,
        {"store_memory": "store_memory"},
    )
    workflow.add_conditional_edges(
        "store_memory", route_after_memory,
        {"responder": "responder", "escalation": "escalation"},
    )
    workflow.add_edge("responder", END)
    workflow.add_edge("escalation", END)

    return workflow.compile(checkpointer=checkpointer)


# ── Singleton ─────────────────────────────────────────────────────────────────

triage_agent = build_triage_agent()
