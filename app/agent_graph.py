"""LangGraph StateGraph for ai-triage-agent.

Nodes: classifier → reflect → tool_runner → store_memory → responder/escalation

MCP notes:
  app/mcp_server.py + app/mcp_client.py expose the same tools over MCP stdio
  for external integrations (IDE plugins, Claude Desktop).
  tool_runner_node calls run_tool() synchronously; LangGraph runs sync nodes
  in a thread pool when ainvoke() is used, so the event loop is never blocked.
"""

import operator
from typing import Annotated, Literal, TypedDict

from langfuse import observe
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from app.classifier import classify
from app.reflection import reflect as reflection_check
from app.response_generator import generate_response
from app.tools import run_tool
from shared.memory import get_session
from shared.precedent_store import find_precedent, store_trace

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
    # Level 2 (app/context_engineer.py): precise tool query formulated from
    # intent + extracted params. Unused by tool_runner_node until structured
    # backend tools exist to dispatch it to — threaded through now so adding
    # those later doesn't require touching the graph wiring.
    tool_query: str | None
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
    # ProofLayer Reasoning Memory — each node appends one Thought->Action->Observation step
    trace_steps: Annotated[list[dict], operator.add]


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
        "tool_query": result.tool_query,
        "context_history": context_history,
        "precedent_context": precedent_text,
        "trace_steps": [{
            "node_type": "classifier",
            "thought": "Classify the customer message into one of the known intents.",
            "action": "classify(message)",
            "observation": f"intent={result.intent}, confidence={result.confidence:.2f}, needs_escalation={result.needs_escalation}",
            "confidence": result.confidence,
        }],
    }


@observe(name="reflect-node")
def reflection_node(state: AgentState) -> dict:
    if state["intent"] in ("greeting", "unknown"):
        return {
            "needs_revision": False,
            "revised_intent": None,
            "revised_confidence": state["confidence"],
            "critique": None,
            "trace_steps": [{
                "node_type": "reflect",
                "thought": "Greeting/unknown intents skip reflection review.",
                "action": "skipped",
                "observation": "no_revision_needed",
                "confidence": state["confidence"],
            }],
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
            "trace_steps": [{
                "node_type": "reflect",
                "thought": "Review the classifier's output for accuracy before acting on it.",
                "action": "reflection_check(query, classification, confidence, context)",
                "observation": f"revised_intent={revised}, confidence_adjustment={adj:+.2f}, critique={(result.get('critique') or '')[:150]}",
                "confidence": max(0.0, state["confidence"] + adj),
            }],
        }

    return {
        "needs_revision": False,
        "revised_intent": state["intent"],
        "revised_confidence": state["confidence"],
        "critique": None,
        "trace_steps": [{
            "node_type": "reflect",
            "thought": "Review the classifier's output for accuracy before acting on it.",
            "action": "reflection_check(query, classification, confidence, context)",
            "observation": "classification_confirmed_no_revision",
            "confidence": state["confidence"],
        }],
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
        "trace_steps": [{
            "node_type": "tool_runner",
            "thought": f"Run the tool bound to intent '{effective_intent}' to gather grounding information.",
            "action": f"run_tool('{effective_intent}', message)",
            "observation": f"resolved={tool_result.resolved if tool_result else False}, output_preview={(tool_result.data if tool_result else '')[:150]!r}",
        }],
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
    store_outcome = "stored"
    try:
        store_trace(trace)
    except Exception as e:
        store_outcome = f"failed: {e}"
        print(f"[Memory] Failed to store precedent: {e}")

    session = get_session(state.get("session_id") or "default")
    session.add_turn("assistant", f"Classified as: {effective_intent}")
    session.escalation_level = 2 if state["needs_escalation"] else 1

    return {
        "trace_steps": [{
            "node_type": "store_memory",
            "thought": "Persist this turn to session memory and the precedent store for future recall.",
            "action": "store_trace(trace)",
            "observation": store_outcome,
        }],
    }


_INTENT_LABELS = {
    "greeting": "Support Assistant",
    "password_reset": "Password Reset",
    "billing": "Billing",
    "technical_support": "Technical Support",
    "product_inquiry": "Product Inquiry",
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
    return {
        "response_text": response,
        "trace_steps": [{
            "node_type": "responder",
            "thought": f"Synthesize a final response for intent '{effective_intent}' from the retrieved tool output and context.",
            "action": "generate_response(message, intent, tool_output, context_history, precedent_context, confidence, critique)",
            "observation": generated[:200],
            "confidence": state["revised_confidence"],
        }],
    }


@observe(name="escalation-node")
def escalation_node(state: AgentState) -> dict:
    effective_intent = (
        state["revised_intent"] if state.get("needs_revision") else state["intent"]
    )

    # Pause here for a real human decision — resumed via POST /triage/resume
    # with Command(resume=approved). Without this, escalation always
    # auto-completed and /triage/resume was unreachable dead code.
    approved = interrupt({
        "reason": "Escalation requires human approval before a ticket is finalized.",
        "intent": effective_intent,
        "message": state["message"],
        "confidence": state["revised_confidence"],
    })

    if approved:
        response = (
            f"**Escalation Required**\n\n"
            f"**Intent:** {effective_intent.replace('_', ' ').title()}\n"
            f"**Confidence:** {state['revised_confidence']:.0%}\n\n"
            f"{state['tool_output']}\n\n"
            f"---\nA senior support agent will follow up shortly."
        )
        observation = "escalation_approved_ticket_created"
    else:
        response = (
            f"**Escalation Reviewed**\n\n"
            f"A senior agent reviewed this request and it did not require formal escalation. "
            f"Here's what I can help with directly:\n\n"
            f"{state['tool_output']}"
        )
        observation = "escalation_declined_by_human_reviewer"

    if state.get("critique"):
        response += f"\n\n*Reflection note: {state['critique']}*"
    return {
        "response_text": response,
        "trace_steps": [{
            "node_type": "escalation",
            "thought": f"Intent '{effective_intent}' was flagged for human escalation — pause for human approval before finalizing.",
            "action": "interrupt() -> await human decision, then build response from resume value",
            "observation": observation,
            "confidence": state["revised_confidence"],
        }],
    }


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
