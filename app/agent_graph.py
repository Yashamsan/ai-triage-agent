"""LangGraph agent for the triage pipeline.

3-node StateGraph:
  1. classifier  → classify user message intent
  2. tool_runner  → execute tool based on intent
  3. responder    → format final response or escalate

Routing:
  tool_runner → responder (if resolved)
             → escalate (if unresolved or user requested escalation)
"""

from typing import Literal, TypedDict

from langfuse import observe
from langgraph.graph import END, StateGraph

from app.classifier import classify, ClassifierOutput
from app.tools import run_tool


# ── State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    """Mutable state passed between graph nodes."""
    message: str
    session_id: str | None
    intent: str
    confidence: float
    needs_escalation: bool
    tool_output: str
    resolved: bool
    response_text: str


# ── Nodes ──────────────────────────────────────────────────────────────

@observe(name="classifier-node")
def classifier_node(state: AgentState) -> dict:
    result: ClassifierOutput = classify(state["message"])
    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "needs_escalation": result.needs_escalation,
    }


@observe(name="tool-runner-node")
def tool_runner_node(state: AgentState) -> dict:
    tool_result = run_tool(state["intent"], state["message"])
    return {
        "tool_output": tool_result.data,
        "resolved": tool_result.resolved,
    }


@observe(name="responder-node")
def responder_node(state: AgentState) -> dict:
    response = (
        f"**{state['intent'].replace('_', ' ').title()}**\n\n"
        f"{state['tool_output']}\n\n"
        f"---\n"
        f"Confidence: {state['confidence']:.0%}"
    )
    return {"response_text": response}


@observe(name="escalation-node")
def escalation_node(state: AgentState) -> dict:
    response = (
        "**Escalated to Senior Agent**\n\n"
        "I understand this needs a human touch. I've created a priority ticket "
        "and a senior support agent will reach out within 15 minutes.\n\n"
        f"---\n"
        f"Intent: {state['intent']} | "
        f"Confidence: {state['confidence']:.0%}"
    )
    return {"response_text": response}


# ── Router ─────────────────────────────────────────────────────────────

def route_after_tool(state: AgentState) -> Literal["responder", "escalation"]:
    """Decide next step based on tool result and classification."""
    if state.get("resolved") and not state.get("needs_escalation"):
        return "responder"
    elif state.get("needs_escalation"):
        return "escalation"
    else:
        return "responder"


# ── Graph Builder ──────────────────────────────────────────────────────

def build_triage_agent() -> StateGraph:
    """Build and compile the triage agent graph."""
    workflow = StateGraph(AgentState)

    workflow.add_node("classifier", classifier_node)
    workflow.add_node("tool_runner", tool_runner_node)
    workflow.add_node("responder", responder_node)
    workflow.add_node("escalation", escalation_node)

    workflow.set_entry_point("classifier")

    workflow.add_edge("classifier", "tool_runner")
    workflow.add_conditional_edges(
        "tool_runner",
        route_after_tool,
        {
            "responder": "responder",
            "escalation": "escalation",
        },
    )
    workflow.add_edge("responder", END)
    workflow.add_edge("escalation", END)

    return workflow.compile()


# ── Singleton ──────────────────────────────────────────────────────────

triage_agent = build_triage_agent()
