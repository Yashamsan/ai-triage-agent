"""Arabic LangGraph agent — 6-node StateGraph with memory, reflection, and precedent.

Nodes: classifier → reflect → tool_runner → store_memory → responder/escalation

Mirrors app/agent_graph.py Week 7 architecture; Arabic-specific:
  - classify_ar() for Qwen3.5 via OpenRouter
  - app_ar.reflection for bilingual LLM-as-Judge (Arabic query, English intents)
  - Arabic response formatting in responder/escalation nodes
  - Shared memory and precedent_store (sessions prefixed ar_ in main.py)
"""

from typing import Literal, TypedDict

from langfuse import observe
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph

from app_ar.classifier import classify_ar
from app_ar.reflection import reflect as reflection_check
from app_ar.response_generator import generate_response_ar
from app_ar.tools import run_tool
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

@observe(name="classifier-node-ar")
def classifier_node(state: AgentState) -> dict:
    result = classify_ar(state["message"])

    session = get_session(state.get("session_id") or "ar_default")
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
                reason = (p.get("reason") or "لا يوجد سبب مسجل")[:100]
                lines.append(f"- حالة مشابهة سابقة → {decision} ({reason})")
            precedent_text = "## سوابق ذات صلة\n" + "\n".join(lines)
    except Exception:
        pass

    return {
        "intent": result.intent,
        "confidence": result.confidence,
        "needs_escalation": result.needs_escalation,
        "context_history": context_history,
        "precedent_context": precedent_text,
    }


@observe(name="reflect-node-ar")
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


@observe(name="tool-runner-node-ar")
def tool_runner_node(state: AgentState) -> dict:
    effective_intent = (
        state["revised_intent"] if state.get("needs_revision") else state["intent"]
    )
    tool_result = run_tool(effective_intent, state["message"])
    return {
        "tool_output": tool_result.data if tool_result else "",
        "resolved": tool_result.resolved if tool_result else False,
    }


@observe(name="store-memory-node-ar")
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
    except Exception:
        pass

    session = get_session(state.get("session_id") or "ar_default")
    session.add_turn("assistant", f"تم التصنيف كـ: {effective_intent}")
    session.escalation_level = 2 if state["needs_escalation"] else 1

    return {}


@observe(name="responder-node-ar")
def responder_node(state: AgentState) -> dict:
    effective_intent = (
        state["revised_intent"] if state.get("needs_revision") else state["intent"]
    )

    intent_labels = {
        "greeting": "مرحباً",
        "password_reset": "إعادة تعيين كلمة المرور",
        "billing": "الفواتير والمدفوعات",
        "technical_support": "الدعم التقني",
        "escalation": "التصعيد",
        "unknown": "غير محدد",
    }
    label = intent_labels.get(effective_intent, effective_intent)

    generated = generate_response_ar(
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
        revision_note = f"\n\n*ملاحظة المراجع: {state['critique']}*"

    response = (
        f"**{label}**\n\n"
        f"{generated}\n\n"
        f"---\n"
        f"الثقة: {state['revised_confidence']:.0%}"
        f"{revision_note}"
    )
    return {"response_text": response}


@observe(name="escalation-node-ar")
def escalation_node(state: AgentState) -> dict:
    effective_intent = (
        state["revised_intent"] if state.get("needs_revision") else state["intent"]
    )

    intent_labels = {
        "greeting": "مرحباً",
        "password_reset": "إعادة تعيين كلمة المرور",
        "billing": "الفواتير والمدفوعات",
        "technical_support": "الدعم التقني",
        "escalation": "التصعيد",
        "unknown": "غير محدد",
    }
    label = intent_labels.get(effective_intent, effective_intent)

    response = (
        f"**يتطلب التصعيد**\n\n"
        f"**النوع:** {label}\n"
        f"**الثقة:** {state['revised_confidence']:.0%}\n\n"
        f"{state['tool_output']}\n\n"
        f"---\nسيتواصل معك وكيل دعم أول قريباً."
    )
    if state.get("critique"):
        response += f"\n\n*ملاحظة المراجع: {state['critique']}*"
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

triage_agent_ar = build_triage_agent()
