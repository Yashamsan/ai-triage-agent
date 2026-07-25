"""Temporal activities — durable wrappers around the existing, UNCHANGED
classify / run_tool / generate_response functions also used by the
LangGraph-based app/agent_graph.py (the current /triage endpoint).

Each activity is a thin wrapper, not new business logic. Temporal persists
each activity's completion in the workflow's event history: if the worker
crashes after an activity finishes but before the workflow records the next
step, the workflow resumes from the *next* activity on restart instead of
re-running the completed one (e.g. never double-creates a support ticket).

These run as plain sync functions — see app/temporal_worker.py, which
supplies a ThreadPoolExecutor so the Temporal SDK can run them off the
asyncio event loop (classify/run_tool do blocking LLM + DB calls).
"""
from __future__ import annotations

from dataclasses import dataclass

from temporalio import activity

from app.classifier import classify
from app.response_generator import generate_response
from app.tools import run_tool


@dataclass
class ClassifyResult:
    intent: str
    confidence: float
    needs_escalation: bool


@dataclass
class ToolResult:
    success: bool
    data: str
    resolved: bool


@activity.defn
def classify_activity(message: str) -> ClassifyResult:
    result = classify(message)
    return ClassifyResult(
        intent=result.intent,
        confidence=result.confidence,
        needs_escalation=result.needs_escalation,
    )


@activity.defn
def run_tool_activity(intent: str, message: str) -> ToolResult:
    result = run_tool(intent, message)
    return ToolResult(success=result.success, data=result.data, resolved=result.resolved)


@activity.defn
def generate_response_activity(message: str, intent: str, tool_output: str, confidence: float) -> str:
    return generate_response(
        message=message, intent=intent, tool_output=tool_output, confidence=confidence,
    )
