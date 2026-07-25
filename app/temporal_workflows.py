"""TriageAgentWorkflow — durable version of the classify -> tool -> respond
pipeline, executed through Temporal instead of LangGraph's in-memory graph.

This is additive: app/agent_graph.py and the existing /triage endpoint are
untouched. This workflow backs the separate POST /triage/durable endpoint
(app/temporal_routes.py) and is run by app/temporal_worker.py.
"""
from __future__ import annotations

from datetime import timedelta

from temporalio import workflow

with workflow.unsafe.imports_passed_through():
    from app.temporal_activities import (
        ClassifyResult,
        ToolResult,
        classify_activity,
        generate_response_activity,
        run_tool_activity,
    )

_ACTIVITY_TIMEOUT = timedelta(seconds=60)


@workflow.defn
class TriageAgentWorkflow:
    @workflow.run
    async def run(self, message: str) -> dict:
        classify_result: ClassifyResult = await workflow.execute_activity(
            classify_activity, message, start_to_close_timeout=_ACTIVITY_TIMEOUT,
        )

        tool_result: ToolResult = await workflow.execute_activity(
            run_tool_activity,
            args=[classify_result.intent, message],
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
        )

        response_text: str = await workflow.execute_activity(
            generate_response_activity,
            args=[message, classify_result.intent, tool_result.data, classify_result.confidence],
            start_to_close_timeout=_ACTIVITY_TIMEOUT,
        )

        return {
            "intent": classify_result.intent,
            "confidence": classify_result.confidence,
            "needs_escalation": classify_result.needs_escalation,
            "response": response_text,
            "resolved": tool_result.resolved,
        }
