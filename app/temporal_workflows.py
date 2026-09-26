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

# 60s originally -- too short for this pipeline's real, measured latency:
# classify/run_tool/generate_response each make an LLM call and/or an
# embedding lookup, which on a CPU-constrained host have run anywhere from
# ~8s to 90s+ this session (see shared/embeddings.py, app/classifier.py).
# A too-tight timeout doesn't fail fast here -- it fails *after* the
# activity's real work already completed, visible server-side as "Activity
# not found on completion ... activity already timed out". 180s gives
# comfortable headroom above the worst case measured so far.
_ACTIVITY_TIMEOUT = timedelta(seconds=180)


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
