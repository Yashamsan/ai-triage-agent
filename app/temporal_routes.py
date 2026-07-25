"""FastAPI routes for the durable (Temporal-backed) triage path.

Additive and separate from the main POST /triage endpoint (app/main.py,
LangGraph-based) — nothing about the existing flow changes. This is a demo
of durable execution: requires docker/docker-compose.temporal.yml running
plus a worker process (python -m app.temporal_worker). If Temporal isn't
reachable, this endpoint fails with a clear 503 rather than silently
falling back to the non-durable path.
"""
from __future__ import annotations

import uuid

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from temporalio.client import Client, WorkflowFailureError

from app.temporal_workflows import TriageAgentWorkflow

router = APIRouter(prefix="/triage", tags=["durable"])

TASK_QUEUE = "triage-queue"
TEMPORAL_ADDRESS = "localhost:7233"

_client: Client | None = None


async def _get_client() -> Client:
    global _client
    if _client is None:
        _client = await Client.connect(TEMPORAL_ADDRESS)
    return _client


class DurableTriageRequest(BaseModel):
    message: str
    session_id: str | None = None


class DurableTriageResponse(BaseModel):
    intent: str
    confidence: float
    needs_escalation: bool
    response: str
    resolved: bool
    workflow_id: str


@router.post("/durable", response_model=DurableTriageResponse)
async def triage_durable(request: DurableTriageRequest) -> DurableTriageResponse:
    if not request.message.strip():
        raise HTTPException(status_code=422, detail="message cannot be empty")

    workflow_id = f"triage-{request.session_id or uuid.uuid4()}"

    try:
        client = await _get_client()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Temporal Server unreachable at {TEMPORAL_ADDRESS}: {exc}",
        ) from exc

    try:
        handle = await client.start_workflow(
            TriageAgentWorkflow.run,
            request.message,
            id=workflow_id,
            task_queue=TASK_QUEUE,
        )
        result: dict = await handle.result()
    except WorkflowFailureError as exc:
        raise HTTPException(status_code=500, detail=f"Workflow failed: {exc}") from exc

    return DurableTriageResponse(**result, workflow_id=workflow_id)


@router.get("/durable/{workflow_id}")
async def triage_durable_status(workflow_id: str) -> dict:
    """Check on (or fetch the result of) a workflow started above, without
    blocking a new request on it — useful if you started one, killed the
    worker, and want to poll rather than hold a connection open."""
    try:
        client = await _get_client()
        handle = client.get_workflow_handle(workflow_id)
        desc = await handle.describe()
        status = desc.status.name if desc.status else "UNKNOWN"
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Workflow not found or Temporal unreachable: {exc}") from exc

    body = {"workflow_id": workflow_id, "status": status}
    if status == "COMPLETED":
        try:
            body["result"] = await handle.result()
        except Exception:
            pass
    return body
