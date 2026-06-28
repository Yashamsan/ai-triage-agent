"""ProofLayer API v3 -- FastAPI router for agent registry, decision ingestion,
search, detail, replay, diff, blame, exceptions, trace steps, and governance.

Route order matters: literal paths (/decisions/diff) are registered before
the parametric /decisions/{id} to prevent shadowing.

v3 additions:
  POST /decisions                  -- unified ingest with trace_steps + PII flag
  POST /exceptions                 -- Ghost Knowledge: attach human narrative
  POST /trace-steps                -- Reasoning Memory: append a step post-hoc
  GET  /decisions/{id}/trace       -- full Thought->Action->Observation chain
  GET  /governance                 -- cross-agent PII dashboard
  GET  /overview                   -- expanded with v3 KPI cards
"""

from __future__ import annotations

import io
import json
import os
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from app.prooflayer_graph import (
    add_trace_step,
    get_audit_report,
    get_trace,
    query_cross_agent,
    record_decision,
    record_exception,
)
from app.prooflayer_replay import blame_decision, diff_decisions, replay_decision

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost/triage_agent",
)

router = APIRouter(prefix="/api/v1", tags=["ProofLayer"])


def _get_conn():
    return psycopg2.connect(DATABASE_URL, connect_timeout=3)


# ============================================================================
# Request / Response models
# ============================================================================


class AgentRegistration(BaseModel):
    agent_name: str
    agent_version: str = "1.0"
    model_id: str = ""
    description: str = ""
    agent_group: str | None = None
    data_classification: str = "internal"
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentOut(BaseModel):
    agent_id: str
    agent_name: str
    agent_version: str
    model_id: str
    agent_group: str | None = None
    data_classification: str | None = None
    registered_at: str
    last_seen: str


class TraceStepIn(BaseModel):
    node_type: str
    thought: str | None = None
    action: str | None = None
    observation: str | None = None
    confidence: float | None = None
    latency_ms: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    decision_value: str
    confidence: float
    input_query: str = ""
    agent_name: str = ""
    session_id: str = ""
    model_id: str = ""
    reasoning_summary: str = ""
    human_override: bool = False
    policy_ids: list[str] = Field(default_factory=list)
    contains_pii: bool = False
    agent_group: str | None = None
    trace_steps: list[TraceStepIn] = Field(default_factory=list)
    model_version: str | None = None
    active_policies: list[str] | None = None
    risk_scores: dict[str, float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestOut(BaseModel):
    decision_id: str
    snapshot_id: str
    policy_edges: int
    trace_steps: int


class DecisionSummary(BaseModel):
    decision_id: str
    agent_name: str | None
    decision_value: str
    confidence: float
    human_override: bool
    model_id: str
    session_id: str
    contains_pii: bool
    agent_group: str | None
    timestamp: str


class ExceptionRequest(BaseModel):
    decision_id: str
    human_narrative: str
    approver: str | None = None
    approval_channel: str | None = None
    policy_violated: str | None = None
    justification: str | None = None
    severity: str = "low"


class ExceptionOut(BaseModel):
    exception_id: int
    exc_node_id: str
    created_at: str


class TraceStepRequest(BaseModel):
    decision_id: str
    node_type: str
    thought: str | None = None
    action: str | None = None
    observation: str | None = None
    confidence: float | None = None
    latency_ms: float | None = None
    step_order: int | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


# ============================================================================
# Agent Registry
# ============================================================================


@router.post("/agents", response_model=AgentOut, status_code=201)
def register_agent(req: AgentRegistration):
    """Register an AI agent. Returns existing record if name+version already exists."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        INSERT INTO pl_agents
            (agent_name, agent_version, model_id, description, metadata,
             agent_group, data_classification)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (agent_name, agent_version) DO UPDATE
            SET model_id            = EXCLUDED.model_id,
                agent_group         = EXCLUDED.agent_group,
                data_classification = EXCLUDED.data_classification,
                last_seen           = NOW()
        RETURNING agent_id, agent_name, agent_version, model_id,
                  agent_group, data_classification, registered_at, last_seen
        """,
        (
            req.agent_name, req.agent_version, req.model_id,
            req.description, json.dumps(req.metadata),
            req.agent_group, req.data_classification,
        ),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return AgentOut(
        agent_id=str(row["agent_id"]),
        agent_name=row["agent_name"],
        agent_version=row["agent_version"],
        model_id=row["model_id"],
        agent_group=row.get("agent_group"),
        data_classification=row.get("data_classification"),
        registered_at=row["registered_at"].isoformat(),
        last_seen=row["last_seen"].isoformat(),
    )


@router.get("/agents", response_model=list[AgentOut])
def list_agents():
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT agent_id, agent_name, agent_version, model_id,
               agent_group, data_classification, registered_at, last_seen
        FROM pl_agents ORDER BY last_seen DESC LIMIT 100
        """
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        AgentOut(
            agent_id=str(r["agent_id"]),
            agent_name=r["agent_name"],
            agent_version=r["agent_version"],
            model_id=r["model_id"],
            agent_group=r.get("agent_group"),
            data_classification=r.get("data_classification"),
            registered_at=r["registered_at"].isoformat(),
            last_seen=r["last_seen"].isoformat(),
        )
        for r in rows
    ]


# NOTE: /agents/active must be declared BEFORE /agents/{id}
@router.get("/agents/active")
def list_active_agents():
    """Distinct agent names that have at least one recorded Decision node."""
    conn = _get_conn()
    cur = conn.cursor()
    cur.execute(
        """
        SELECT DISTINCT agent_name
        FROM pl_nodes
        WHERE node_type = 'Decision' AND agent_name IS NOT NULL
        ORDER BY agent_name
        """
    )
    names = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return names


# ============================================================================
# Decision Ingestion
# ============================================================================


@router.post("/decisions", response_model=IngestOut, status_code=201)
def ingest(req: IngestRequest):
    """Unified decision ingestion -- v3: trace steps, PII flag, agent group."""
    try:
        result = record_decision(
            decision_value=req.decision_value,
            confidence=req.confidence,
            human_override=req.human_override,
            reasoning_summary=req.reasoning_summary,
            session_id=req.session_id,
            input_query=req.input_query,
            policy_ids=req.policy_ids,
            model_id=req.model_id or None,
            agent_name=req.agent_name or None,
            contains_pii=req.contains_pii,
            agent_group=req.agent_group,
            trace_steps=[s.model_dump() for s in req.trace_steps],
            model_version=req.model_version,
            active_policies=req.active_policies,
            risk_scores=req.risk_scores,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestOut(
        decision_id=result["decision_id"],
        snapshot_id=result["snapshot_id"],
        policy_edges=result["policy_edges"],
        trace_steps=result["trace_steps"],
    )


# Legacy alias kept for backward compatibility
@router.post("/ingest", response_model=IngestOut, status_code=201)
def ingest_legacy(req: IngestRequest):
    """Alias for POST /decisions -- maintained for backward compatibility."""
    return ingest(req)


# ============================================================================
# v3: Exceptions  (Ghost Knowledge)
# ============================================================================


@router.post("/exceptions", response_model=ExceptionOut, status_code=201)
def create_exception(req: ExceptionRequest):
    """Attach a human exception narrative (ghost knowledge) to a decision."""
    try:
        result = record_exception(
            decision_node_id=req.decision_id,
            human_narrative=req.human_narrative,
            approver=req.approver,
            approval_channel=req.approval_channel,
            policy_violated=req.policy_violated,
            justification=req.justification,
            severity=req.severity,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ExceptionOut(**result)


@router.get("/exceptions")
def list_exceptions(
    since_days: int = Query(30, ge=1, le=365),
    severity: str | None = Query(None),
    limit: int = Query(50, ge=1, le=500),
):
    """List all recorded exceptions with optional severity filter."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    clauses = ["created_at >= NOW() - INTERVAL '1 day' * %(since_days)s"]
    params: dict[str, Any] = {"since_days": since_days, "limit": limit}
    if severity:
        clauses.append("severity = %(severity)s")
        params["severity"] = severity

    where = " AND ".join(clauses)
    cur.execute(
        f"""
        SELECT e.exception_id, e.decision_node_id, e.human_narrative,
               e.approver, e.approval_channel, e.policy_violated,
               e.justification, e.severity, e.created_at,
               n.agent_name,
               n.properties->>'decision_value' AS decision_value
        FROM pl_exceptions e
        JOIN pl_nodes n ON n.node_id = e.decision_node_id
        WHERE {where}
        ORDER BY e.created_at DESC
        LIMIT %(limit)s
        """,
        params,
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "exception_id": r["exception_id"],
            "decision_id": str(r["decision_node_id"]),
            "agent_name": r["agent_name"],
            "decision_value": r["decision_value"],
            "human_narrative": r["human_narrative"],
            "approver": r["approver"],
            "approval_channel": r["approval_channel"],
            "policy_violated": r["policy_violated"],
            "justification": r["justification"],
            "severity": r["severity"],
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


# ============================================================================
# v3: Trace Steps  (Reasoning Memory)
# ============================================================================


@router.post("/trace-steps", status_code=201)
def append_trace_step(req: TraceStepRequest):
    """Append a Thought->Action->Observation step to an existing decision."""
    try:
        result = add_trace_step(
            decision_node_id=req.decision_id,
            node_type=req.node_type,
            thought=req.thought,
            action=req.action,
            observation=req.observation,
            confidence=req.confidence,
            latency_ms=req.latency_ms,
            step_order=req.step_order,
            extra=req.extra,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result


# ============================================================================
# v3: Governance  (Cross-Agent PII Dashboard)
# ============================================================================


@router.get("/governance")
def governance(
    contains_pii: bool | None = Query(None),
    agent_group: str | None = Query(None),
    agent_name: str | None = Query(None),
    since_days: int = Query(7, ge=1, le=365),
    limit: int = Query(200, ge=1, le=1000),
):
    """Cross-agent governance query.

    The demo-closing endpoint: shows every decision across all agents that
    touched PII in the requested window, grouped by agent.
    """
    try:
        return query_cross_agent(
            contains_pii=contains_pii,
            agent_group=agent_group,
            agent_name=agent_name,
            since_days=since_days,
            limit=limit,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ============================================================================
# Search
# ============================================================================


@router.get("/search", response_model=list[DecisionSummary])
@router.get("/decisions", response_model=list[DecisionSummary])
def search_decisions(
    q: str | None = Query(None),
    agent_name: str | None = Query(None),
    decision_value: str | None = Query(None),
    contains_pii: bool | None = Query(None),
    since_days: int = Query(30, ge=1, le=365),
    limit: int = Query(50, ge=1, le=500),
):
    """Search decisions. v3 adds contains_pii filter."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    clauses = [
        "n.node_type = 'Decision'",
        "n.created_at >= NOW() - INTERVAL '1 day' * %(since_days)s",
    ]
    params: dict[str, Any] = {"since_days": since_days, "limit": limit}

    if q:
        clauses.append("n.search_vector @@ plainto_tsquery('english', %(q)s)")
        params["q"] = q
    if agent_name:
        clauses.append("n.agent_name ILIKE %(agent_name)s")
        params["agent_name"] = f"%{agent_name}%"
    if decision_value:
        clauses.append("n.properties->>'decision_value' ILIKE %(decision_value)s")
        params["decision_value"] = f"%{decision_value}%"
    if contains_pii is not None:
        clauses.append("n.contains_pii = %(contains_pii)s")
        params["contains_pii"] = contains_pii

    where = " AND ".join(clauses)
    cur.execute(
        f"""
        SELECT node_id, properties, created_at, agent_name,
               contains_pii, agent_group
        FROM pl_nodes n
        WHERE {where}
        ORDER BY n.created_at DESC
        LIMIT %(limit)s
        """,
        params,
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    summaries = []
    for row in rows:
        p = row["properties"] if isinstance(row["properties"], dict) else json.loads(row["properties"])
        summaries.append(
            DecisionSummary(
                decision_id=str(row["node_id"]),
                agent_name=row["agent_name"],
                decision_value=p.get("decision_value", ""),
                confidence=float(p.get("confidence_score", 0.0)),
                human_override=bool(p.get("human_override", False)),
                model_id=p.get("model_id", ""),
                session_id=p.get("session_id", ""),
                contains_pii=bool(row.get("contains_pii", False)),
                agent_group=row.get("agent_group"),
                timestamp=row["created_at"].isoformat(),
            )
        )
    return summaries


# ============================================================================
# Detail + Timeline
# ============================================================================

# NOTE: literal paths must come before /decisions/{decision_id}


@router.get("/decisions/diff")
def diff(
    left: str = Query(..., description="Left decision UUID"),
    right: str = Query(..., description="Right decision UUID"),
):
    """Compare two decisions."""
    result = diff_decisions(left, right)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/decisions/{decision_id}/trace")
def get_decision_trace(decision_id: str):
    """Return the full Thought->Action->Observation reasoning chain for a decision."""
    try:
        steps = get_trace(decision_id)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"decision_id": decision_id, "steps": steps, "total_steps": len(steps)}


@router.get("/decisions/{decision_id}/replay")
def replay(decision_id: str):
    """Full forensic reconstruction of a decision."""
    result = replay_decision(decision_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
    return result


@router.get("/decisions/{decision_id}/blame")
def blame(decision_id: str):
    """Trace decision outcome to root cause."""
    result = blame_decision(decision_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@router.get("/decisions/{decision_id}")
def get_decision(decision_id: str):
    """Full decision detail including context snapshot, applied policies, and timeline."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    cur.execute(
        """
        SELECT node_id, properties, created_at, agent_name,
               contains_pii, agent_group
        FROM pl_nodes
        WHERE node_id = %s AND node_type = 'Decision'
        """,
        (decision_id,),
    )
    row = cur.fetchone()
    if not row:
        cur.close()
        conn.close()
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")

    p = row["properties"] if isinstance(row["properties"], dict) else json.loads(row["properties"])

    cur.execute(
        """
        SELECT e.edge_type, e.properties AS edge_props, e.valid_from AS edge_at,
               n.node_type, n.properties AS node_props
        FROM pl_edges e
        JOIN pl_nodes n ON n.node_id = e.to_node_id
        WHERE e.from_node_id = %s
        ORDER BY e.valid_from
        """,
        (decision_id,),
    )
    edge_rows = cur.fetchall()

    # v3: exception count
    cur.execute(
        "SELECT COUNT(*) AS cnt FROM pl_exceptions WHERE decision_node_id = %s",
        (decision_id,),
    )
    exc_count = cur.fetchone()["cnt"]

    # v3: trace step count
    cur.execute(
        "SELECT COUNT(*) AS cnt FROM pl_trace_steps WHERE decision_node_id = %s",
        (decision_id,),
    )
    step_count = cur.fetchone()["cnt"]

    cur.close()
    conn.close()

    timeline = []
    for er in edge_rows:
        np_ = er["node_props"] if isinstance(er["node_props"], dict) else json.loads(er["node_props"])
        ep = er["edge_props"] if isinstance(er["edge_props"], dict) else json.loads(er["edge_props"] or "{}")
        timeline.append({
            "edge_type": er["edge_type"],
            "connected_node_type": er["node_type"],
            "edge_properties": ep,
            "node_properties": np_,
            "at": er["edge_at"].isoformat() if er["edge_at"] else None,
        })

    return {
        "decision_id": decision_id,
        "agent_name": row["agent_name"],
        "agent_group": row.get("agent_group"),
        "contains_pii": bool(row.get("contains_pii", False)),
        "timestamp": row["created_at"].isoformat(),
        "properties": p,
        "timeline": timeline,
        "exception_count": int(exc_count),
        "trace_step_count": int(step_count),
    }


# ============================================================================
# ISO 42001 Compliance
# ============================================================================


@router.get("/compliance/iso-42001")
def api_iso_42001_report(
    agent_name: str | None = Query(None),
) -> dict[str, Any]:
    """Full ISO 42001:2023 gap analysis -- 65+ requirements with live evidence."""
    from app.compliance import generate_iso_42001_report
    try:
        return generate_iso_42001_report(agent_name=agent_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/compliance/summary")
def api_compliance_summary(
    agent_name: str | None = Query(None),
) -> dict[str, Any]:
    """Quick ISO 42001 compliance scorecard."""
    from app.compliance import generate_compliance_summary
    try:
        return generate_compliance_summary(agent_name=agent_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/compliance/remediation")
def api_compliance_remediation(
    agent_name: str | None = Query(None),
) -> dict[str, Any]:
    """Prioritised remediation checklist."""
    from app.compliance import generate_remediation_checklist
    try:
        return generate_remediation_checklist(agent_name=agent_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/compliance/report.pdf")
def api_compliance_pdf(
    agent_name: str | None = Query(None),
):
    """Generate and download ISO 42001:2023 compliance report as PDF."""
    from app.compliance_pdf import generate_compliance_pdf
    try:
        pdf_bytes = generate_compliance_pdf(agent_name=agent_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    filename = f"iso42001_compliance{'_' + agent_name if agent_name else ''}.pdf"
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ============================================================================
# Overview  (v3: 4 new KPI cards)
# ============================================================================


@router.get("/overview")
def overview(since_days: int = Query(30, ge=1, le=365)):
    """Dashboard overview -- v3 adds exception_count, trace_step_count,
    pii_decision_count, agent_group_count.
    """
    try:
        report = get_audit_report(since_days=since_days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {
        "since_days": since_days,
        # Core KPIs (v2)
        "total_decisions": int(report.get("total_decisions") or 0),
        "human_overrides": int(report.get("human_overrides") or 0),
        "escalations": int(report.get("escalations") or 0),
        "avg_confidence": round(float(report.get("avg_confidence") or 0), 3),
        # v3 KPIs
        "exception_count": int(report.get("exception_count") or 0),
        "trace_step_count": int(report.get("trace_step_count") or 0),
        "pii_decision_count": int(report.get("pii_decisions") or 0),
        "agent_group_count": int(report.get("agent_group_count") or 0),
    }


# ============================================================================
# Audit Report
# ============================================================================


@router.get("/audit/report")
def audit_report(since_days: int = Query(30, ge=1, le=365)):
    """Aggregate statistics across all decisions in the window."""
    try:
        return get_audit_report(since_days=since_days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
