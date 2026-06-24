"""ProofLayer API — FastAPI router for agent registration, decision ingestion,
search, detail, replay, diff, and blame.

Route order matters: literal paths (/decisions/diff, /decisions/compare) are
registered before the parametric /decisions/{id} to prevent shadowing.
"""

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

from app.prooflayer_graph import record_decision
from app.prooflayer_replay import blame_decision, diff_decisions, replay_decision

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost/triage_agent",
)

router = APIRouter(prefix="/api/v1", tags=["ProofLayer"])


def _get_conn():
    return psycopg2.connect(DATABASE_URL, connect_timeout=3)


# ── Request / Response models ─────────────────────────────────────────────────


class AgentRegistration(BaseModel):
    agent_name: str
    agent_version: str = "1.0"
    model_id: str = ""
    description: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AgentOut(BaseModel):
    agent_id: str
    agent_name: str
    agent_version: str
    model_id: str
    registered_at: str
    last_seen: str


class IngestRequest(BaseModel):
    decision_value: str
    confidence: float
    input_query: str
    agent_name: str = ""
    session_id: str = ""
    model_id: str = ""
    reasoning_summary: str = ""
    human_override: bool = False
    policy_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestOut(BaseModel):
    decision_id: str
    snapshot_id: str
    policy_edges: int


class DecisionSummary(BaseModel):
    decision_id: str
    agent_name: str | None
    decision_value: str
    confidence: float
    human_override: bool
    model_id: str
    session_id: str
    timestamp: str


# ── Agent Registry ────────────────────────────────────────────────────────────


@router.post("/agents", response_model=AgentOut, status_code=201)
def register_agent(req: AgentRegistration):
    """Register an AI agent. Returns existing record if name+version already exists."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        INSERT INTO pl_agents (agent_name, agent_version, model_id, description, metadata)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (agent_name, agent_version) DO UPDATE
            SET model_id   = EXCLUDED.model_id,
                last_seen  = NOW()
        RETURNING agent_id, agent_name, agent_version, model_id, registered_at, last_seen
        """,
        (req.agent_name, req.agent_version, req.model_id, req.description, json.dumps(req.metadata)),
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
        registered_at=row["registered_at"].isoformat(),
        last_seen=row["last_seen"].isoformat(),
    )


@router.get("/agents", response_model=list[AgentOut])
def list_agents():
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        "SELECT agent_id, agent_name, agent_version, model_id, registered_at, last_seen "
        "FROM pl_agents ORDER BY last_seen DESC LIMIT 100"
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
            registered_at=r["registered_at"].isoformat(),
            last_seen=r["last_seen"].isoformat(),
        )
        for r in rows
    ]


# NOTE: /agents/active must be declared before any future /agents/{id} route.
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


# ── Ingestion ─────────────────────────────────────────────────────────────────


@router.post("/ingest", response_model=IngestOut, status_code=201)
def ingest(req: IngestRequest):
    """Unified decision ingestion endpoint.

    Wraps record_decision() and optionally tags the Decision node with the
    calling agent's name. Pass model_id to override the graph-level default.
    """
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
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return IngestOut(
        decision_id=result["decision_id"],
        snapshot_id=result["snapshot_id"],
        policy_edges=result["policy_edges"],
    )


# ── Search ────────────────────────────────────────────────────────────────────


@router.get("/decisions", response_model=list[DecisionSummary])
def search_decisions(
    q: str | None = Query(None, description="Full-text search across all decision fields"),
    agent_name: str | None = Query(None),
    decision_value: str | None = Query(None),
    since_days: int = Query(30, ge=1, le=365),
    limit: int = Query(50, ge=1, le=500),
):
    """Search decisions with optional full-text query, agent, and outcome filters."""
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

    where = " AND ".join(clauses)
    cur.execute(
        f"""
        SELECT node_id, properties, created_at, agent_name
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
                timestamp=row["created_at"].isoformat(),
            )
        )
    return summaries


# ── Detail + Timeline ─────────────────────────────────────────────────────────

# NOTE: /decisions/diff and /decisions/compare must be declared BEFORE
# /decisions/{decision_id} so FastAPI matches them as literals, not UUID params.


@router.get("/decisions/diff")
def diff(
    left: str = Query(..., description="Left decision UUID"),
    right: str = Query(..., description="Right decision UUID"),
):
    """Compare two decisions — changed model, policies, context, and output."""
    result = diff_decisions(left, right)
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
        SELECT node_id, properties, created_at, agent_name
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

    # Edges + connected nodes (timeline)
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
        "timestamp": row["created_at"].isoformat(),
        "properties": p,
        "timeline": timeline,
    }


@router.get("/decisions/{decision_id}/replay")
def replay(decision_id: str):
    """Full forensic reconstruction of a decision. Equivalent to git show."""
    result = replay_decision(decision_id)
    if not result:
        raise HTTPException(status_code=404, detail=f"Decision {decision_id} not found")
    return result


@router.get("/decisions/{decision_id}/blame")
def blame(decision_id: str):
    """Trace decision outcome to root cause. Equivalent to git blame."""
    result = blame_decision(decision_id)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ── ISO 42001 Compliance ──────────────────────────────────────────────────────


@router.get("/compliance/iso-42001")
def api_iso_42001_report(
    agent_name: str | None = Query(None, description="Filter by agent name"),
) -> dict[str, Any]:
    """Full ISO 42001:2023 gap analysis — 65 requirements with live evidence."""
    from app.compliance import generate_iso_42001_report
    try:
        return generate_iso_42001_report(agent_name=agent_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/compliance/summary")
def api_compliance_summary(
    agent_name: str | None = Query(None, description="Filter by agent name"),
) -> dict[str, Any]:
    """Quick ISO 42001 compliance scorecard for the admin dashboard header."""
    from app.compliance import generate_compliance_summary
    try:
        return generate_compliance_summary(agent_name=agent_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/compliance/remediation")
def api_compliance_remediation(
    agent_name: str | None = Query(None, description="Filter by agent name"),
) -> dict[str, Any]:
    """Prioritised remediation checklist: quick wins, documentation gaps, and compliant items."""
    from app.compliance import generate_remediation_checklist
    try:
        return generate_remediation_checklist(agent_name=agent_name)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.get("/compliance/report.pdf")
def api_compliance_pdf(
    agent_name: str | None = Query(None, description="Filter by agent name"),
):
    """Generate and download an ISO 42001:2023 compliance report as a PDF."""
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


# ── Audit Report ──────────────────────────────────────────────────────────────


@router.get("/audit/report")
def audit_report(since_days: int = Query(30, ge=1, le=365)):
    """Aggregate statistics across all decisions in the window."""
    from app.prooflayer_graph import get_audit_report

    try:
        return get_audit_report(since_days=since_days)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
