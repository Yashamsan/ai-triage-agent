"""ProofLayer SDAIA-P145 National AI Risk Management Framework -- FastAPI
router (/api/v1/rmf).

Ported from the standalone prototype at
prooflayer-sdaia/P145/prooflayer-rmf (own port 8090, own SQLite DB, own
dashboard) so it can run as a tab in the same admin UI as the rest of
ProofLayer. Business logic (the 4x4 matrix, taxonomy, validation rules) is
unchanged -- see app/rmf_core.py. Storage moves from SQLite to this app's
own Postgres database, in dedicated rmf_* tables with no foreign key into
pl_agents or the SDAIA Responsible AI Policy `agents` table: standalone
means data isolation, not a separate server, once it's a tab in the same
dashboard.
"""
from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app import rmf_core

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost/triage_agent",
)

router = APIRouter(prefix="/api/v1/rmf", tags=["RMF (P145)"])


def _get_conn():
    return psycopg2.connect(DATABASE_URL, connect_timeout=3)


def _cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


# ============================================================================
# Request models
# ============================================================================


class ContextIn(BaseModel):
    agent_id: str
    description: str = ""
    data_io: str = ""
    automation_level: str = ""
    human_role: str = ""
    lifecycle_stage: str = ""
    change_plan: str = ""


class RiskIn(BaseModel):
    agent_id: str
    category: str
    description: str
    source: str = ""
    intent: str = ""
    timing: str = ""
    identified_by: str = ""


class AssessIn(BaseModel):
    likelihood: int = Field(ge=1, le=4)
    impact: int = Field(ge=1, le=4)
    evidence: str = ""
    assessed_by: str = ""


class TreatIn(BaseModel):
    strategy: str
    rationale: str
    controls: str = ""
    residual_likelihood: int | None = Field(default=None, ge=1, le=4)
    residual_impact: int | None = Field(default=None, ge=1, le=4)
    approver: str
    approval_mechanism: str = ""


class ReviewIn(BaseModel):
    agent_id: str
    review_type: str = "periodic"
    findings: str = ""
    incident_rca: str = ""
    reviewed_by: str = ""


# ============================================================================
# Stage 1: Context & scope
# ============================================================================


@router.post("/context")
def post_context(body: ContextIn) -> dict:
    with _get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                INSERT INTO rmf_contexts
                    (context_id, agent_id, description, data_io, automation_level,
                     human_role, lifecycle_stage, change_plan)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (agent_id) DO UPDATE SET
                    description = EXCLUDED.description,
                    data_io = EXCLUDED.data_io,
                    automation_level = EXCLUDED.automation_level,
                    human_role = EXCLUDED.human_role,
                    lifecycle_stage = EXCLUDED.lifecycle_stage,
                    change_plan = EXCLUDED.change_plan
                """,
                (
                    str(uuid4()), body.agent_id, body.description, body.data_io,
                    body.automation_level, body.human_role, body.lifecycle_stage,
                    body.change_plan,
                ),
            )
        conn.commit()
    return {
        "agent_id": body.agent_id,
        "context": body.model_dump(exclude={"agent_id"}),
        "stage": 1,
    }


@router.get("/context/{agent_id}")
def get_context(agent_id: str) -> dict:
    with _get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute("SELECT * FROM rmf_contexts WHERE agent_id = %s", (agent_id,))
            row = cur.fetchone()
    if row is None:
        raise HTTPException(404, "no context registered for this agent")
    return dict(row)


# ============================================================================
# Stage 2: Risk identification
# ============================================================================


@router.post("/risks")
def post_risk(body: RiskIn) -> dict:
    try:
        rmf_core.validate_category(body.category)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    risk_id = str(uuid4())
    with _get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                INSERT INTO rmf_risks
                    (risk_id, agent_id, category, description, source, intent,
                     timing, identified_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    risk_id, body.agent_id, body.category, body.description,
                    body.source, body.intent, body.timing, body.identified_by,
                ),
            )
        conn.commit()
    return {"risk_id": risk_id, "agent_id": body.agent_id, "category": body.category,
            "status": "open", "stage": 2}


@router.get("/risks")
def get_risks(agent_id: str | None = None, category: str | None = None) -> list[dict]:
    where, params = [], []
    if agent_id:
        where.append("agent_id = %s")
        params.append(agent_id)
    if category:
        where.append("category = %s")
        params.append(category)
    clause = f"WHERE {' AND '.join(where)}" if where else ""

    with _get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                f"SELECT * FROM rmf_risks {clause} ORDER BY created_at DESC",
                params,
            )
            risks = cur.fetchall()

            out = []
            for r in risks:
                cur.execute(
                    "SELECT * FROM rmf_assessments WHERE risk_id = %s "
                    "ORDER BY created_at DESC LIMIT 1",
                    (r["risk_id"],),
                )
                assessment = cur.fetchone()
                cur.execute(
                    "SELECT * FROM rmf_treatments WHERE risk_id = %s "
                    "ORDER BY created_at DESC LIMIT 1",
                    (r["risk_id"],),
                )
                treatment = cur.fetchone()
                out.append({
                    "risk_id": r["risk_id"],
                    "agent_id": r["agent_id"],
                    "category": r["category"],
                    "description": r["description"],
                    "source": r["source"],
                    "intent": r["intent"],
                    "timing": r["timing"],
                    "status": r["status"],
                    "identified_at": r["created_at"].isoformat(),
                    "assessment": _serialize(assessment),
                    "treatment": _serialize(treatment),
                })
    return out


def _serialize(row: Any) -> dict | None:
    if row is None:
        return None
    d = dict(row)
    if d.get("created_at") is not None:
        d["created_at"] = d["created_at"].isoformat()
    return d


# ============================================================================
# Stage 3: Risk assessment
# ============================================================================


@router.post("/risks/{risk_id}/assess")
def post_assess(risk_id: str, body: AssessIn) -> dict:
    try:
        result = rmf_core.assess(body.likelihood, body.impact)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    with _get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute("SELECT risk_id FROM rmf_risks WHERE risk_id = %s", (risk_id,))
            if cur.fetchone() is None:
                raise HTTPException(404, f"risk {risk_id} not found")

            cur.execute(
                """
                INSERT INTO rmf_assessments
                    (assessment_id, risk_id, likelihood, impact, risk_level,
                     band, evidence, assessed_by)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid4()), risk_id, body.likelihood, body.impact,
                    result.risk_level, result.band, body.evidence, body.assessed_by,
                ),
            )
            cur.execute(
                "UPDATE rmf_risks SET status = 'assessed' WHERE risk_id = %s",
                (risk_id,),
            )
        conn.commit()
    return {"risk_id": risk_id, "likelihood": body.likelihood, "impact": body.impact,
            "risk_level": result.risk_level, "band": result.band, "stage": 3}


# ============================================================================
# Stage 4: Risk treatment
# ============================================================================


@router.post("/risks/{risk_id}/treat")
def post_treat(risk_id: str, body: TreatIn) -> dict:
    try:
        rmf_core.validate_treatment(body.strategy, body.rationale, body.approver)
    except ValueError as e:
        raise HTTPException(422, str(e)) from e

    residual_level = None
    if body.residual_likelihood is not None and body.residual_impact is not None:
        residual_level = rmf_core.assess(body.residual_likelihood, body.residual_impact).risk_level

    with _get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute("SELECT risk_id FROM rmf_risks WHERE risk_id = %s", (risk_id,))
            if cur.fetchone() is None:
                raise HTTPException(404, f"risk {risk_id} not found")

            cur.execute(
                """
                INSERT INTO rmf_treatments
                    (treatment_id, risk_id, strategy, rationale, controls,
                     residual_likelihood, residual_impact, residual_level,
                     approver, approval_mechanism)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid4()), risk_id, body.strategy, body.rationale, body.controls,
                    body.residual_likelihood, body.residual_impact, residual_level,
                    body.approver, body.approval_mechanism,
                ),
            )
            new_status = "accepted" if body.strategy == "accept" else "treated"
            cur.execute(
                "UPDATE rmf_risks SET status = %s WHERE risk_id = %s",
                (new_status, risk_id),
            )
        conn.commit()
    return {"risk_id": risk_id, "strategy": body.strategy,
            "approver": body.approver, "residual_level": residual_level, "stage": 4}


# ============================================================================
# Stage 5: Monitoring & review
# ============================================================================


@router.post("/reviews")
def post_review(body: ReviewIn) -> dict:
    with _get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                INSERT INTO rmf_reviews
                    (review_id, agent_id, review_type, findings, incident_rca, reviewed_by)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    str(uuid4()), body.agent_id, body.review_type, body.findings,
                    body.incident_rca, body.reviewed_by,
                ),
            )
        conn.commit()
    return {"agent_id": body.agent_id, "review_type": body.review_type, "stage": 5}


@router.get("/reviews")
def get_reviews(agent_id: str | None = None) -> list[dict]:
    with _get_conn() as conn:
        with _cursor(conn) as cur:
            if agent_id:
                cur.execute(
                    "SELECT * FROM rmf_reviews WHERE agent_id = %s ORDER BY created_at DESC",
                    (agent_id,),
                )
            else:
                cur.execute("SELECT * FROM rmf_reviews ORDER BY created_at DESC")
            rows = cur.fetchall()
    return [_serialize(r) for r in rows]


# ============================================================================
# Read side
# ============================================================================


@router.get("/matrix")
def get_matrix() -> dict:
    """Fleet-wide risk distribution: latest assessment per risk, grouped by band."""
    with _get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT a.band, COUNT(*) AS n
                FROM rmf_assessments a
                WHERE a.assessment_id = (
                    SELECT b.assessment_id FROM rmf_assessments b
                    WHERE b.risk_id = a.risk_id
                    ORDER BY b.created_at DESC LIMIT 1
                )
                GROUP BY a.band
                """
            )
            dist = {r["band"]: r["n"] for r in cur.fetchall()}
    return {
        "low": dist.get("low", 0),
        "medium": dist.get("medium", 0),
        "high": dist.get("high", 0),
        "catastrophic": dist.get("catastrophic", 0),
    }


@router.get("/report/{agent_id}")
def get_report(agent_id: str) -> dict:
    """The demo artifact: one document proving all five stages for an agent."""
    risks = get_risks(agent_id=agent_id)
    with _get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute("SELECT * FROM rmf_contexts WHERE agent_id = %s", (agent_id,))
            context = cur.fetchone()
            cur.execute(
                "SELECT * FROM rmf_reviews WHERE agent_id = %s ORDER BY created_at DESC",
                (agent_id,),
            )
            reviews = cur.fetchall()

    return {
        "agent_id": agent_id,
        "stage_1_context_and_scope": _serialize(context),
        "stage_2_risk_identification": [
            {k: r[k] for k in ("risk_id", "category", "description", "source",
                                "intent", "timing", "status")}
            for r in risks
        ],
        "stage_3_risk_assessment": [r["assessment"] for r in risks if r["assessment"]],
        "stage_4_risk_treatment": [r["treatment"] for r in risks if r["treatment"]],
        "stage_5_monitoring_and_review": [_serialize(r) for r in reviews],
        "stages_evidenced": sum([
            bool(context),
            bool(risks),
            any(r["assessment"] for r in risks),
            any(r["treatment"] for r in risks),
            bool(reviews),
        ]),
    }
