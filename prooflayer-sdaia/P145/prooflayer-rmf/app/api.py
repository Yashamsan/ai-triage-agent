"""FastAPI router for the SDAIA-P145 RMF module.

All routes are namespaced under /api/v1/rmf so they cannot collide with the
existing Responsible AI Policy module (/api/v1/... from prooflayer-sdaia).
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from . import sdaia_rmf

router = APIRouter(prefix="/api/v1/rmf", tags=["rmf"])


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


@router.post("/context")
def post_context(body: ContextIn):
    return sdaia_rmf.register_context(
        agent_id=body.agent_id,
        description=body.description,
        data_io=body.data_io,
        automation_level=body.automation_level,
        human_role=body.human_role,
        lifecycle_stage=body.lifecycle_stage,
        change_plan=body.change_plan,
    )


@router.get("/context/{agent_id}")
def get_context(agent_id: str):
    ctx = sdaia_rmf.st.get_context(agent_id)
    if ctx is None:
        raise HTTPException(404, "no context registered for this agent")
    return dict(ctx)


@router.post("/risks")
def post_risk(body: RiskIn):
    try:
        return sdaia_rmf.register_risk(
            agent_id=body.agent_id,
            category=body.category,
            description=body.description,
            source=body.source,
            intent=body.intent,
            timing=body.timing,
            identified_by=body.identified_by,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.get("/risks")
def get_risks(agent_id: str | None = None, category: str | None = None):
    return sdaia_rmf.risk_register(agent_id, category)


@router.post("/risks/{risk_id}/assess")
def post_assess(risk_id: str, body: AssessIn):
    try:
        return sdaia_rmf.assess_risk(
            risk_id=risk_id,
            likelihood=body.likelihood,
            impact=body.impact,
            evidence=body.evidence,
            assessed_by=body.assessed_by,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/risks/{risk_id}/treat")
def post_treat(risk_id: str, body: TreatIn):
    try:
        return sdaia_rmf.treat_risk(
            risk_id=risk_id,
            strategy=body.strategy,
            rationale=body.rationale,
            controls=body.controls,
            residual_likelihood=body.residual_likelihood,
            residual_impact=body.residual_impact,
            approver=body.approver,
            approval_mechanism=body.approval_mechanism,
        )
    except ValueError as e:
        raise HTTPException(422, str(e))


@router.post("/reviews")
def post_review(body: ReviewIn):
    return sdaia_rmf.add_review(
        agent_id=body.agent_id,
        review_type=body.review_type,
        findings=body.findings,
        incident_rca=body.incident_rca,
        reviewed_by=body.reviewed_by,
    )


@router.get("/reviews")
def get_reviews(agent_id: str | None = None):
    return [dict(r) for r in sdaia_rmf.st.list_reviews(agent_id)]


@router.get("/matrix")
def get_matrix():
    return sdaia_rmf.fleet_matrix()


@router.get("/report/{agent_id}")
def get_report(agent_id: str):
    return sdaia_rmf.five_stage_report(agent_id)
