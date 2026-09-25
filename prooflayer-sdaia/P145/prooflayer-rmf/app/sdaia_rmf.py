"""SDAIA-P145 five-stage risk cycle engine.

Implements the National AI Risk Management Framework methodology:
  Stage 1: Context & scope      -> register_context()
  Stage 2: Risk identification  -> register_risk()
  Stage 3: Risk assessment      -> assess_risk()   (likelihood x impact, 4x4)
  Stage 4: Risk treatment       -> treat_risk()    (avoid/mitigate/transfer/accept)
  Stage 5: Monitoring & review  -> add_review()

Risk level = likelihood * impact on a 4x4 matrix.
Bands: 1-2 low, 3-6 medium, 8-12 high, 16 catastrophic.
Verified against SDAIA-P145 Arabic text (Figure 5): levels 7, 13, 14, 15
are impossible products of 1-4 x 1-4 and are not valid risk levels.
"""
from dataclasses import dataclass
from uuid import uuid4

from . import storage as st

# Framework's seven-category taxonomy (SDAIA-P145, risk identification stage)
TAXONOMY = [
    "bias_discrimination_abuse",
    "privacy_security",
    "misinformation",
    "malicious_use",
    "human_machine_interaction",
    "social_economic_environmental",
    "safety_limitations",
]

TREATMENT_STRATEGIES = ["avoid", "mitigate", "transfer", "accept"]

BANDS = [
    ("low", 1, 2),
    ("medium", 3, 6),
    ("high", 8, 12),
    ("catastrophic", 16, 16),
]


def band_for(level: int) -> str:
    for name, lo, hi in BANDS:
        if lo <= level <= hi:
            return name
    raise ValueError(f"risk level {level} is not a valid product of 1-4 x 1-4")


@dataclass
class AssessmentResult:
    likelihood: int
    impact: int
    risk_level: int
    band: str


def assess(likelihood: int, impact: int) -> AssessmentResult:
    """Compute the 4x4 matrix result. Both axes are 1-4."""
    if likelihood not in (1, 2, 3, 4):
        raise ValueError("likelihood must be 1-4 (rare, unlikely, likely, almost certain)")
    if impact not in (1, 2, 3, 4):
        raise ValueError("impact must be 1-4 (low, medium, high, catastrophic)")
    level = likelihood * impact
    return AssessmentResult(
        likelihood=likelihood,
        impact=impact,
        risk_level=level,
        band=band_for(level),
    )


# --- Stage 1: Context & scope ------------------------------------------------

def register_context(agent_id: str, description: str = "", data_io: str = "",
                     automation_level: str = "", human_role: str = "",
                     lifecycle_stage: str = "", change_plan: str = "") -> dict:
    data = {
        "description": description,
        "data_io": data_io,
        "automation_level": automation_level,
        "human_role": human_role,
        "lifecycle_stage": lifecycle_stage,
        "change_plan": change_plan,
    }
    st.upsert_context(agent_id, data)
    return {"agent_id": agent_id, "context": data, "stage": 1}


# --- Stage 2: Risk identification --------------------------------------------

def register_risk(agent_id: str, category: str, description: str,
                  source: str = "", intent: str = "", timing: str = "",
                  identified_by: str = "") -> dict:
    if category not in TAXONOMY:
        raise ValueError(f"category must be one of {TAXONOMY}")
    risk_id = str(uuid4())
    st.insert_risk(risk_id, agent_id, category, description, source, intent,
                   timing, identified_by)
    return {"risk_id": risk_id, "agent_id": agent_id, "category": category,
            "status": "open", "stage": 2}


# --- Stage 3: Risk assessment ------------------------------------------------

def assess_risk(risk_id: str, likelihood: int, impact: int,
                evidence: str = "", assessed_by: str = "") -> dict:
    risk = st.get_risk(risk_id)
    if risk is None:
        raise ValueError(f"risk {risk_id} not found")
    result = assess(likelihood, impact)
    st.insert_assessment(str(uuid4()), risk_id, likelihood, impact,
                         result.risk_level, result.band, evidence, assessed_by)
    st.update_risk_status(risk_id, "assessed")
    return {"risk_id": risk_id, "likelihood": likelihood, "impact": impact,
            "risk_level": result.risk_level, "band": result.band, "stage": 3}


# --- Stage 4: Risk treatment -------------------------------------------------

def treat_risk(risk_id: str, strategy: str, rationale: str, controls: str = "",
               residual_likelihood: int | None = None,
               residual_impact: int | None = None,
               approver: str = "", approval_mechanism: str = "") -> dict:
    if strategy not in TREATMENT_STRATEGIES:
        raise ValueError(f"strategy must be one of {TREATMENT_STRATEGIES}")
    if not rationale.strip():
        raise ValueError("rationale is required: the framework demands documented decisions")
    if not approver.strip():
        raise ValueError(
            "approver is required: SDAIA-P145 requires identified responsibilities "
            "and approval mechanisms for acceptance decisions"
        )

    residual_level = None
    if residual_likelihood is not None and residual_impact is not None:
        residual_level = assess(residual_likelihood, residual_impact).risk_level

    st.insert_treatment(str(uuid4()), risk_id, strategy, rationale, controls,
                        residual_likelihood, residual_impact, residual_level,
                        approver, approval_mechanism)
    st.update_risk_status(risk_id, "accepted" if strategy == "accept" else "treated")
    return {"risk_id": risk_id, "strategy": strategy,
            "approver": approver, "residual_level": residual_level, "stage": 4}


# --- Stage 5: Monitoring & review --------------------------------------------

def add_review(agent_id: str, review_type: str = "periodic",
               findings: str = "", incident_rca: str = "",
               reviewed_by: str = "") -> dict:
    st.insert_review(str(uuid4()), agent_id, review_type, findings,
                     incident_rca, reviewed_by)
    return {"agent_id": agent_id, "review_type": review_type, "stage": 5}


# --- Read side ---------------------------------------------------------------

def risk_register(agent_id: str | None = None, category: str | None = None) -> list[dict]:
    """Full register: each risk with its latest assessment + treatment."""
    out = []
    for r in st.list_risks(agent_id, category):
        assessment = st.latest_assessment(r["risk_id"])
        treatment = st.latest_treatment(r["risk_id"])
        out.append({
            "risk_id": r["risk_id"],
            "agent_id": r["agent_id"],
            "category": r["category"],
            "description": r["description"],
            "source": r["source"],
            "intent": r["intent"],
            "timing": r["timing"],
            "status": r["status"],
            "identified_at": r["created_at"],
            "assessment": dict(assessment) if assessment else None,
            "treatment": dict(treatment) if treatment else None,
        })
    return out


def fleet_matrix() -> dict:
    dist = st.matrix_distribution()
    return {
        "low": dist.get("low", 0),
        "medium": dist.get("medium", 0),
        "high": dist.get("high", 0),
        "catastrophic": dist.get("catastrophic", 0),
    }


def five_stage_report(agent_id: str) -> dict:
    """The demo artifact: one document proving all five stages for an agent."""
    context = st.get_context(agent_id)
    risks = risk_register(agent_id)
    reviews = st.list_reviews(agent_id)
    return {
        "agent_id": agent_id,
        "stage_1_context_and_scope": dict(context) if context else None,
        "stage_2_risk_identification": [
            {k: r[k] for k in ("risk_id", "category", "description", "source",
                               "intent", "timing", "status")}
            for r in risks
        ],
        "stage_3_risk_assessment": [
            r["assessment"] for r in risks if r["assessment"]
        ],
        "stage_4_risk_treatment": [
            r["treatment"] for r in risks if r["treatment"]
        ],
        "stage_5_monitoring_and_review": [dict(x) for x in reviews],
        "stages_evidenced": sum([
            bool(context),
            bool(risks),
            any(r["assessment"] for r in risks),
            any(r["treatment"] for r in risks),
            bool(reviews),
        ]),
    }
