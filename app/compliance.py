"""ISO 42001:2023 AI Management System — Compliance Engine.

Evaluates 65 requirements across Clauses 4-10 and Annex A controls using
live evidence from the ProofLayer context graph (pl_nodes / pl_edges).

All public generators accept an optional `agent_name` parameter so the
compliance tab can scope results to a single deployed agent.

Status tiers:
  MET          — direct evidence found in the graph
  PARTIAL      — partial evidence; gap action required
  NOT_MET      — no evidence; gap identified
  NOT_APPLICABLE — requirement does not apply to this deployment
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost/triage_agent",
)

# ── Status / weight constants ─────────────────────────────────────────────────

MET      = "MET"
PARTIAL  = "PARTIAL"
NOT_MET  = "NOT_MET"
NA       = "NOT_APPLICABLE"

CRITICAL = 3.0
MAJOR    = 2.0
MINOR    = 1.0

# ── Data structures ───────────────────────────────────────────────────────────


@dataclass
class CheckResult:
    status: str
    evidence: str
    recommendation: str
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Requirement:
    req_id: str
    clause: str
    section: str
    title: str
    description: str
    weight: float
    check_fn: Callable[[], CheckResult]


# ── DB helpers ────────────────────────────────────────────────────────────────


def _q1(sql: str, params: tuple = ()) -> dict | None:
    with psycopg2.connect(DATABASE_URL, connect_timeout=3) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return dict(cur.fetchone()) if cur.rowcount != 0 else None


def _qa(sql: str, params: tuple = ()) -> list[dict]:
    with psycopg2.connect(DATABASE_URL, connect_timeout=3) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def _count(sql: str, params: tuple = ()) -> int:
    row = _q1(sql, params)
    if not row:
        return 0
    return int(list(row.values())[0])


def _af(agent_name: str | None) -> tuple[str, tuple]:
    """Returns (WHERE SQL fragment, params tuple) for decision agent filtering."""
    if agent_name:
        return " AND agent_name = %s", (agent_name,)
    return "", ()


# ═════════════════════════════════════════════════════════════════════════════
# Evidence check functions — all accept agent_name: str | None = None
# ═════════════════════════════════════════════════════════════════════════════


# ── Clause 4: Context ────────────────────────────────────────────────────────

def _c4_1(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    agents = _qa(
        f"SELECT DISTINCT agent_name FROM pl_nodes "
        f"WHERE node_type='Decision' AND agent_name IS NOT NULL{af}",
        ap,
    )
    names = [r["agent_name"] for r in agents]
    if names:
        return CheckResult(
            PARTIAL,
            f"AI systems identified in graph: {', '.join(names)}.",
            "Document a formal organizational context analysis covering internal/external factors affecting AI use.",
            {"agents": names},
        )
    return CheckResult(
        NOT_MET, "No AI systems identified in ProofLayer.",
        "Record AI system context: purpose, stakeholders, regulatory environment.",
    )


def _c4_2(agent_name: str | None = None) -> CheckResult:
    policies = _count(
        "SELECT COUNT(*) FROM pl_nodes WHERE node_type='Policy' AND valid_to IS NULL"
    )
    if policies >= 3:
        return CheckResult(
            PARTIAL,
            f"{policies} active policy nodes found covering interested-party requirements.",
            "Maintain a formal stakeholder register mapping requirements to policy nodes.",
            {"active_policies": policies},
        )
    return CheckResult(
        NOT_MET, "Fewer than 3 active policy nodes — insufficient stakeholder coverage.",
        "Define policies for each key interested party (customers, regulators, staff).",
        {"active_policies": policies},
    )


def _c4_3(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    agents = _qa(
        f"SELECT DISTINCT agent_name FROM pl_nodes "
        f"WHERE node_type='Decision' AND agent_name IS NOT NULL{af}",
        ap,
    )
    names = [r["agent_name"] for r in agents]
    if names:
        return CheckResult(
            PARTIAL,
            f"AIMS scope covers: {', '.join(names)}.",
            "Produce a written AIMS scope statement referencing these AI systems and their boundaries.",
            {"in_scope_agents": names},
        )
    return CheckResult(
        NOT_MET, "No AI systems in scope identified.",
        "Define and document the AIMS scope, including which AI systems are included.",
    )


def _c4_4(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    policies = _count("SELECT COUNT(*) FROM pl_nodes WHERE node_type='Policy'")
    has_edges = _count("SELECT COUNT(*) FROM pl_edges WHERE edge_type='APPLIED_POLICY'")
    if total > 0 and policies > 0 and has_edges > 0:
        return CheckResult(
            MET,
            f"AIMS operational: {total} decisions, {policies} policies, {has_edges} policy applications recorded.",
            "Maintain and continually improve the AIMS as the system scales.",
            {"decisions": total, "policies": policies, "policy_applications": has_edges},
        )
    if total > 0:
        return CheckResult(
            PARTIAL,
            f"{total} decisions recorded but policy governance graph is incomplete.",
            "Link decisions to policies via APPLIED_POLICY edges for full AIMS coverage.",
            {"decisions": total},
        )
    return CheckResult(
        NOT_MET, "No evidence of an operational AI Management System.",
        "Establish the AIMS: log decisions, document policies, and link them in ProofLayer.",
    )


# ── Clause 5: Leadership ─────────────────────────────────────────────────────

def _c5_1(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    overrides = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND (properties->>'human_override')::boolean = true{af}",
        ap,
    )
    escalations = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'decision_value'='escalation'{af}",
        ap,
    )
    if overrides > 0 or escalations > 0:
        return CheckResult(
            PARTIAL,
            f"Leadership engagement evidenced by {overrides} human overrides and {escalations} escalations.",
            "Document top management's formal commitment to the AIMS via an AI Policy signed by leadership.",
            {"human_overrides": overrides, "escalations": escalations},
        )
    return CheckResult(
        NOT_MET, "No evidence of leadership engagement (no overrides or escalations).",
        "Top management must demonstrate commitment: sign the AI policy, review metrics, approve escalations.",
    )


def _c5_2(agent_name: str | None = None) -> CheckResult:
    policies = _qa(
        "SELECT properties FROM pl_nodes WHERE node_type='Policy' AND valid_to IS NULL "
        "ORDER BY valid_from DESC LIMIT 5"
    )
    if len(policies) >= 2:
        return CheckResult(
            PARTIAL,
            f"{len(policies)} active AI policy nodes found in ProofLayer.",
            "Formalize policies into a signed, versioned AI Policy document communicated to all staff.",
            {"policy_count": len(policies)},
        )
    return CheckResult(
        NOT_MET, "Fewer than 2 active policy nodes — AI policy is insufficient.",
        "Establish a documented AI Policy covering objectives, responsibilities, and ethical principles.",
    )


def _c5_3(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    agents = _qa(
        f"SELECT DISTINCT agent_name FROM pl_nodes "
        f"WHERE node_type='Decision' AND agent_name IS NOT NULL{af}",
        ap,
    )
    if agents:
        return CheckResult(
            PARTIAL,
            f"Agent responsibilities partially defined: {[r['agent_name'] for r in agents]}.",
            "Assign named owners for each AI system and document their authority in an RACI matrix.",
            {"agents": [r["agent_name"] for r in agents]},
        )
    return CheckResult(
        NOT_MET, "No agent roles or responsibilities defined in ProofLayer.",
        "Define organizational roles: AI Owner, Data Steward, Ethics Reviewer, AIMS Manager.",
    )


# ── Clause 6: Planning ───────────────────────────────────────────────────────

def _c6_1(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    low_conf = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND (properties->>'confidence_score')::float < 0.6{af}",
        ap,
    )
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    if total == 0:
        return CheckResult(NOT_MET, "No decisions to assess.", "Begin logging decisions to enable risk identification.")
    pct = round(low_conf / total * 100, 1) if total else 0
    if pct < 15:
        return CheckResult(
            PARTIAL,
            f"{low_conf}/{total} decisions ({pct}%) below confidence threshold — risks identified.",
            "Formalize a risk register documenting these low-confidence scenarios and mitigation actions.",
            {"low_confidence_decisions": low_conf, "total": total, "pct": pct},
        )
    return CheckResult(
        PARTIAL,
        f"High risk rate: {pct}% of decisions are low-confidence.",
        "Urgent: risk register required. Implement thresholds and automatic escalation for low-confidence.",
        {"low_confidence_decisions": low_conf, "total": total, "pct": pct},
    )


def _c6_2(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    rows = _q1(
        f"SELECT COUNT(*) AS total, AVG((properties->>'confidence_score')::float) AS avg_conf "
        f"FROM pl_nodes WHERE node_type='Decision'{af}",
        ap,
    )
    if not rows or int(rows["total"]) == 0:
        return CheckResult(NOT_MET, "No risk data available.", "Log decisions to enable AI risk assessment.")
    avg = float(rows["avg_conf"] or 0)
    if avg >= 0.75:
        return CheckResult(
            PARTIAL,
            f"Average confidence {avg:.0%} — risk level acceptable. Formal risk assessment process still needed.",
            "Document an AI risk assessment methodology with criteria for acceptable risk levels.",
            {"avg_confidence": avg},
        )
    return CheckResult(
        NOT_MET,
        f"Average confidence {avg:.0%} is below acceptable threshold — formal risk assessment overdue.",
        "Conduct AI risk assessment: identify failure modes, likelihood, impact, and mitigations.",
        {"avg_confidence": avg},
    )


def _c6_3(agent_name: str | None = None) -> CheckResult:
    policies = _count(
        "SELECT COUNT(*) FROM pl_nodes WHERE node_type='Policy' AND valid_to IS NULL"
    )
    applied = _count("SELECT COUNT(*) FROM pl_edges WHERE edge_type='APPLIED_POLICY'")
    if policies > 0 and applied > 0:
        return CheckResult(
            PARTIAL,
            f"{policies} risk treatment policies active; {applied} applications recorded.",
            "Document a formal risk treatment plan linking each identified risk to a policy control.",
            {"policies": policies, "applications": applied},
        )
    return CheckResult(
        NOT_MET, "No risk treatment policies or applications found.",
        "Define risk treatment options (avoid, mitigate, transfer, accept) and link to Policy nodes.",
    )


def _c6_4(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    rows = _q1(
        f"SELECT COUNT(*) AS with_session FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'session_id' != '' "
        f"AND properties->>'session_id' IS NOT NULL{af}",
        ap,
    )
    with_session = int(rows["with_session"]) if rows else 0
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    if total == 0:
        return CheckResult(NOT_MET, "No decisions logged.", "Log decisions to assess impact traceability.")
    pct = round(with_session / total * 100) if total else 0
    if pct >= 80:
        return CheckResult(
            MET,
            f"{pct}% of decisions have session tracking for impact traceability.",
            "Maintain session tracking and link to user impact assessments.",
            {"pct_with_session": pct},
        )
    return CheckResult(
        PARTIAL,
        f"Only {pct}% of decisions have session IDs — impact traceability incomplete.",
        "Ensure all triage calls pass session_id to record_decision().",
        {"pct_with_session": pct},
    )


def _c6_5(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    recent = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND created_at >= NOW() - INTERVAL '30 days'{af}",
        ap,
    )
    if total > 0:
        return CheckResult(
            PARTIAL,
            f"{recent} decisions in last 30 days — AI objectives being tracked.",
            "Formalize AI objectives (accuracy targets, escalation rate caps) with measurable KPIs.",
            {"total": total, "recent_30d": recent},
        )
    return CheckResult(
        NOT_MET, "No decisions to track against AI objectives.",
        "Define SMART AI objectives: target confidence ≥ 85%, escalation rate < 5%.",
    )


# ── Clause 7: Support ────────────────────────────────────────────────────────

def _c7_1(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    agents = _count(
        f"SELECT COUNT(DISTINCT agent_name) FROM pl_nodes "
        f"WHERE node_type='Decision' AND agent_name IS NOT NULL{af}",
        ap,
    )
    if agents >= 2:
        return CheckResult(
            PARTIAL,
            f"{agents} distinct AI agents operational — compute resources allocated.",
            "Document resource allocation: compute, storage, API budgets for each AI system.",
            {"agent_count": agents},
        )
    elif agents == 1:
        return CheckResult(
            PARTIAL, "1 AI agent operational.",
            "Document resource requirements and capacity planning for scaling.",
            {"agent_count": 1},
        )
    return CheckResult(NOT_MET, "No AI agents identified.", "Provision and document compute resources for AI systems.")


def _c7_2(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "No competence records in ProofLayer (competence is documented externally).",
        "Maintain training records for AI system operators, data stewards, and ethics reviewers.",
    )


def _c7_3(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Awareness training records not tracked in ProofLayer.",
        "Conduct and record AI awareness training for all staff interacting with AI outputs.",
    )


def _c7_4(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "No communication plan evidenced in ProofLayer.",
        "Define internal/external communication plan: who communicates what about AI decisions, when, and how.",
    )


def _c7_5(agent_name: str | None = None) -> CheckResult:
    policies = _qa(
        "SELECT properties->>'policy_id' AS pid, properties->>'policy_version' AS ver "
        "FROM pl_nodes WHERE node_type='Policy' ORDER BY valid_from DESC"
    )
    if len(policies) >= 3:
        return CheckResult(
            PARTIAL,
            f"{len(policies)} policy documents versioned in ProofLayer.",
            "Extend documented information to include AIMS scope, risk register, and AI system specifications.",
            {"documented_policies": len(policies)},
        )
    return CheckResult(
        NOT_MET, "Fewer than 3 policy documents — documented information system is insufficient.",
        "Create: AIMS scope document, AI policy, risk register, system descriptions, and operating procedures.",
    )


def _c7_6(agent_name: str | None = None) -> CheckResult:
    versioned = _qa(
        "SELECT properties->>'policy_id' AS pid, properties->>'policy_version' AS ver "
        "FROM pl_nodes WHERE node_type='Policy'"
    )
    with_version = [r for r in versioned if r.get("ver")]
    if len(with_version) >= 2:
        return CheckResult(
            MET,
            f"{len(with_version)} policy nodes have version numbers.",
            "Ensure all documents follow the version control procedure (review, approve, timestamp).",
            {"versioned_policies": len(with_version)},
        )
    return CheckResult(
        NOT_MET, "Policy nodes lack version numbers.",
        "Implement version control: each policy must have policy_version, effective_from, and approver.",
    )


# ── Clause 8: Operation ──────────────────────────────────────────────────────

def _c8_1(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    recent = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND created_at >= NOW() - INTERVAL '7 days'{af}",
        ap,
    )
    if recent > 0:
        return CheckResult(
            MET,
            f"{recent} decisions recorded in the last 7 days — AI systems operationally controlled.",
            "Document operational procedures: deployment gates, monitoring cadence, incident escalation path.",
            {"recent_decisions": recent},
        )
    return CheckResult(
        NOT_MET, "No recent operational decisions found.",
        "Establish operational control: deployment procedures, change management, monitoring processes.",
    )


def _c8_2(agent_name: str | None = None) -> CheckResult:
    snapshots = _count("SELECT COUNT(*) FROM pl_nodes WHERE node_type='ContextSnapshot'")
    linked = _count("SELECT COUNT(*) FROM pl_edges WHERE edge_type='USED_CONTEXT'")
    if snapshots > 0 and linked > 0:
        return CheckResult(
            PARTIAL,
            f"{snapshots} context snapshots captured; {linked} linked to decisions.",
            "Formalize impact assessments for each AI use case, not just individual decisions.",
            {"snapshots": snapshots, "linked": linked},
        )
    return CheckResult(
        NOT_MET, "No context snapshots or impact evidence in graph.",
        "Implement AI impact assessment: capture context, assess risks, document mitigations before deployment.",
    )


def _c8_3(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    agents = _qa(
        f"SELECT DISTINCT agent_name FROM pl_nodes "
        f"WHERE node_type='Decision' AND agent_name IS NOT NULL{af}",
        ap,
    )
    models = _qa(
        f"SELECT DISTINCT properties->>'model_id' AS model FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'model_id' IS NOT NULL{af}",
        ap,
    )
    if agents and models:
        return CheckResult(
            PARTIAL,
            f"AI system design evidenced: {len(agents)} agents using {len(models)} model(s).",
            "Document AI system design specifications: architecture, training data, evaluation criteria, bias testing.",
            {"agents": [r["agent_name"] for r in agents], "models": [r["model"] for r in models]},
        )
    return CheckResult(
        NOT_MET, "AI system design not evidenced in ProofLayer.",
        "Document AI system design: objectives, architecture, data sources, model selection rationale.",
    )


def _c8_4(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Data governance records not tracked in ProofLayer.",
        "Document data sources, quality criteria, preprocessing steps, and retention policies for all AI training data.",
    )


def _c8_5(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    with_reasoning = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'reasoning_summary' != '' "
        f"AND properties->>'reasoning_summary' IS NOT NULL{af}",
        ap,
    )
    if total == 0:
        return CheckResult(NOT_MET, "No decisions logged.", "Log decisions to assess implementation quality.")
    pct = round(with_reasoning / total * 100) if total else 0
    if pct >= 50:
        return CheckResult(
            MET,
            f"{pct}% of decisions include reasoning summaries — implementation traceable.",
            "Maintain reasoning capture across all decision paths including Arabic agent.",
            {"pct_with_reasoning": pct},
        )
    return CheckResult(
        PARTIAL,
        f"Only {pct}% of decisions have reasoning summaries.",
        "Capture reasoning_summary for all decisions to meet implementation traceability requirement.",
        {"pct_with_reasoning": pct},
    )


def _c8_6(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Third-party AI provider agreements not tracked in ProofLayer.",
        "Document contracts with AI providers (DeepSeek, OpenRouter/Qwen) covering data processing, liability, SLAs.",
    )


# ── Clause 9: Performance Evaluation ─────────────────────────────────────────

def _c9_1(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    row = _q1(
        f"SELECT COUNT(*) AS total, AVG((properties->>'confidence_score')::float) AS avg_conf, "
        f"COUNT(*) FILTER(WHERE created_at >= NOW() - INTERVAL '7 days') AS recent "
        f"FROM pl_nodes WHERE node_type='Decision'{af}",
        ap,
    )
    if not row or int(row["total"]) == 0:
        return CheckResult(NOT_MET, "No performance data available.", "Log decisions to enable performance monitoring.")
    avg = float(row["avg_conf"] or 0)
    recent = int(row["recent"])
    if recent > 0 and avg >= 0.70:
        return CheckResult(
            MET,
            f"Performance monitoring active: avg confidence {avg:.0%}, {recent} decisions in last 7 days.",
            "Define formal KPIs with thresholds and automated alerting when metrics degrade.",
            {"avg_confidence": avg, "recent_7d": recent},
        )
    return CheckResult(
        PARTIAL,
        f"Metrics available but avg confidence {avg:.0%} is below target or no recent activity.",
        "Set KPI thresholds (e.g. confidence ≥ 85%) and establish monitoring cadence.",
        {"avg_confidence": avg},
    )


def _c9_2(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Internal audit records not tracked in ProofLayer.",
        "Schedule annual ISO 42001 internal audits; maintain audit programme with scope, frequency, and findings.",
    )


def _c9_3(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Management review records not in ProofLayer.",
        "Conduct management reviews of AIMS at planned intervals; document inputs, outputs, and decisions.",
    )


# ── Clause 10: Improvement ────────────────────────────────────────────────────

def _c10_1(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    overrides = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND (properties->>'human_override')::boolean = true{af}",
        ap,
    )
    if overrides > 0:
        return CheckResult(
            PARTIAL,
            f"{overrides} human override(s) recorded — nonconformities captured.",
            "Implement corrective action process: root cause analysis for each override with documented closure.",
            {"human_overrides": overrides},
        )
    return CheckResult(
        NOT_MET, "No nonconformity records found.",
        "Define nonconformity process: detection, root cause analysis, corrective action, effectiveness review.",
    )


def _c10_2(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    trend = _qa(
        f"SELECT DATE_TRUNC('week', created_at) AS week, "
        f"AVG((properties->>'confidence_score')::float) AS avg_conf "
        f"FROM pl_nodes WHERE node_type='Decision'{af} "
        f"GROUP BY week ORDER BY week DESC LIMIT 4",
        ap,
    )
    if len(trend) >= 2:
        latest = float(trend[0]["avg_conf"] or 0)
        earliest = float(trend[-1]["avg_conf"] or 0)
        improving = latest > earliest
        return CheckResult(
            MET if improving else PARTIAL,
            f"{'Improving' if improving else 'Flat/declining'} trend: {earliest:.0%} → {latest:.0%} avg confidence.",
            "Formalise continual improvement plan with quarterly targets and PDCA cycle reviews.",
            {"trend_weeks": len(trend), "latest_avg_conf": latest},
        )
    return CheckResult(
        NOT_MET, "Insufficient history for improvement trend analysis.",
        "Establish baseline metrics now; review monthly to demonstrate continual improvement.",
    )


def _c10_3(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    escalations = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'decision_value'='escalation'{af}",
        ap,
    )
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    if total == 0:
        return CheckResult(NOT_MET, "No data.", "Log decisions to enable improvement tracking.")
    rate = round(escalations / total * 100, 1) if total else 0
    if rate <= 10:
        return CheckResult(
            MET,
            f"Escalation rate {rate}% — within acceptable range for continual improvement.",
            "Track escalation rate as a leading improvement indicator; investigate spikes.",
            {"escalation_rate_pct": rate},
        )
    return CheckResult(
        PARTIAL,
        f"High escalation rate {rate}% — improvement action needed.",
        "Investigate root causes of escalations; update classifier training or routing rules.",
        {"escalation_rate_pct": rate},
    )


# ── Annex A.2: Policies for AI ───────────────────────────────────────────────

def _a2_1(agent_name: str | None = None) -> CheckResult:
    policies = _qa(
        "SELECT properties FROM pl_nodes WHERE node_type='Policy' AND valid_to IS NULL "
        "ORDER BY valid_from DESC"
    )
    if len(policies) >= 3:
        return CheckResult(
            PARTIAL,
            f"{len(policies)} active AI policies in ProofLayer.",
            "Consolidate into a formal Responsible AI Policy document covering fairness, transparency, accountability.",
            {"active_policies": len(policies)},
        )
    return CheckResult(
        NOT_MET, f"Only {len(policies)} active policies — responsible use policy is insufficient.",
        "Publish a Responsible AI Use Policy covering ethical principles, prohibited uses, and review cadence.",
    )


def _a2_2(agent_name: str | None = None) -> CheckResult:
    applied = _count("SELECT COUNT(*) FROM pl_edges WHERE edge_type='APPLIED_POLICY'")
    if applied >= 5:
        return CheckResult(
            MET,
            f"{applied} policy applications recorded — AI policies are operationally enforced.",
            "Add human-readable policy references to decision reasoning summaries for auditability.",
            {"policy_applications": applied},
        )
    return CheckResult(
        PARTIAL,
        f"Only {applied} policy applications — policies not consistently enforced.",
        "Ensure every decision records which policies were applied via APPLIED_POLICY edges.",
        {"policy_applications": applied},
    )


def _a2_3(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Internal AI use guidelines not tracked in ProofLayer.",
        "Publish internal guidelines for staff using AI outputs: verification requirements, escalation triggers, documentation duties.",
    )


def _a2_4(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "AI ethics review process not evidenced in graph.",
        "Establish an AI Ethics Review Board or process; document review outputs in the AIMS.",
    )


# ── Annex A.3: Internal Organisation ─────────────────────────────────────────

def _a3_1(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    agents = _count(
        f"SELECT COUNT(DISTINCT agent_name) FROM pl_nodes "
        f"WHERE node_type='Decision' AND agent_name IS NOT NULL{af}",
        ap,
    )
    if agents >= 1:
        return CheckResult(
            PARTIAL,
            f"{agents} named AI agent(s) with identifiable ownership in ProofLayer.",
            "Formalise division of responsibilities: name an AI System Owner for each agent.",
            {"named_agents": agents},
        )
    return CheckResult(
        NOT_MET, "No named agent responsibilities defined.",
        "Assign AI System Owners; document in org chart and AIMS scope statement.",
    )


def _a3_2(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Internal AI governance structure not documented in ProofLayer.",
        "Establish AI Governance Committee; define RACI for AI development, deployment, monitoring, and retirement.",
    )


def _a3_3(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Incident response roles not defined in ProofLayer.",
        "Appoint AI Incident Response roles with documented responsibilities and contact escalation path.",
    )


# ── Annex A.4: Resources ─────────────────────────────────────────────────────

def _a4_1(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Human resource competency records not in ProofLayer.",
        "Maintain skills inventory for AI roles; identify gaps; deliver targeted training.",
    )


def _a4_2(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    models = _qa(
        f"SELECT DISTINCT properties->>'model_id' AS model FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'model_id' IS NOT NULL{af}",
        ap,
    )
    if models:
        model_names = [r["model"] for r in models if r["model"]]
        return CheckResult(
            PARTIAL,
            f"Technical resources evidenced: models {model_names} in use.",
            "Document infrastructure: compute specs, API rate limits, failover strategy, cost controls.",
            {"models_in_use": model_names},
        )
    return CheckResult(
        NOT_MET, "Technical resources not documented in ProofLayer.",
        "Document AI infrastructure: model endpoints, hardware, networking, and redundancy.",
    )


def _a4_3(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Financial resource allocation not tracked in ProofLayer.",
        "Budget AI operations explicitly; track API costs; define approval process for budget changes.",
    )


# ── Annex A.5: Assessing Impacts ─────────────────────────────────────────────

def _a5_1(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Societal impact assessment not evidenced in ProofLayer.",
        "Conduct Societal AI Impact Assessment covering affected groups, potential harms, and mitigation measures.",
    )


def _a5_2(agent_name: str | None = None) -> CheckResult:
    snapshots = _count("SELECT COUNT(*) FROM pl_nodes WHERE node_type='ContextSnapshot'")
    if snapshots > 0:
        return CheckResult(
            PARTIAL,
            f"{snapshots} context snapshots provide per-decision impact evidence.",
            "Formalise impact assessment framework: define risk categories, scoring criteria, and review process.",
            {"context_snapshots": snapshots},
        )
    return CheckResult(
        NOT_MET, "No context snapshots — impact assessment framework missing.",
        "Define and implement AI Impact Assessment Framework; record per-use-case assessments.",
    )


def _a5_3(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    agents = _qa(
        f"SELECT DISTINCT agent_name FROM pl_nodes "
        f"WHERE node_type='Decision' AND agent_name IS NOT NULL{af}",
        ap,
    )
    if agents:
        return CheckResult(
            PARTIAL,
            f"Use cases identified via agents: {[r['agent_name'] for r in agents]}.",
            "Perform formal use case impact assessment for each agent, documenting risk and mitigations.",
            {"use_cases": [r["agent_name"] for r in agents]},
        )
    return CheckResult(
        NOT_MET, "No use cases identified for impact assessment.",
        "Identify all AI use cases; conduct impact assessment for each before deployment.",
    )


def _a5_4(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Impact assessment update tracking not in ProofLayer.",
        "Establish review trigger: reassess impacts when model, data, scope, or regulation changes.",
    )


# ── Annex A.6: AI System Lifecycle ───────────────────────────────────────────

def _a6_1(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "AI system lifecycle policy not evidenced in ProofLayer.",
        "Document lifecycle policy: design → test → deploy → monitor → retire with approval gates at each stage.",
    )


def _a6_2(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    models = _count(
        f"SELECT COUNT(DISTINCT properties->>'model_id') FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'model_id' IS NOT NULL{af}",
        ap,
    )
    if models >= 1:
        return CheckResult(
            PARTIAL,
            f"{models} model(s) deployed — operational stage evidenced.",
            "Document data acquisition plan for model fine-tuning and evaluation datasets.",
            {"deployed_models": models},
        )
    return CheckResult(NOT_MET, "No model deployment evidenced.", "Document model selection, acquisition, and version control.")


def _a6_3(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Data preparation controls not in ProofLayer.",
        "Document preprocessing pipeline: cleaning, normalisation, augmentation, validation steps with quality gates.",
    )


def _a6_4(agent_name: str | None = None) -> CheckResult:
    with_input = _count(
        "SELECT COUNT(*) FROM pl_nodes "
        "WHERE node_type='ContextSnapshot' AND properties->>'input_query' != '' "
        "AND properties->>'input_query' IS NOT NULL"
    )
    if with_input > 0:
        return CheckResult(
            PARTIAL,
            f"{with_input} context snapshots with captured input data — data quality partially evidenced.",
            "Define formal data quality criteria (completeness, accuracy, timeliness) and validation procedures.",
            {"inputs_captured": with_input},
        )
    return CheckResult(
        NOT_MET, "No input data captured in context snapshots.",
        "Capture and validate input data quality; define quality thresholds and rejection criteria.",
    )


# ── Annex A.7: Information for Users ─────────────────────────────────────────

def _a7_1(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "External AI system documentation not in ProofLayer.",
        "Publish user-facing documentation: what the AI does, its limitations, how to appeal decisions.",
    )


def _a7_2(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    if total >= 100:
        return CheckResult(
            MET,
            f"{total} decision records maintained — record keeping requirement met.",
            "Define retention policy for decision records; ensure GDPR-compliant deletion on request.",
            {"total_records": total},
        )
    elif total > 0:
        return CheckResult(
            PARTIAL, f"{total} records (target: 100+ for meaningful audit trail).",
            "Continue logging all decisions; define minimum retention period (recommend 3 years).",
            {"total_records": total},
        )
    return CheckResult(NOT_MET, "No records kept.", "Implement decision record keeping immediately.")


def _a7_3(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "AI system communication plan not in ProofLayer.",
        "Define communication plan for interested parties: regular AI performance reports, incident notifications.",
    )


def _a7_4(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "Intended use documentation not in ProofLayer.",
        "Publish clear intended use statement: supported languages, use cases, known limitations, contraindications.",
    )


def _a7_5(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "AI system labelling not evidenced in ProofLayer.",
        "Label all AI-generated content; inform users when decisions are AI-assisted vs. human-made.",
    )


# ── Annex A.8: Data and Information ──────────────────────────────────────────

def _a8_1(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "AI use policy for end users not in ProofLayer.",
        "Publish acceptable use policy: who may use the AI, approved use cases, prohibited activities.",
    )


def _a8_2(agent_name: str | None = None) -> CheckResult:
    return CheckResult(
        NOT_MET, "User guidance not evidenced in ProofLayer.",
        "Provide operator guidance: how to interpret AI outputs, when to override, how to report issues.",
    )


def _a8_3(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    overrides = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND (properties->>'human_override')::boolean = true{af}",
        ap,
    )
    escalations = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'decision_value'='escalation'{af}",
        ap,
    )
    total = overrides + escalations
    if total > 0:
        return CheckResult(
            MET,
            f"Human oversight active: {overrides} overrides + {escalations} escalations = {total} human interventions.",
            "Document human oversight procedures; train staff on override responsibility and documentation.",
            {"overrides": overrides, "escalations": escalations},
        )
    return CheckResult(
        NOT_MET, "No human oversight records found.",
        "Implement mandatory human review for high-stakes decisions; record all human interventions.",
    )


def _a8_4(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    recent_7d = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND created_at >= NOW() - INTERVAL '7 days'{af}",
        ap,
    )
    if recent_7d > 0:
        return CheckResult(
            MET,
            f"{recent_7d} decisions monitored in last 7 days — in-use monitoring active.",
            "Add automated anomaly detection; alert on confidence drops or unusual escalation spikes.",
            {"monitored_7d": recent_7d},
        )
    return CheckResult(
        NOT_MET, "No recent monitoring data.",
        "Implement real-time monitoring of AI system outputs with alerting thresholds.",
    )


# ── Annex A.9: AI System Lifecycle – Decision-Making ─────────────────────────

def _a9_1(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    with_reasoning = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'reasoning_summary' != '' "
        f"AND properties->>'reasoning_summary' IS NOT NULL{af}",
        ap,
    )
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    if total == 0:
        return CheckResult(NOT_MET, "No decision records.", "Log decisions with documented reasoning.")
    pct = round(with_reasoning / total * 100) if total else 0
    if pct >= 70:
        return CheckResult(
            MET,
            f"{pct}% of decisions have documented decision rationale.",
            "Ensure all automated decisions include a reasoning summary accessible for audit.",
            {"pct_with_reasoning": pct},
        )
    return CheckResult(
        PARTIAL,
        f"Only {pct}% of decisions have documented rationale.",
        "Capture reasoning_summary for all decisions; document how each decision was reached.",
        {"pct_with_reasoning": pct},
    )


def _a9_2(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    overrides = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND (properties->>'human_override')::boolean = true{af}",
        ap,
    )
    if overrides > 0:
        return CheckResult(
            PARTIAL,
            f"{overrides} human intervention records found — human oversight in AI decisions evidenced.",
            "Document formal human intervention procedures: when to intervene, who approves, how to record.",
            {"human_interventions": overrides},
        )
    return CheckResult(
        NOT_MET, "No human intervention in AI decision-making evidenced.",
        "Implement human-in-the-loop controls: define escalation criteria and document all interventions.",
    )


def _a9_3(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    if total >= 10:
        return CheckResult(
            PARTIAL,
            f"{total} AI decision records available for transparency review.",
            "Publish transparency report on AI decision-making: accuracy rates, bias analysis, audit findings.",
            {"total_decisions": total},
        )
    return CheckResult(
        NOT_MET, "Insufficient decision records for transparency reporting.",
        "Build decision audit trail; publish regular transparency reports on AI decision outcomes.",
    )


# ── Annex A.10: Improvement ────────────────────────────────────────────────────

def _a10_1(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    models = _qa(
        f"SELECT DISTINCT properties->>'model_id' AS model FROM pl_nodes "
        f"WHERE node_type='Decision'{af}",
        ap,
    )
    model_names = [r["model"] for r in models if r.get("model")]
    if model_names:
        return CheckResult(
            PARTIAL,
            f"Models in use: {model_names}. Responsible development partially evidenced.",
            "Document model selection criteria: fairness evaluation, bias testing, red-teaming results.",
            {"models": model_names},
        )
    return CheckResult(NOT_MET, "No model development evidence.", "Document responsible development practices for all AI models.")


def _a10_2(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    agents = _qa(
        f"SELECT DISTINCT agent_name FROM pl_nodes "
        f"WHERE node_type='Decision' AND agent_name IS NOT NULL{af}",
        ap,
    )
    multilingual = any("ar" in (r["agent_name"] or "") for r in agents)
    if multilingual:
        return CheckResult(
            PARTIAL,
            "Multi-language support (Arabic + English) evidenced — demographic coverage considered.",
            "Conduct fairness analysis across languages; test for differential error rates by language/demographic.",
            {"multilingual": True},
        )
    return CheckResult(
        NOT_MET, "No fairness or non-discrimination analysis evidenced.",
        "Test AI for discriminatory outcomes across protected characteristics; document and remediate bias.",
    )


def _a10_3(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    with_reasoning = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'reasoning_summary' != '' "
        f"AND properties->>'reasoning_summary' IS NOT NULL{af}",
        ap,
    )
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    if total == 0:
        return CheckResult(NOT_MET, "No decisions to assess.", "Log decisions to evaluate explainability.")
    pct = round(with_reasoning / total * 100) if total else 0
    if pct >= 70:
        return CheckResult(
            MET,
            f"{pct}% of decisions include reasoning summaries — explainability requirement met.",
            "Extend reasoning capture to all decisions; make summaries accessible to affected users on request.",
            {"explainability_pct": pct},
        )
    return CheckResult(
        PARTIAL,
        f"Only {pct}% of decisions have reasoning summaries — explainability gap.",
        "Capture reasoning_summary for all decisions; provide explainability on request.",
        {"explainability_pct": pct},
    )


def _a10_4(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    overrides = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND (properties->>'human_override')::boolean = true{af}",
        ap,
    )
    agents = _count(
        f"SELECT COUNT(DISTINCT agent_name) FROM pl_nodes "
        f"WHERE node_type='Decision' AND agent_name IS NOT NULL{af}",
        ap,
    )
    if agents >= 1:
        return CheckResult(
            PARTIAL,
            f"{agents} named agent(s) with {overrides} documented overrides — accountability partially met.",
            "Assign legal accountability for each AI system; document in AI system cards.",
            {"named_agents": agents, "overrides": overrides},
        )
    return CheckResult(NOT_MET, "Accountability not established.", "Assign named accountable owner for each AI system.")


def _a10_5(agent_name: str | None = None) -> CheckResult:
    af, ap = _af(agent_name)
    with_session = _count(
        f"SELECT COUNT(*) FROM pl_nodes "
        f"WHERE node_type='Decision' AND properties->>'session_id' != '' "
        f"AND properties->>'session_id' IS NOT NULL{af}",
        ap,
    )
    total = _count(f"SELECT COUNT(*) FROM pl_nodes WHERE node_type='Decision'{af}", ap)
    pct = round(with_session / total * 100) if total else 0
    if pct >= 80:
        return CheckResult(
            PARTIAL,
            f"{pct}% of decisions have session tracking — privacy traceability evidenced.",
            "Ensure session IDs do not expose PII; implement data minimisation and retention limits.",
            {"pct_with_session": pct},
        )
    return CheckResult(
        NOT_MET,
        f"Only {pct}% session tracking — privacy accountability gap.",
        "Implement privacy-by-design: pseudonymise session IDs, enforce retention limits, enable right-to-deletion.",
        {"pct_with_session": pct},
    )


# ═════════════════════════════════════════════════════════════════════════════
# Requirement catalogue — 65 requirements
# ═════════════════════════════════════════════════════════════════════════════

def _build_requirements(agent_name: str | None = None) -> list[Requirement]:
    def bind(fn: Callable) -> Callable:
        return partial(fn, agent_name=agent_name) if agent_name is not None else fn

    return [
        # ── Clause 4: Context ────────────────────────────────────────────────
        Requirement("4.1",   "Clause 4",  "Context of the Organization",             "Understanding the organization",      "Determine internal and external issues relevant to the AI management system.", CRITICAL, bind(_c4_1)),
        Requirement("4.2",   "Clause 4",  "Context of the Organization",             "Interested parties",                  "Identify interested parties and their requirements.", CRITICAL, bind(_c4_2)),
        Requirement("4.3",   "Clause 4",  "Context of the Organization",             "AIMS scope",                          "Determine the scope of the AI management system.", CRITICAL, bind(_c4_3)),
        Requirement("4.4",   "Clause 4",  "Context of the Organization",             "AI management system",                "Establish, implement, maintain, and continually improve the AIMS.", CRITICAL, bind(_c4_4)),
        # ── Clause 5: Leadership ─────────────────────────────────────────────
        Requirement("5.1",   "Clause 5",  "Leadership",                              "Leadership commitment",               "Top management demonstrates commitment to the AIMS.", CRITICAL, bind(_c5_1)),
        Requirement("5.2",   "Clause 5",  "Leadership",                              "AI policy",                          "Establish, implement, and maintain an AI policy.", CRITICAL, bind(_c5_2)),
        Requirement("5.3",   "Clause 5",  "Leadership",                              "Roles and responsibilities",          "Assign and communicate AIMS roles and responsibilities.", MAJOR, bind(_c5_3)),
        # ── Clause 6: Planning ───────────────────────────────────────────────
        Requirement("6.1.1", "Clause 6",  "Planning",                                "Risks and opportunities",             "Identify risks and opportunities related to AI systems.", CRITICAL, bind(_c6_1)),
        Requirement("6.1.2", "Clause 6",  "Planning",                                "AI risk assessment",                  "Conduct and document AI risk assessments.", CRITICAL, bind(_c6_2)),
        Requirement("6.1.3", "Clause 6",  "Planning",                                "Risk treatment",                      "Define and implement risk treatment options.", MAJOR, bind(_c6_3)),
        Requirement("6.1.4", "Clause 6",  "Planning",                                "Impact assessment planning",          "Plan AI system impact assessments.", MAJOR, bind(_c6_4)),
        Requirement("6.2",   "Clause 6",  "Planning",                                "AI objectives",                       "Establish AI objectives consistent with the AI policy.", MAJOR, bind(_c6_5)),
        # ── Clause 7: Support ────────────────────────────────────────────────
        Requirement("7.1",   "Clause 7",  "Support",                                 "Resources",                           "Determine and provide resources needed for the AIMS.", MAJOR, bind(_c7_1)),
        Requirement("7.2",   "Clause 7",  "Support",                                 "Competence",                          "Ensure persons are competent to perform AI-related work.", MAJOR, bind(_c7_2)),
        Requirement("7.3",   "Clause 7",  "Support",                                 "Awareness",                           "Ensure awareness of AI policy and personal contribution.", MINOR, bind(_c7_3)),
        Requirement("7.4",   "Clause 7",  "Support",                                 "Communication",                       "Determine internal/external communications about the AIMS.", MINOR, bind(_c7_4)),
        Requirement("7.5.1", "Clause 7",  "Support",                                 "Documented information",              "Maintain documented information required by ISO 42001.", MAJOR, bind(_c7_5)),
        Requirement("7.5.2", "Clause 7",  "Support",                                 "Document version control",            "Create and update documented information with controls.", MINOR, bind(_c7_6)),
        # ── Clause 8: Operation ──────────────────────────────────────────────
        Requirement("8.1",   "Clause 8",  "Operation",                               "Operational planning",                "Plan, implement, and control AI operational processes.", CRITICAL, bind(_c8_1)),
        Requirement("8.2",   "Clause 8",  "Operation",                               "AI impact assessment",                "Conduct impact assessments before deploying AI systems.", CRITICAL, bind(_c8_2)),
        Requirement("8.3",   "Clause 8",  "Operation",                               "AI system design",                    "Design AI systems with appropriate controls.", MAJOR, bind(_c8_3)),
        Requirement("8.4",   "Clause 8",  "Operation",                               "Data for AI systems",                 "Ensure data quality, provenance, and governance.", CRITICAL, bind(_c8_4)),
        Requirement("8.5",   "Clause 8",  "Operation",                               "AI system implementation",            "Implement AI systems with traceability and controls.", MAJOR, bind(_c8_5)),
        Requirement("8.6",   "Clause 8",  "Operation",                               "Third-party relations",               "Manage third-party AI providers and customer relationships.", MAJOR, bind(_c8_6)),
        # ── Clause 9: Performance ────────────────────────────────────────────
        Requirement("9.1",   "Clause 9",  "Performance Evaluation",                  "Monitoring and measurement",          "Monitor, measure, analyse, and evaluate AI performance.", CRITICAL, bind(_c9_1)),
        Requirement("9.2",   "Clause 9",  "Performance Evaluation",                  "Internal audit",                      "Conduct internal audits at planned intervals.", MAJOR, bind(_c9_2)),
        Requirement("9.3",   "Clause 9",  "Performance Evaluation",                  "Management review",                   "Top management reviews the AIMS at planned intervals.", MAJOR, bind(_c9_3)),
        # ── Clause 10: Improvement ───────────────────────────────────────────
        Requirement("10.1",  "Clause 10", "Improvement",                             "Nonconformity and corrective action", "React to nonconformities and implement corrective actions.", MAJOR, bind(_c10_1)),
        Requirement("10.2",  "Clause 10", "Improvement",                             "Continual improvement",              "Continually improve the suitability and effectiveness of the AIMS.", MAJOR, bind(_c10_2)),
        Requirement("10.3",  "Clause 10", "Improvement",                             "Improvement indicators",             "Track leading indicators of improvement.", MINOR, bind(_c10_3)),
        # ── Annex A.2: Policies for AI ───────────────────────────────────────
        Requirement("A.2.1", "Annex A",   "A.2 Policies for AI",                    "Responsible use policy",              "Establish policies for responsible use of AI.", CRITICAL, bind(_a2_1)),
        Requirement("A.2.2", "Annex A",   "A.2 Policies for AI",                    "Policy enforcement",                  "Enforce AI policies consistently across all systems.", MAJOR, bind(_a2_2)),
        Requirement("A.2.3", "Annex A",   "A.2 Policies for AI",                    "Internal use guidelines",             "Define guidelines for internal use of AI systems.", MAJOR, bind(_a2_3)),
        Requirement("A.2.4", "Annex A",   "A.2 Policies for AI",                    "Ethics review",                       "Conduct ethics reviews for AI system deployments.", MAJOR, bind(_a2_4)),
        # ── Annex A.3: Internal Organisation ─────────────────────────────────
        Requirement("A.3.1", "Annex A",   "A.3 Internal Organization",              "Division of responsibilities",        "Divide AI responsibilities across roles.", MAJOR, bind(_a3_1)),
        Requirement("A.3.2", "Annex A",   "A.3 Internal Organization",              "Internal AI governance",              "Establish internal AI governance structure.", CRITICAL, bind(_a3_2)),
        Requirement("A.3.3", "Annex A",   "A.3 Internal Organization",              "Incident response roles",             "Assign AI incident response roles.", MAJOR, bind(_a3_3)),
        # ── Annex A.4: Resources ─────────────────────────────────────────────
        Requirement("A.4.1", "Annex A",   "A.4 Resources",                           "Human resources",                    "Ensure adequate human resources for AI management.", MAJOR, bind(_a4_1)),
        Requirement("A.4.2", "Annex A",   "A.4 Resources",                           "Technical resources",                "Provision and document technical resources for AI.", MAJOR, bind(_a4_2)),
        Requirement("A.4.3", "Annex A",   "A.4 Resources",                           "Financial resources",                "Allocate sufficient financial resources for the AIMS.", MINOR, bind(_a4_3)),
        # ── Annex A.5: Assessing Impacts ─────────────────────────────────────
        Requirement("A.5.1", "Annex A",   "A.5 Assessing Impacts",                  "Societal impact assessment",          "Assess societal impacts of AI systems.", CRITICAL, bind(_a5_1)),
        Requirement("A.5.2", "Annex A",   "A.5 Assessing Impacts",                  "Impact assessment framework",         "Establish an AI impact assessment framework.", CRITICAL, bind(_a5_2)),
        Requirement("A.5.3", "Annex A",   "A.5 Assessing Impacts",                  "Use case impact assessment",          "Assess impacts for each AI use case.", MAJOR, bind(_a5_3)),
        Requirement("A.5.4", "Annex A",   "A.5 Assessing Impacts",                  "Impact assessment updates",           "Update impact assessments when conditions change.", MAJOR, bind(_a5_4)),
        # ── Annex A.6: AI System Lifecycle ───────────────────────────────────
        Requirement("A.6.1", "Annex A",   "A.6 AI System Lifecycle",                "Lifecycle policy",                    "Define AI system operational and maintenance policy.", MAJOR, bind(_a6_1)),
        Requirement("A.6.2", "Annex A",   "A.6 AI System Lifecycle",                "Data acquisition plan",               "Document data acquisition and model training plan.", MAJOR, bind(_a6_2)),
        Requirement("A.6.3", "Annex A",   "A.6 AI System Lifecycle",                "Data preparation controls",           "Implement controls for data preparation and processing.", MAJOR, bind(_a6_3)),
        Requirement("A.6.4", "Annex A",   "A.6 AI System Lifecycle",                "Data quality",                        "Define and enforce data quality criteria.", CRITICAL, bind(_a6_4)),
        # ── Annex A.7: Information for Users ─────────────────────────────────
        Requirement("A.7.1", "Annex A",   "A.7 Information for Users",              "AI system documentation",             "Provide documentation about the AI system to interested parties.", MAJOR, bind(_a7_1)),
        Requirement("A.7.2", "Annex A",   "A.7 Information for Users",              "Record keeping",                      "Maintain records of AI system decisions and operations.", CRITICAL, bind(_a7_2)),
        Requirement("A.7.3", "Annex A",   "A.7 Information for Users",              "Communication plan",                  "Communicate AI system information to interested parties.", MINOR, bind(_a7_3)),
        Requirement("A.7.4", "Annex A",   "A.7 Information for Users",              "Intended use disclosure",             "Communicate intended use and limitations of AI systems.", MAJOR, bind(_a7_4)),
        Requirement("A.7.5", "Annex A",   "A.7 Information for Users",              "Labelling and marking",               "Label AI-generated content and decisions appropriately.", MAJOR, bind(_a7_5)),
        # ── Annex A.8: Data and Information ──────────────────────────────────
        Requirement("A.8.1", "Annex A",   "A.8 Data and Information",               "AI use policy",                       "Define policy for acceptable use of AI systems.", MAJOR, bind(_a8_1)),
        Requirement("A.8.2", "Annex A",   "A.8 Data and Information",               "User guidance",                       "Provide guidance to operators on using AI system outputs.", MAJOR, bind(_a8_2)),
        Requirement("A.8.3", "Annex A",   "A.8 Data and Information",               "Human oversight",                     "Ensure human oversight of AI decisions.", CRITICAL, bind(_a8_3)),
        Requirement("A.8.4", "Annex A",   "A.8 Data and Information",               "In-use monitoring",                   "Monitor AI system behaviour during operation.", MAJOR, bind(_a8_4)),
        # ── Annex A.9: AI System Lifecycle – Decision-Making ─────────────────
        Requirement("A.9.1", "Annex A",   "A.9 AI System Lifecycle – Decision-Making", "Decision documentation",           "Document the basis and rationale for AI decisions.", CRITICAL, bind(_a9_1)),
        Requirement("A.9.2", "Annex A",   "A.9 AI System Lifecycle – Decision-Making", "Human intervention records",       "Record human interventions in AI decision-making.", MAJOR, bind(_a9_2)),
        Requirement("A.9.3", "Annex A",   "A.9 AI System Lifecycle – Decision-Making", "Decision transparency",            "Ensure AI decision-making is transparent and auditable.", MAJOR, bind(_a9_3)),
        # ── Annex A.10: Improvement ───────────────────────────────────────────
        Requirement("A.10.1","Annex A",   "Improvement",                             "Responsible development",             "Develop AI systems responsibly with documented practices.", CRITICAL, bind(_a10_1)),
        Requirement("A.10.2","Annex A",   "Improvement",                             "Fairness",                            "Ensure AI systems are fair and non-discriminatory.", CRITICAL, bind(_a10_2)),
        Requirement("A.10.3","Annex A",   "Improvement",                             "Transparency and explainability",     "Make AI decisions transparent and explainable.", CRITICAL, bind(_a10_3)),
        Requirement("A.10.4","Annex A",   "Improvement",                             "Accountability",                      "Establish accountability for AI system outcomes.", CRITICAL, bind(_a10_4)),
        Requirement("A.10.5","Annex A",   "Improvement",                             "Privacy and security",                "Protect privacy and security in AI systems.", CRITICAL, bind(_a10_5)),
    ]


# ═════════════════════════════════════════════════════════════════════════════
# Report generators
# ═════════════════════════════════════════════════════════════════════════════

_STATUS_SCORE = {MET: 1.0, PARTIAL: 0.5, NOT_MET: 0.0, NA: None}


def generate_iso_42001_report(agent_name: str | None = None) -> dict[str, Any]:
    """Run all 65 requirements and return the full gap analysis report."""
    requirements = _build_requirements(agent_name)
    results = []
    weighted_score = 0.0
    weighted_total = 0.0
    section_stats: dict[str, dict] = {}

    for req in requirements:
        try:
            result = req.check_fn()
        except Exception as exc:
            result = CheckResult(
                NOT_MET,
                f"Evidence query failed: {exc}",
                "Investigate database connectivity or schema issues.",
            )

        score = _STATUS_SCORE.get(result.status)
        if score is not None:
            weighted_score += score * req.weight
            weighted_total += req.weight

        sec = req.section
        if sec not in section_stats:
            section_stats[sec] = {"met": 0, "partial": 0, "not_met": 0, "na": 0, "clause": req.clause}
        key = result.status.lower().replace("-", "_")
        section_stats[sec][key] = section_stats[sec].get(key, 0) + 1

        results.append({
            "req_id": req.req_id,
            "clause": req.clause,
            "section": req.section,
            "title": req.title,
            "description": req.description,
            "weight": req.weight,
            "status": result.status,
            "evidence": result.evidence,
            "recommendation": result.recommendation,
            "data": result.data,
        })

    overall_pct = round(weighted_score / weighted_total * 100, 1) if weighted_total else 0.0
    grade = _grade(overall_pct)
    counts = {
        "met":     sum(1 for r in results if r["status"] == MET),
        "partial": sum(1 for r in results if r["status"] == PARTIAL),
        "not_met": sum(1 for r in results if r["status"] == NOT_MET),
        "na":      sum(1 for r in results if r["status"] == NA),
    }

    # Derive per-axis scores for core vs annex
    core_ws = annex_ws = core_wt = annex_wt = 0.0
    for req, r in zip(requirements, results):
        score = _STATUS_SCORE.get(r["status"])
        if score is not None:
            if req.clause.startswith("Clause"):
                core_ws += score * req.weight; core_wt += req.weight
            else:
                annex_ws += score * req.weight; annex_wt += req.weight

    return {
        "standard": "ISO/IEC 42001:2023",
        "agent_filter": agent_name,
        "total_requirements": len(results),
        "overall_score_pct": overall_pct,
        "score_grade": grade,
        "counts": counts,
        "core_clauses_pct":   round(core_ws / core_wt * 100, 1) if core_wt else 0.0,
        "annex_controls_pct": round(annex_ws / annex_wt * 100, 1) if annex_wt else 0.0,
        "section_breakdown": section_stats,
        "requirements": results,
    }


def generate_compliance_summary(agent_name: str | None = None) -> dict[str, Any]:
    """Quick scorecard — designed for the admin dashboard header."""
    requirements = _build_requirements(agent_name)
    weighted_score = core_score = annex_score = 0.0
    weighted_total = core_total = annex_total = 0.0
    counts: dict[str, int] = {MET: 0, PARTIAL: 0, NOT_MET: 0, NA: 0}

    for req in requirements:
        try:
            result = req.check_fn()
        except Exception:
            result = CheckResult(NOT_MET, "", "")

        score = _STATUS_SCORE.get(result.status)
        counts[result.status] = counts.get(result.status, 0) + 1

        if score is not None:
            weighted_score += score * req.weight
            weighted_total += req.weight
            if req.clause.startswith("Clause"):
                core_score += score * req.weight; core_total += req.weight
            else:
                annex_score += score * req.weight; annex_total += req.weight

    overall_pct = round(weighted_score / weighted_total * 100, 1) if weighted_total else 0.0
    return {
        "standard": "ISO/IEC 42001:2023",
        "agent_filter": agent_name,
        "overall_score_pct": overall_pct,
        "score_grade": _grade(overall_pct),
        "requirements_met":     counts[MET],
        "requirements_partial": counts[PARTIAL],
        "requirements_not_met": counts[NOT_MET],
        "requirements_na":      counts[NA],
        "total_requirements":   len(requirements),
        "core_clauses_pct":     round(core_score / core_total * 100, 1) if core_total else 0.0,
        "annex_controls_pct":   round(annex_score / annex_total * 100, 1) if annex_total else 0.0,
    }


def generate_remediation_checklist(agent_name: str | None = None) -> dict[str, Any]:
    """Prioritised remediation checklist grouping requirements by action type.

    quick_wins         — PARTIAL: evidence exists; seed more data or add config
    documentation_needed — NOT_MET: no evidence; requires external docs / process
    already_compliant  — MET: nothing to do
    """
    requirements = _build_requirements(agent_name)
    quick_wins: list[dict] = []
    doc_needed: list[dict] = []
    compliant:  list[dict] = []

    for req in requirements:
        try:
            result = req.check_fn()
        except Exception:
            result = CheckResult(NOT_MET, "", "")

        item = {
            "req_id":         req.req_id,
            "clause":         req.clause,
            "section":        req.section,
            "title":          req.title,
            "evidence":       result.evidence,
            "recommendation": result.recommendation,
        }

        if result.status == MET:
            compliant.append(item)
        elif result.status == PARTIAL:
            quick_wins.append(item)
        else:
            doc_needed.append(item)

    return {
        "agent_filter":         agent_name,
        "quick_wins":           quick_wins,
        "documentation_needed": doc_needed,
        "already_compliant":    compliant,
        "total_actionable":     len(quick_wins) + len(doc_needed),
    }


def _grade(pct: float) -> str:
    if pct >= 90: return "A"
    if pct >= 75: return "B"
    if pct >= 60: return "C"
    if pct >= 40: return "D"
    return "F"
