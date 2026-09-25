"""ProofLayer SDAIA — FastAPI router (/api/v1/sdaia).

Queries the SDAIA compliance tables (agents, risk_assessments, decisions,
incidents, safety_reports, ethics_labels — see prooflayer-sdaia/app/schema_sdaia.sql)
directly via psycopg2, reusing the same triage_agent database connection
as the rest of this app (app/database.py).

This is a governance *scaffold* — see prooflayer-sdaia/app/*.py docstrings
for the same caveat: table/field names follow a plausible compliance
structure, not a verified SDAIA data model.
"""
from __future__ import annotations

import psycopg2.extras
from fastapi import APIRouter, HTTPException, Query

from app.database import get_conn

router = APIRouter(prefix="/api/v1/sdaia", tags=["SDAIA"])

# §7.2.1-7.2.4 of the SDAIA Responsible AI Policy: مخاطر حرجة (Critical) /
# عالية (High) / محدودة (Limited) / بسيطة أو منعدمة (Minimal), lowest to
# highest — must match prooflayer-sdaia/app/sdaia_risk.py LEVELS.
_LEVEL_RANK = {"MINIMAL": 0, "LIMITED": 1, "HIGH": 2, "CRITICAL": 3}


def _cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def _latest_risk_overall_by_agent(cur) -> dict[str, dict]:
    """Latest risk-assessment batch per agent, reduced to an overall level.

    Each classify_agent() run writes one row per category with the same
    assessed_at timestamp, so "latest batch" = rows sharing the max
    assessed_at per agent.
    """
    cur.execute(
        """
        WITH latest AS (
            SELECT agent_id, MAX(assessed_at) AS latest_ts
            FROM risk_assessments
            GROUP BY agent_id
        )
        SELECT ra.agent_id, ra.category, ra.level, ra.rationale, ra.assessed_at
        FROM risk_assessments ra
        JOIN latest l ON l.agent_id = ra.agent_id AND l.latest_ts = ra.assessed_at
        """
    )
    by_agent: dict[str, dict] = {}
    for row in cur.fetchall():
        entry = by_agent.setdefault(
            row["agent_id"], {"overall": "MINIMAL", "categories": [], "assessed_at": row["assessed_at"]}
        )
        entry["categories"].append(
            {"category": row["category"], "level": row["level"], "rationale": row["rationale"]}
        )
        if _LEVEL_RANK[row["level"]] > _LEVEL_RANK[entry["overall"]]:
            entry["overall"] = row["level"]
    return by_agent


def _latest_ethics_label_by_agent(cur) -> dict[str, dict]:
    cur.execute(
        """
        SELECT DISTINCT ON (agent_id) agent_id, tier, tier_name_ar, tier_name_en, score, computed_at
        FROM ethics_labels
        ORDER BY agent_id, computed_at DESC
        """
    )
    return {row["agent_id"]: row for row in cur.fetchall()}


@router.get("/overview")
def sdaia_overview() -> dict:
    """KPI overview for the SDAIA compliance dashboard."""
    with get_conn() as conn:
        with _cursor(conn) as cur:
            risk_by_agent = _latest_risk_overall_by_agent(cur)
            ethics_by_agent = _latest_ethics_label_by_agent(cur)

            cur.execute("SELECT COUNT(*) AS n FROM incidents WHERE resolved_at IS NULL")
            open_incidents = cur.fetchone()["n"]

            cur.execute(
                "SELECT COUNT(*) AS n FROM incidents WHERE severity = 'CRITICAL' AND resolved_at IS NULL"
            )
            critical_open_incidents = cur.fetchone()["n"]

            cur.execute("SELECT COUNT(*) AS n FROM incidents")
            total_incidents = cur.fetchone()["n"]

            cur.execute("SELECT COUNT(*) AS n FROM safety_reports")
            total_safety_reports = cur.fetchone()["n"]

            cur.execute("SELECT COUNT(*) AS n FROM incidents WHERE reported_to_regulator = TRUE")
            reported_to_regulator = cur.fetchone()["n"]

    risk_distribution: dict[str, int] = {}
    for entry in risk_by_agent.values():
        risk_distribution[entry["overall"]] = risk_distribution.get(entry["overall"], 0) + 1

    ethics_distribution: dict[str, int] = {}
    for label in ethics_by_agent.values():
        ethics_distribution[label["tier_name_en"]] = ethics_distribution.get(label["tier_name_en"], 0) + 1

    return {
        "agents_classified": len(risk_by_agent),
        "open_incidents": open_incidents,
        "critical_incidents": critical_open_incidents,
        "total_incidents": total_incidents,
        "total_safety_reports": total_safety_reports,
        "ethics_labels_assigned": len(ethics_by_agent),
        "reported_to_regulator": reported_to_regulator,
        "risk_distribution": risk_distribution,
        "ethics_distribution": ethics_distribution,
    }


@router.get("/agents")
def sdaia_agents() -> list[dict]:
    """All SDAIA-registered agents with their latest risk level + ethics label."""
    with get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute("SELECT * FROM agents ORDER BY created_at DESC")
            agents = cur.fetchall()
            risk_by_agent = _latest_risk_overall_by_agent(cur)
            ethics_by_agent = _latest_ethics_label_by_agent(cur)

            out = []
            for a in agents:
                cur.execute(
                    "SELECT COUNT(*) AS n FROM incidents WHERE agent_id = %s AND resolved_at IS NULL",
                    (a["agent_id"],),
                )
                open_incidents = cur.fetchone()["n"]
                cur.execute(
                    "SELECT COUNT(*) AS n FROM safety_reports WHERE agent_id = %s", (a["agent_id"],)
                )
                safety_report_count = cur.fetchone()["n"]

                risk = risk_by_agent.get(a["agent_id"])
                ethics = ethics_by_agent.get(a["agent_id"])
                out.append(
                    {
                        **dict(a),
                        "risk_level": risk["overall"] if risk else None,
                        "ethics_tier": ethics["tier"] if ethics else None,
                        "ethics_label_en": ethics["tier_name_en"] if ethics else None,
                        "ethics_label_ar": ethics["tier_name_ar"] if ethics else None,
                        "open_incidents": open_incidents,
                        "safety_report_count": safety_report_count,
                    }
                )
    return out


@router.get("/agents/{agent_id}")
def sdaia_agent_detail(agent_id: str) -> dict:
    """Full SDAIA detail for a single agent: risk categories, incidents,
    safety reports, ethics label."""
    with get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute("SELECT * FROM agents WHERE agent_id = %s", (agent_id,))
            agent = cur.fetchone()
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")

            risk_by_agent = _latest_risk_overall_by_agent(cur)
            risk = risk_by_agent.get(agent_id)

            cur.execute(
                "SELECT * FROM incidents WHERE agent_id = %s ORDER BY detected_at DESC LIMIT 20",
                (agent_id,),
            )
            incidents = cur.fetchall()

            cur.execute(
                "SELECT * FROM safety_reports WHERE agent_id = %s ORDER BY generated_at DESC LIMIT 10",
                (agent_id,),
            )
            safety_reports = cur.fetchall()

            ethics_by_agent = _latest_ethics_label_by_agent(cur)
            ethics_label = ethics_by_agent.get(agent_id)

    return {
        "agent": agent,
        "risk_overall": risk["overall"] if risk else None,
        "risk_categories": risk["categories"] if risk else [],
        "incidents": incidents,
        "safety_reports": safety_reports,
        "ethics_label": ethics_label,
    }


@router.get("/incidents")
def sdaia_incidents(
    open_only: bool = Query(False, description="Only unresolved incidents"),
    severity: str | None = Query(None, description="Filter: MINIMAL, LIMITED, HIGH, CRITICAL"),
    agent_id: str | None = Query(None, description="Filter by agent"),
    limit: int = Query(50, description="Max results"),
) -> list[dict]:
    """List SDAIA incidents with optional filters."""
    where = ["1=1"]
    params: list = []
    if open_only:
        where.append("i.resolved_at IS NULL")
    if severity:
        where.append("i.severity = %s")
        params.append(severity)
    if agent_id:
        where.append("i.agent_id = %s")
        params.append(agent_id)
    params.append(limit)

    with get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT i.*, a.name AS agent_name
                FROM incidents i
                LEFT JOIN agents a ON a.agent_id = i.agent_id
                WHERE {' AND '.join(where)}
                ORDER BY i.detected_at DESC
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()


@router.get("/safety-reports")
def sdaia_safety_reports(
    agent_id: str | None = Query(None, description="Filter by agent"),
    limit: int = Query(20, description="Max results"),
) -> list[dict]:
    """List SDAIA safety reports."""
    where = "sr.agent_id = %s" if agent_id else "1=1"
    params = [agent_id, limit] if agent_id else [limit]

    with get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                f"""
                SELECT sr.*, a.name AS agent_name
                FROM safety_reports sr
                LEFT JOIN agents a ON a.agent_id = sr.agent_id
                WHERE {where}
                ORDER BY sr.generated_at DESC
                LIMIT %s
                """,
                params,
            )
            return cur.fetchall()


# ═══════════════════════════════════════════════════════════════════════════
# Ethics Label Auto-Assessment
# ═══════════════════════════════════════════════════════════════════════════
# Evidence-based checklist taxonomy for the SDAIA Responsible AI Policy §6.8
# 5-tier scale (واعٍ/Aware -> متبنٍ/Adopting -> ملتزم/Committed -> موثوق/Trusted
# -> رائد/Leading), separate from the weighted-score model in
# prooflayer-sdaia/app/sdaia_labels.py — both write to the same ethics_labels
# table (tier/tier_name_ar/tier_name_en/score); whichever ran most recently
# for an agent is "current" (existing DISTINCT ON ... ORDER BY computed_at
# DESC lookups already pick the latest row, so no extra versioning column is
# needed here). The two vocabularies must stay in sync — see the module
# docstring in sdaia_labels.py. Evidence signals are read from real columns
# already in this schema; items with no real signal (continuous_monitoring,
# third_party_test, cross_agent_governance, benchmarking,
# standards_contribution) are left False rather than guessed.

ETHICS_LABELS_EN = ["aware", "adopting", "committed", "trusted", "leading"]
ETHICS_LABELS_AR = {
    "aware": "واعٍ", "adopting": "متبنٍ",
    "committed": "ملتزم", "trusted": "موثوق", "leading": "رائد",
}

# §7.2.2/§7.2.3: minimum ethics-label tier (1-indexed into ETHICS_LABELS_EN)
# required per risk level before deployment. §7.2.4 "prefers" (does not
# require) Aware for Minimal risk, so it's omitted here (nothing to gate).
# §7.8 bans Critical risk outright regardless of label — handled separately.
_MIN_LABEL_TIER_FOR_RISK = {
    "HIGH": 4,     # موثوق / Trusted
    "LIMITED": 3,  # ملتزم / Committed
}

LABEL_REQUIREMENTS = {
    "aware": ["agent_registered", "risk_classified"],
    "adopting": ["pii_classified", "safety_barriers"],
    "committed": ["safety_reports_done", "human_oversight", "incident_response"],
    "trusted": ["continuous_monitoring", "incident_reported", "third_party_test"],
    "leading": ["cross_agent_governance", "benchmarking", "standards_contribution"],
}

LABEL_PREREQUISITES = {
    "aware": [], "adopting": ["aware"],
    "committed": ["aware", "adopting"],
    "trusted": ["aware", "adopting", "committed"],
    "leading": ["aware", "adopting", "committed", "trusted"],
}


def _gather_label_evidence(cur, agent_id: str) -> dict:
    evidence = {}

    cur.execute("SELECT COUNT(*) AS n FROM agents WHERE agent_id = %s", (agent_id,))
    evidence["agent_registered"] = cur.fetchone()["n"] > 0

    cur.execute("SELECT COUNT(*) AS n FROM risk_assessments WHERE agent_id = %s", (agent_id,))
    evidence["risk_classified"] = cur.fetchone()["n"] > 0

    cur.execute("SELECT handles_pii FROM agents WHERE agent_id = %s", (agent_id,))
    row = cur.fetchone()
    evidence["pii_classified"] = row is not None and row["handles_pii"] is not None

    cur.execute("SELECT autonomy_level FROM agents WHERE agent_id = %s", (agent_id,))
    row = cur.fetchone()
    evidence["safety_barriers"] = bool(row and row["autonomy_level"] in ("assisted", "supervised"))

    cur.execute("SELECT COUNT(*) AS n FROM safety_reports WHERE agent_id = %s", (agent_id,))
    evidence["safety_reports_done"] = cur.fetchone()["n"] > 0

    cur.execute(
        "SELECT COUNT(*) AS n FROM decisions WHERE agent_id = %s AND requires_human_review = TRUE",
        (agent_id,),
    )
    evidence["human_oversight"] = cur.fetchone()["n"] > 0

    cur.execute(
        "SELECT COUNT(*) AS n FROM incidents WHERE agent_id = %s AND resolved_at IS NOT NULL",
        (agent_id,),
    )
    evidence["incident_response"] = cur.fetchone()["n"] > 0

    evidence["continuous_monitoring"] = False

    cur.execute(
        "SELECT COUNT(*) AS n FROM incidents WHERE agent_id = %s AND reported_to_regulator = TRUE",
        (agent_id,),
    )
    evidence["incident_reported"] = cur.fetchone()["n"] > 0

    evidence["third_party_test"] = False
    evidence["cross_agent_governance"] = False
    evidence["benchmarking"] = False
    evidence["standards_contribution"] = False

    return evidence


def _get_all_reqs(level: str) -> list[str]:
    seen: set[str] = set()
    reqs: list[str] = []
    for prereq in LABEL_PREREQUISITES.get(level, []):
        for r in LABEL_REQUIREMENTS.get(prereq, []):
            if r not in seen:
                seen.add(r)
                reqs.append(r)
    for r in LABEL_REQUIREMENTS.get(level, []):
        if r not in seen:
            seen.add(r)
            reqs.append(r)
    return reqs


@router.post("/agents/{agent_id}/assess-label")
def sdaia_assess_label(agent_id: str) -> dict:
    """Scan real evidence for an agent and assign the highest ethics label
    whose full requirement chain (including prerequisite levels) is met."""
    with get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute("SELECT name FROM agents WHERE agent_id = %s", (agent_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            agent_name = row["name"]

            evidence = _gather_label_evidence(cur, agent_id)

            achieved = None
            for level in reversed(ETHICS_LABELS_EN):
                missing = [r for r in _get_all_reqs(level) if not evidence.get(r, False)]
                if not missing:
                    achieved = level
                    break

            if not achieved:
                return {
                    "agent_id": agent_id,
                    "agent_name": agent_name,
                    "status": "not_assessed",
                    "current_label": None,
                    "current_label_ar": "غير مصنف",
                    "evidence_summary": evidence,
                }

            tier = ETHICS_LABELS_EN.index(achieved) + 1
            score = round(100 * sum(1 for v in evidence.values() if v) / len(evidence))

            cur.execute(
                "INSERT INTO ethics_labels (agent_id, tier, tier_name_ar, tier_name_en, score) "
                "VALUES (%s, %s, %s, %s, %s)",
                (agent_id, tier, ETHICS_LABELS_AR[achieved], achieved.capitalize(), score),
            )
            conn.commit()

            next_idx = ETHICS_LABELS_EN.index(achieved) + 1
            next_level = ETHICS_LABELS_EN[next_idx] if next_idx < len(ETHICS_LABELS_EN) else None
            next_gaps = (
                [r for r in _get_all_reqs(next_level) if not evidence.get(r, False)]
                if next_level else []
            )

    return {
        "agent_id": agent_id,
        "agent_name": agent_name,
        "current_label": achieved,
        "current_label_ar": ETHICS_LABELS_AR[achieved],
        "score": score,
        "next_level": next_level,
        "next_level_ar": ETHICS_LABELS_AR.get(next_level, "") if next_level else "",
        "next_gaps": next_gaps,
        "evidence_summary": evidence,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Pre-Deployment Gate
# ═══════════════════════════════════════════════════════════════════════════
# The first two checks below are direct enforcement of the policy's actual
# launch rules (§7.8, §7.2.2, §7.2.3) — cross-referencing risk level against
# ethics-label tier, not a single evidence flag, so they're computed
# separately in sdaia_deployment_check rather than listed in
# PRE_DEPLOYMENT_CHECKS. The remaining items reuse _gather_label_evidence so
# the same underlying signals back both the ethics label and this gate;
# their IDs are plain sequence numbers, not section citations.

PRE_DEPLOYMENT_CHECKS = [
    {"id": "3",  "title": "Safety barriers active (human-in-the-loop autonomy)", "key": "safety_barriers",      "severity": "blocker"},
    {"id": "4",  "title": "Safety report on file",                              "key": "safety_reports_done",  "severity": "blocker"},
    {"id": "5",  "title": "Misuse/oversight scenarios exercised",               "key": "human_oversight",      "severity": "blocker"},
    {"id": "6",  "title": "Incident response demonstrated",                     "key": "incident_response",    "severity": "blocker"},
    {"id": "7",  "title": "Expert / safety-report review cadence",              "key": "safety_reports_done",  "severity": "warning"},
    {"id": "8",  "title": "Data classification (PII) assessed",                 "key": "pii_classified",       "severity": "blocker"},
    {"id": "9",  "title": "Risk classification complete",                      "key": "risk_classified",      "severity": "blocker"},
    {"id": "10", "title": "Risk monitoring active",                            "key": "risk_classified",      "severity": "warning"},
    {"id": "11", "title": "Continuous post-deployment monitoring",             "key": "continuous_monitoring","severity": "warning"},
    {"id": "12", "title": "Periodic update / re-assessment cadence",           "key": "safety_reports_done",  "severity": "info"},
]


def compute_deployment_decision(cur, agent_id: str, agent_name: str) -> dict:
    """Core §7.8 + §7.2 + evidence-checklist gate computation.

    Pulled out of the route handler so other callers (e.g. app/passport_api.py's
    Revocation Status field) can get the exact same go/no-go decision instead
    of re-deriving a simplified version that could silently drift out of sync
    with this one -- there is exactly one place this policy logic is allowed
    to live.

    §7.8 and §7.2.2/§7.2.3 are enforced as hard, non-negotiable checks
    (items 1-2) ahead of the evidence checklist: a Critical-risk agent is
    always blocked regardless of anything else, and a High/Limited-risk
    agent needs its current ethics-label tier to already meet the policy's
    minimum before the rest of the checklist even matters.
    """
    evidence = _gather_label_evidence(cur, agent_id)

    risk = _latest_risk_overall_by_agent(cur).get(agent_id)
    risk_level = risk["overall"] if risk else None
    ethics = _latest_ethics_label_by_agent(cur).get(agent_id)
    ethics_tier = ethics["tier"] if ethics else 0

    items = []
    blockers = warnings = passed = 0

    # §7.8 — launching a Critical-risk system is prohibited outright, full
    # stop, no matter how well-governed it is otherwise.
    critical_ok = risk_level != "CRITICAL"
    items.append({
        "id": "1",
        "title": "Not classified Critical risk (§7.8 — Critical-risk systems may never launch)",
        "severity": "blocker",
        "status": "pass" if critical_ok else "fail",
    })
    if critical_ok:
        passed += 1
    else:
        blockers += 1

    # §7.2.2/§7.2.3 — High risk requires Trusted (tier 4) at minimum, Limited
    # risk requires Committed (tier 3) at minimum, before deployment.
    min_tier = _MIN_LABEL_TIER_FOR_RISK.get(risk_level)
    label_ok = min_tier is None or ethics_tier >= min_tier
    min_tier_name = ETHICS_LABELS_AR.get(
        ETHICS_LABELS_EN[min_tier - 1], ""
    ) if min_tier else None
    items.append({
        "id": "2",
        "title": (
            f"Ethics label meets §7.2 minimum for {risk_level} risk"
            f" (requires ≥ {min_tier_name})" if min_tier else
            "Ethics label meets §7.2 minimum for current risk level (none required)"
        ),
        "severity": "blocker",
        "status": "pass" if label_ok else "fail",
    })
    if label_ok:
        passed += 1
    else:
        blockers += 1

    for check in PRE_DEPLOYMENT_CHECKS:
        ok = evidence.get(check["key"], False)
        items.append({
            "id": check["id"],
            "title": check["title"],
            "severity": check["severity"],
            "status": "pass" if ok else "fail",
        })
        if ok:
            passed += 1
        elif check["severity"] == "blocker":
            blockers += 1
        elif check["severity"] == "warning":
            warnings += 1

    if not critical_ok:
        decision, reason = "prohibited", "Critical-risk systems may not launch under §7.8, regardless of governance evidence"
    elif blockers:
        decision, reason = "blocked", f"{blockers} blocker(s) unresolved"
    elif warnings:
        decision, reason = "warning", f"{warnings} warning(s) — deploy with caution"
    else:
        decision, reason = "approved", "All checks pass"

    return {
        "agent_id": agent_id,
        "risk_level": risk_level,
        "ethics_tier": ethics_tier or None,
        "agent_name": agent_name,
        "decision": decision,
        "reason": reason,
        "summary": {"passed": passed, "blockers": blockers, "warnings": warnings, "total": len(items)},
        "items": items,
    }


@router.get("/agents/{agent_id}/deployment-check")
def sdaia_deployment_check(agent_id: str) -> dict:
    """Run the pre-deployment checklist and return a go/no-go decision."""
    with get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute("SELECT name FROM agents WHERE agent_id = %s", (agent_id,))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
            return compute_deployment_decision(cur, agent_id, row["name"])
