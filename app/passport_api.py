"""ProofLayer Agent Passport -- FastAPI router (/api/v1/passport).

Assembles a single per-agent credential card from data already recorded
elsewhere in ProofLayer: the pl_agents behavioral-contract registry, and
(when the same agent name also has an SDAIA compliance record) risk
classification, ethics label, incident history, and safety reports.

This is a read view, not a new store or a new enforcement point — every
field is either sourced from an existing table or explicitly marked as not
tracked. See the ProofLayer roadmap discussion: build the passport as a
composite read view first, and only add enforcement later if agents need
to check each other's passports before acting.

Matching pl_agents <-> SDAIA agents: the two tables are keyed independently
(pl_agents.agent_name vs sdaia agents.agent_id) and the pl_agent_id link
column exists in schema but is not populated by current registration code,
so this module matches on literal name equality (e.g. "triage-agent-en" in
both tables) -- true for every agent registered by this codebase today.
"""
from __future__ import annotations

import json
import os
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException

from app.sdaia_api import compute_deployment_decision

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost/triage_agent",
)

router = APIRouter(prefix="/api/v1/passport", tags=["Agent Passport"])

# Best-effort key-substring match for surfacing numeric policy limits
# (refund caps, transfer caps, thresholds, ...) out of an agent's freeform
# _POLICIES dict as a "Spending Limits" quick-view, separate from the full
# contract shown under "Contracts". Not a guarantee of completeness --
# a policy key that doesn't match one of these substrings still appears
# in Contracts, just not pulled out here.
_LIMIT_KEY_HINTS = ("max", "min", "cap", "limit", "threshold", "rate", "sar", "above", "below")


def _get_conn():
    return psycopg2.connect(DATABASE_URL, connect_timeout=3)


def _cursor(conn):
    return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)


def _parse_meta(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return {}
    return {}


def _extract_limits(policies: dict) -> list[dict]:
    limits = []
    for key, value in policies.items():
        if isinstance(value, (int, float)) and any(hint in key.lower() for hint in _LIMIT_KEY_HINTS):
            limits.append({"key": key, "value": value})
    return limits


@router.get("/{agent_name}")
def get_agent_passport(agent_name: str) -> dict:
    with _get_conn() as conn:
        with _cursor(conn) as cur:
            cur.execute(
                """
                SELECT a.agent_id, a.agent_name, a.agent_version, a.model_id,
                       a.agent_group, a.data_classification, a.description,
                       a.metadata, a.registered_at, a.last_seen,
                       COUNT(n.node_id)                                AS decision_count,
                       AVG((n.properties->>'confidence_score')::float) AS avg_confidence
                FROM pl_agents a
                LEFT JOIN pl_nodes n
                       ON n.agent_name = a.agent_name AND n.node_type = 'Decision'
                WHERE a.agent_name = %s
                GROUP BY a.agent_id, a.agent_name, a.agent_version, a.model_id,
                         a.agent_group, a.data_classification, a.description,
                         a.metadata, a.registered_at, a.last_seen
                """,
                (agent_name,),
            )
            agent = cur.fetchone()
            if not agent:
                raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not registered")

            metadata = _parse_meta(agent["metadata"])
            policies = metadata.get("policies", {}) or {}

            cur.execute("SELECT COUNT(*) AS n FROM pl_exceptions e "
                        "JOIN pl_nodes n ON n.node_id = e.decision_node_id "
                        "WHERE n.agent_name = %s", (agent_name,))
            exceptions_count = cur.fetchone()["n"]

            cur.execute(
                """
                SELECT p.policy_code, p.name, p.version
                FROM pl_policy_agent_map m
                JOIN pl_policies p ON p.policy_id = m.policy_id
                WHERE m.agent_name = %s AND (m.applied_to IS NULL OR m.applied_to > NOW())
                ORDER BY m.applied_from DESC
                """,
                (agent_name,),
            )
            active_policies = cur.fetchall()

            # SDAIA compliance record, matched by literal agent-id/agent-name
            # equality (see module docstring).
            cur.execute("SELECT * FROM agents WHERE agent_id = %s", (agent_name,))
            sdaia_agent = cur.fetchone()

            risk = None
            ethics = None
            incidents_summary = None
            safety_report_count = 0
            latest_safety_report_at = None
            gate = None

            if sdaia_agent:
                cur.execute(
                    """
                    SELECT category, level, rationale
                    FROM risk_assessments
                    WHERE agent_id = %s AND assessed_at = (
                        SELECT MAX(assessed_at) FROM risk_assessments WHERE agent_id = %s
                    )
                    """,
                    (agent_name, agent_name),
                )
                risk_rows = cur.fetchall()
                if risk_rows:
                    rank = {"MINIMAL": 0, "LIMITED": 1, "HIGH": 2, "CRITICAL": 3}
                    overall = max((r["level"] for r in risk_rows), key=lambda lvl: rank[lvl])
                    risk = {
                        "overall": overall,
                        "categories": [dict(r) for r in risk_rows],
                    }

                cur.execute(
                    "SELECT tier, tier_name_ar, tier_name_en, score, computed_at "
                    "FROM ethics_labels WHERE agent_id = %s "
                    "ORDER BY computed_at DESC LIMIT 1",
                    (agent_name,),
                )
                ethics_row = cur.fetchone()
                if ethics_row:
                    ethics = dict(ethics_row)
                    ethics["computed_at"] = ethics["computed_at"].isoformat()

                cur.execute(
                    "SELECT COUNT(*) AS total, "
                    "COUNT(*) FILTER (WHERE resolved_at IS NOT NULL) AS resolved, "
                    "COUNT(*) FILTER (WHERE severity = 'CRITICAL' AND reported_to_regulator) AS critical_reported "
                    "FROM incidents WHERE agent_id = %s",
                    (agent_name,),
                )
                inc = cur.fetchone()
                incidents_summary = {
                    "total": inc["total"],
                    "resolved": inc["resolved"],
                    "open": inc["total"] - inc["resolved"],
                    "critical_reported_to_regulator": inc["critical_reported"],
                }

                cur.execute(
                    "SELECT COUNT(*) AS n, MAX(generated_at) AS latest "
                    "FROM safety_reports WHERE agent_id = %s",
                    (agent_name,),
                )
                sr = cur.fetchone()
                safety_report_count = sr["n"]
                latest_safety_report_at = sr["latest"].isoformat() if sr["latest"] else None

                # Same gate app/sdaia_api.py's /deployment-check route calls --
                # not a re-derivation, so this can never disagree with it.
                gate = compute_deployment_decision(cur, agent_name, sdaia_agent["name"])

    _GATE_TO_STATUS = {"prohibited": "PROHIBITED", "blocked": "BLOCKED", "warning": "WARNING", "approved": "ACTIVE"}
    if not sdaia_agent:
        revocation = {"status": "UNVERIFIED", "reason": "No SDAIA compliance record for this agent -- risk and label unknown"}
    else:
        revocation = {"status": _GATE_TO_STATUS[gate["decision"]], "reason": gate["reason"]}

    return {
        "identity": {
            "agent_id": str(agent["agent_id"]),
            "agent_name": agent["agent_name"],
            "agent_version": agent["agent_version"],
            "model_id": agent["model_id"],
            "description": agent["description"] or "",
            "registered_at": agent["registered_at"].isoformat(),
            "last_seen": agent["last_seen"].isoformat(),
        },
        "owner": {"tracked": False, "value": None, "note": "No owner/responsible-human field exists in the current schema"},
        "organization": agent["agent_group"],
        "capabilities": metadata.get("intents", []),
        "delegated_authority": {
            "autonomy_level": sdaia_agent["autonomy_level"] if sdaia_agent else None,
            "contains_pii": metadata.get("contains_pii", False),
            "data_classification": agent["data_classification"],
        },
        "spending_limits": {
            "extracted": _extract_limits(policies),
            "note": "Best-effort key-match over the behavioral contract below -- see Contracts for the full policy set",
        },
        "reputation": ethics or {"tracked": False, "note": "No ethics-label assessment on file for this agent"},
        "verified_history": {
            "decision_count": int(agent["decision_count"] or 0),
            "avg_confidence": round(float(agent["avg_confidence"]), 3) if agent["avg_confidence"] else None,
            "incidents": incidents_summary,
        },
        "contracts": {
            "policies": policies,
            "active_pl_policies": [dict(p) for p in active_policies],
        },
        "evidence": {
            "safety_report_count": safety_report_count,
            "latest_safety_report_at": latest_safety_report_at,
            "exceptions_count": exceptions_count,
        },
        "risk_score": risk or {"tracked": False, "note": "No SDAIA risk classification on file for this agent"},
        "revocation_status": revocation,
    }
