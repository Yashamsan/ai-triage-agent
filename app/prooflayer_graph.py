"""ProofLayer Context Graph v3 -- Python helpers for insert + query.

Bridges live triage agent decisions into the pl_nodes/pl_edges graph.

v3 additions:
  record_exception()         -- Ghost Knowledge: human narrative behind overrides
  add_trace_step()           -- Reasoning Memory: Thought->Action->Observation per node
  record_context_snapshot()  -- Event Clock: system state at decision time
  record_cross_agent_edge()  -- Cross-Agent Governance: link decisions across agents
  query_cross_agent()        -- Governance query: PII decisions across all agents
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from typing import Any

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost/triage_agent",
)

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _get_conn():
    return psycopg2.connect(DATABASE_URL, connect_timeout=3)


_DEFAULT_MODEL_ID = os.getenv("DEFAULT_MODEL_ID", "deepseek/deepseek-chat")


# ============================================================================
# v2: record_decision
# ============================================================================


def record_decision(
    decision_value: str,
    confidence: float,
    human_override: bool = False,
    reasoning_summary: str = "",
    session_id: str = "",
    input_query: str = "",
    policy_ids: list[str] | None = None,
    model_id: str | None = None,
    agent_name: str | None = None,
    contains_pii: bool = False,
    agent_group: str | None = None,
    trace_steps: list[dict] | None = None,
    model_version: str | None = None,
    active_policies: list[str] | None = None,
    risk_scores: dict | None = None,
) -> dict:
    """Record a Decision node + ContextSnapshot + optional v3 trace steps.

    v3 extras accepted here so callers can do everything in one call:
      contains_pii    -- flag decision node for PII governance queries
      agent_group     -- organisational group (e.g. "contact-center")
      trace_steps     -- list of {node_type, thought, action, observation, ...}
      model_version   -- for DecisionContext Event Clock snapshot
      active_policies -- policy codes active at decision time
      risk_scores     -- {key: float} risk signal map
    """
    conn = _get_conn()
    register_vector(conn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # 1. ContextSnapshot node
    snapshot_props = {
        "input_query": input_query[:500],
        "session_id": session_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    snapshot_embedding = _get_model().encode(input_query) if input_query else None
    cur.execute(
        """
        INSERT INTO pl_nodes (node_type, properties, embedding, agent_name,
                              contains_pii, agent_group)
        VALUES ('ContextSnapshot', %s, %s::vector, %s, %s, %s)
        RETURNING node_id
        """,
        (
            json.dumps(snapshot_props),
            snapshot_embedding.tolist() if snapshot_embedding is not None else None,
            agent_name,
            contains_pii,
            agent_group,
        ),
    )
    snapshot_id = cur.fetchone()["node_id"]

    # 2. Decision node
    decision_props = {
        "decision_value": decision_value,
        "confidence_score": confidence,
        "human_override": human_override,
        "reasoning_summary": reasoning_summary[:500],
        "session_id": session_id,
        "model_id": model_id or _DEFAULT_MODEL_ID,
    }
    cur.execute(
        """
        INSERT INTO pl_nodes (node_type, properties, agent_name,
                              contains_pii, agent_group)
        VALUES ('Decision', %s, %s, %s, %s)
        RETURNING node_id
        """,
        (json.dumps(decision_props), agent_name, contains_pii, agent_group),
    )
    decision_id = cur.fetchone()["node_id"]

    # 3. Decision -[USED_CONTEXT]-> ContextSnapshot
    cur.execute(
        """
        INSERT INTO pl_edges (from_node_id, to_node_id, edge_type, properties)
        VALUES (%s, %s, 'USED_CONTEXT', '{}')
        """,
        (decision_id, snapshot_id),
    )

    # 4. Policy edges
    policy_nodes: list[int] = []
    if policy_ids:
        for pid in policy_ids:
            cur.execute(
                """
                SELECT node_id FROM pl_nodes
                WHERE node_type = 'Policy'
                  AND properties->>'policy_id' = %s
                  AND valid_to IS NULL
                """,
                (pid,),
            )
            row = cur.fetchone()
            if row:
                policy_nodes.append(row["node_id"])
                cur.execute(
                    """
                    INSERT INTO pl_edges (from_node_id, to_node_id, edge_type, properties)
                    VALUES (%s, %s, 'APPLIED_POLICY', '{"application_type": "followed"}')
                    """,
                    (decision_id, row["node_id"]),
                )

    # 5. v3: trace steps
    step_ids: list[int] = []
    if trace_steps:
        for order, step in enumerate(trace_steps):
            cur.execute(
                """
                INSERT INTO pl_trace_steps
                    (decision_node_id, step_order, node_type, thought, action,
                     observation, confidence, latency_ms, extra)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING step_id
                """,
                (
                    decision_id,
                    order,
                    step.get("node_type", ""),
                    step.get("thought"),
                    step.get("action"),
                    step.get("observation"),
                    step.get("confidence"),
                    step.get("latency_ms"),
                    json.dumps(step.get("extra") or {}),
                ),
            )
            step_ids.append(cur.fetchone()["step_id"])

    # 6. v3: DecisionContext (Event Clock)
    if model_version or active_policies or risk_scores:
        cur.execute(
            """
            INSERT INTO pl_decision_contexts
                (decision_node_id, model_version, active_policies, risk_scores)
            VALUES (%s, %s, %s, %s)
            """,
            (
                decision_id,
                model_version,
                json.dumps(active_policies or []),
                json.dumps(risk_scores or {}),
            ),
        )

    conn.commit()
    cur.close()
    conn.close()

    return {
        "decision_id": str(decision_id),
        "snapshot_id": str(snapshot_id),
        "policy_edges": len(policy_nodes),
        "trace_steps": len(step_ids),
    }


# ============================================================================
# v3: record_exception  -- Ghost Knowledge
# ============================================================================


def record_exception(
    decision_node_id: str | int,
    human_narrative: str,
    approver: str | None = None,
    approval_channel: str | None = None,
    policy_violated: str | None = None,
    justification: str | None = None,
    severity: str = "low",
) -> dict:
    """Attach a human exception narrative to an existing Decision node.

    Creates a pl_exceptions row AND an Exception pl_node linked via
    HAS_EXCEPTION edge so the graph remains queryable.
    """
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # Insert into pl_exceptions table
    cur.execute(
        """
        INSERT INTO pl_exceptions
            (decision_node_id, human_narrative, approver, approval_channel,
             policy_violated, justification, severity)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING exception_id, created_at
        """,
        (
            int(decision_node_id),
            human_narrative,
            approver,
            approval_channel,
            policy_violated,
            justification,
            severity,
        ),
    )
    row = cur.fetchone()
    exception_id = row["exception_id"]
    created_at = row["created_at"]

    # Also create an Exception pl_node for graph traversal
    exc_props = {
        "human_narrative": human_narrative[:1000],
        "approver": approver,
        "approval_channel": approval_channel,
        "policy_violated": policy_violated,
        "justification": (justification or "")[:500],
        "severity": severity,
        "pl_exception_id": exception_id,
    }
    cur.execute(
        """
        INSERT INTO pl_nodes (node_type, properties)
        VALUES ('Exception', %s)
        RETURNING node_id
        """,
        (json.dumps(exc_props),),
    )
    exc_node_id = cur.fetchone()["node_id"]

    # Decision -[HAS_EXCEPTION]-> Exception node
    cur.execute(
        """
        INSERT INTO pl_edges (from_node_id, to_node_id, edge_type, properties)
        VALUES (%s, %s, 'HAS_EXCEPTION',
                %s)
        """,
        (
            int(decision_node_id),
            exc_node_id,
            json.dumps({"severity": severity, "approver": approver}),
        ),
    )

    conn.commit()
    cur.close()
    conn.close()

    return {
        "exception_id": exception_id,
        "exc_node_id": str(exc_node_id),
        "created_at": created_at.isoformat(),
    }


# ============================================================================
# v3: add_trace_step  -- Reasoning Memory
# ============================================================================


def add_trace_step(
    decision_node_id: str | int,
    node_type: str,
    thought: str | None = None,
    action: str | None = None,
    observation: str | None = None,
    confidence: float | None = None,
    latency_ms: float | None = None,
    step_order: int | None = None,
    extra: dict | None = None,
) -> dict:
    """Append a Thought->Action->Observation step to an existing decision."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    if step_order is None:
        cur.execute(
            "SELECT COALESCE(MAX(step_order), -1) + 1 AS next_order "
            "FROM pl_trace_steps WHERE decision_node_id = %s",
            (int(decision_node_id),),
        )
        step_order = cur.fetchone()["next_order"]

    cur.execute(
        """
        INSERT INTO pl_trace_steps
            (decision_node_id, step_order, node_type, thought, action,
             observation, confidence, latency_ms, extra)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING step_id, created_at
        """,
        (
            int(decision_node_id),
            step_order,
            node_type,
            thought,
            action,
            observation,
            confidence,
            latency_ms,
            json.dumps(extra or {}),
        ),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return {
        "step_id": row["step_id"],
        "step_order": step_order,
        "created_at": row["created_at"].isoformat(),
    }


def get_trace(decision_node_id: str | int) -> list[dict]:
    """Return all trace steps for a decision, ordered by step_order."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT step_id, step_order, node_type, thought, action, observation,
               confidence, latency_ms, extra, created_at
        FROM pl_trace_steps
        WHERE decision_node_id = %s
        ORDER BY step_order
        """,
        (int(decision_node_id),),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [
        {
            "step_id": r["step_id"],
            "step_order": r["step_order"],
            "node_type": r["node_type"],
            "thought": r["thought"],
            "action": r["action"],
            "observation": r["observation"],
            "confidence": r["confidence"],
            "latency_ms": r["latency_ms"],
            "extra": r["extra"] if isinstance(r["extra"], dict) else json.loads(r["extra"] or "{}"),
            "created_at": r["created_at"].isoformat(),
        }
        for r in rows
    ]


# ============================================================================
# v3: record_context_snapshot  -- Event Clock
# ============================================================================


def record_context_snapshot(
    decision_node_id: str | int,
    model_version: str | None = None,
    active_policies: list[str] | None = None,
    risk_scores: dict | None = None,
    feature_flags: dict | None = None,
    system_load: dict | None = None,
) -> dict:
    """Record the system state (Event Clock) at the moment a decision was made."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        INSERT INTO pl_decision_contexts
            (decision_node_id, model_version, active_policies,
             risk_scores, feature_flags, system_load)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT DO NOTHING
        RETURNING context_id, captured_at
        """,
        (
            int(decision_node_id),
            model_version,
            json.dumps(active_policies or []),
            json.dumps(risk_scores or {}),
            json.dumps(feature_flags or {}),
            json.dumps(system_load or {}),
        ),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    if row:
        return {"context_id": row["context_id"], "captured_at": row["captured_at"].isoformat()}
    return {}


# ============================================================================
# v3: record_cross_agent_edge  -- Cross-Agent Governance
# ============================================================================


def record_cross_agent_edge(
    from_decision_id: str | int,
    to_decision_id: str | int,
    relationship: str = "CROSS_AGENT_REFERENCE",
    metadata: dict | None = None,
) -> dict:
    """Create a directed edge between two decisions from different agents."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        INSERT INTO pl_edges
            (from_node_id, to_node_id, edge_type, edge_metadata, properties)
        VALUES (%s, %s, %s, %s, '{}')
        RETURNING edge_id, valid_from
        """,
        (
            int(from_decision_id),
            int(to_decision_id),
            relationship,
            json.dumps(metadata or {}),
        ),
    )
    row = cur.fetchone()
    conn.commit()
    cur.close()
    conn.close()
    return {"edge_id": row["edge_id"], "created_at": row["valid_from"].isoformat()}


# ============================================================================
# v3: query_cross_agent  -- Governance Query
# ============================================================================


def query_cross_agent(
    contains_pii: bool | None = None,
    agent_group: str | None = None,
    agent_name: str | None = None,
    since_days: int = 7,
    limit: int = 200,
) -> dict:
    """Cross-agent governance query: decisions with PII flags, grouped by agent.

    Returns a summary suitable for the Governance dashboard tab.
    """
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    clauses = [
        "node_type = 'Decision'",
        "created_at >= NOW() - INTERVAL '1 day' * %(since_days)s",
    ]
    params: dict[str, Any] = {"since_days": since_days, "limit": limit}

    if contains_pii is True:
        clauses.append("contains_pii = TRUE")
    if agent_group:
        clauses.append("agent_group = %(agent_group)s")
        params["agent_group"] = agent_group
    if agent_name:
        clauses.append("agent_name = %(agent_name)s")
        params["agent_name"] = agent_name

    where = " AND ".join(clauses)

    # Per-agent summary
    cur.execute(
        f"""
        SELECT
            agent_name,
            agent_group,
            COUNT(*)                                         AS total_decisions,
            COUNT(*) FILTER (WHERE contains_pii = TRUE)      AS pii_decisions,
            COUNT(*) FILTER (WHERE (properties->>'human_override')::BOOLEAN = TRUE)
                                                             AS human_overrides,
            AVG((properties->>'confidence_score')::FLOAT)   AS avg_confidence,
            MAX(created_at)                                  AS last_decision_at
        FROM pl_nodes
        WHERE {where}
        GROUP BY agent_name, agent_group
        ORDER BY total_decisions DESC
        LIMIT %(limit)s
        """,
        params,
    )
    agent_rows = cur.fetchall()

    # Recent decisions list (for table view)
    cur.execute(
        f"""
        SELECT node_id, agent_name, agent_group, contains_pii,
               properties, created_at
        FROM pl_nodes
        WHERE {where}
        ORDER BY created_at DESC
        LIMIT %(limit)s
        """,
        params,
    )
    decision_rows = cur.fetchall()

    # Aggregate totals
    cur.execute(
        f"""
        SELECT
            COUNT(*)                                        AS total,
            COUNT(*) FILTER (WHERE contains_pii = TRUE)     AS pii_count,
            COUNT(DISTINCT agent_name)                      AS agent_count,
            COUNT(DISTINCT agent_group)
                FILTER (WHERE agent_group IS NOT NULL)      AS group_count
        FROM pl_nodes
        WHERE {where}
        """,
        params,
    )
    totals = cur.fetchone()

    cur.close()
    conn.close()

    decisions = []
    for r in decision_rows:
        p = r["properties"] if isinstance(r["properties"], dict) else json.loads(r["properties"])
        decisions.append({
            "decision_id": str(r["node_id"]),
            "agent_name": r["agent_name"],
            "agent_group": r["agent_group"],
            "contains_pii": r["contains_pii"],
            "decision_value": p.get("decision_value", ""),
            "confidence": float(p.get("confidence_score", 0)),
            "human_override": bool(p.get("human_override", False)),
            "timestamp": r["created_at"].isoformat(),
        })

    return {
        "since_days": since_days,
        "filters": {
            "contains_pii": contains_pii,
            "agent_group": agent_group,
            "agent_name": agent_name,
        },
        "totals": {
            "total_decisions": int(totals["total"] or 0),
            "pii_decisions": int(totals["pii_count"] or 0),
            "agent_count": int(totals["agent_count"] or 0),
            "group_count": int(totals["group_count"] or 0),
        },
        "by_agent": [
            {
                "agent_name": r["agent_name"],
                "agent_group": r["agent_group"],
                "total_decisions": int(r["total_decisions"]),
                "pii_decisions": int(r["pii_decisions"]),
                "human_overrides": int(r["human_overrides"]),
                "avg_confidence": round(float(r["avg_confidence"] or 0), 3),
                "last_decision_at": r["last_decision_at"].isoformat() if r["last_decision_at"] else None,
            }
            for r in agent_rows
        ],
        "decisions": decisions,
    }


# ============================================================================
# v2: unchanged query helpers
# ============================================================================


def get_decision_chain(decision_id: str, max_depth: int = 5) -> list[dict]:
    """Walk backward through causal chain (recursive CTE)."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        WITH RECURSIVE chain AS (
            SELECT n.*, 0 AS depth FROM pl_nodes n
            WHERE n.node_id = %s
            UNION ALL
            SELECT pn.*, chain.depth + 1 FROM chain
            JOIN pl_edges e ON e.to_node_id = chain.node_id
                AND e.edge_type = 'USED_CONTEXT'
            JOIN pl_nodes pn ON pn.node_id = e.from_node_id
            WHERE chain.depth < %s
        )
        SELECT * FROM chain ORDER BY depth
        """,
        (decision_id, max_depth),
    )
    results = cur.fetchall()
    cur.close()
    conn.close()
    return [dict(r) for r in results]


def get_audit_report(since_days: int = 90) -> dict:
    """Summary of all decisions in the time window, including v3 KPIs."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT
            COUNT(*)                                            AS total_decisions,
            COUNT(*) FILTER (WHERE (properties->>'human_override')::BOOLEAN = TRUE)
                                                                AS human_overrides,
            COUNT(*) FILTER (WHERE properties->>'decision_value' = 'escalation')
                                                                AS escalations,
            AVG((properties->>'confidence_score')::FLOAT)      AS avg_confidence,
            COUNT(*) FILTER (WHERE contains_pii = TRUE)         AS pii_decisions,
            COUNT(DISTINCT agent_group)
                FILTER (WHERE agent_group IS NOT NULL)          AS agent_group_count
        FROM pl_nodes
        WHERE node_type = 'Decision'
          AND created_at >= NOW() - INTERVAL '1 day' * %s
        """,
        (since_days,),
    )
    summary = dict(cur.fetchone())

    # Exception count
    cur.execute(
        """
        SELECT COUNT(*) AS exception_count
        FROM pl_exceptions
        WHERE created_at >= NOW() - INTERVAL '1 day' * %s
        """,
        (since_days,),
    )
    exc = cur.fetchone()
    summary["exception_count"] = int(exc["exception_count"] or 0)

    # Trace step count
    cur.execute(
        """
        SELECT COUNT(*) AS trace_step_count
        FROM pl_trace_steps
        WHERE created_at >= NOW() - INTERVAL '1 day' * %s
        """,
        (since_days,),
    )
    tr = cur.fetchone()
    summary["trace_step_count"] = int(tr["trace_step_count"] or 0)

    cur.close()
    conn.close()
    return summary
