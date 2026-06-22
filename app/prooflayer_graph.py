"""ProofLayer Context Graph — Python helper for insert + query.

Bridges live triage agent decisions into the pl_nodes/pl_edges graph.
"""

import json
import os
from datetime import UTC, datetime

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


# ── Insert ────────────────────────────────────────────────────────────


def record_decision(
    decision_value: str,
    confidence: float,
    human_override: bool = False,
    reasoning_summary: str = "",
    session_id: str = "",
    input_query: str = "",
    policy_ids: list[str] | None = None,
) -> dict:
    """Record a Decision node + ContextSnapshot + edges to applied policies."""
    conn = _get_conn()
    register_vector(conn)
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # 1. Create ContextSnapshot node
    snapshot_props = {
        "input_query": input_query[:500],
        "session_id": session_id,
        "timestamp": datetime.now(UTC).isoformat(),
    }
    snapshot_embedding = _get_model().encode(input_query) if input_query else None
    cur.execute(
        """
        INSERT INTO pl_nodes (node_type, properties, embedding)
        VALUES ('ContextSnapshot', %s, %s::vector)
        RETURNING node_id
        """,
        (
            json.dumps(snapshot_props),
            snapshot_embedding.tolist() if snapshot_embedding is not None else None,
        ),
    )
    snapshot_id = cur.fetchone()["node_id"]

    # 2. Create Decision node
    decision_props = {
        "decision_value": decision_value,
        "confidence_score": confidence,
        "human_override": human_override,
        "reasoning_summary": reasoning_summary[:500],
        "session_id": session_id,
        "model_id": "deepseek/deepseek-chat",
    }
    cur.execute(
        """
        INSERT INTO pl_nodes (node_type, properties)
        VALUES ('Decision', %s)
        RETURNING node_id
        """,
        (json.dumps(decision_props),),
    )
    decision_id = cur.fetchone()["node_id"]

    # 3. Link: Decision → USED_CONTEXT → ContextSnapshot
    cur.execute(
        """
        INSERT INTO pl_edges (from_node_id, to_node_id, edge_type, properties)
        VALUES (%s, %s, 'USED_CONTEXT', '{}')
        """,
        (decision_id, snapshot_id),
    )

    # 4. Link: Decision → APPLIED_POLICY → each known policy
    policy_nodes = []
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

    conn.commit()
    cur.close()
    conn.close()

    return {
        "decision_id": str(decision_id),
        "snapshot_id": str(snapshot_id),
        "policy_edges": len(policy_nodes),
    }


# ── Query ─────────────────────────────────────────────────────────────


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
    """Summary of all decisions in the time window."""
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    cur.execute(
        """
        SELECT
            COUNT(*) AS total_decisions,
            COUNT(*) FILTER (WHERE (properties->>'human_override')::BOOLEAN = TRUE)
                AS human_overrides,
            COUNT(*) FILTER (WHERE properties->>'decision_value' = 'escalation')
                AS escalations,
            AVG((properties->>'confidence_score')::FLOAT) AS avg_confidence
        FROM pl_nodes
        WHERE node_type = 'Decision'
          AND created_at >= NOW() - INTERVAL '1 day' * %s
        """,
        (since_days,),
    )
    summary = cur.fetchone()
    cur.close()
    conn.close()
    return dict(summary)
