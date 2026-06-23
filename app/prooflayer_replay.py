"""ProofLayer Replay Engine — Replay · Diff · Blame.

Three analytical functions that operate on the pl_nodes/pl_edges graph.
No FastAPI dependencies here — called from prooflayer_api.py endpoints.
"""

import json
import os
from typing import Any

import numpy as np
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost/triage_agent",
)

_LOW_CONFIDENCE_THRESHOLD = 0.60


def _get_conn():
    return psycopg2.connect(DATABASE_URL, connect_timeout=3)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a, dtype=float), np.array(b, dtype=float)
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def _chain_integrity() -> dict[str, Any]:
    """Verify the JSONL audit ledger hash chain. Returns status + summary."""
    try:
        from audit.hasher import verify_chain
        from audit.ledger import JSONLLedger

        ledger = JSONLLedger()
        records = ledger.read_all()
        if not records:
            return {"status": "NO_RECORDS", "record_count": 0}
        violations = verify_chain(records)
        return {
            "status": "INTACT" if not violations else "BROKEN",
            "record_count": len(records),
            "broken_at_index": violations[0] if violations else None,
        }
    except Exception as exc:
        return {"status": "UNAVAILABLE", "reason": str(exc)}


# ── Replay ────────────────────────────────────────────────────────────────────


def replay_decision(decision_id: str) -> dict[str, Any]:
    """Reconstruct the full forensic record of a decision.

    Returns input, context snapshot, applied policies, model, output, and
    audit-chain integrity — equivalent to viewing a git commit.
    """
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    # 1. Decision node
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
        return {}

    props: dict = row["properties"] if isinstance(row["properties"], dict) else json.loads(row["properties"])
    created_at = row["created_at"]

    # 2. ContextSnapshot via USED_CONTEXT edge
    cur.execute(
        """
        SELECT n.properties, n.embedding::text AS embedding_raw
        FROM pl_nodes n
        JOIN pl_edges e ON e.to_node_id = n.node_id
        WHERE e.from_node_id = %s
          AND e.edge_type = 'USED_CONTEXT'
          AND n.node_type = 'ContextSnapshot'
        LIMIT 1
        """,
        (decision_id,),
    )
    ctx_row = cur.fetchone()
    context: dict = {}
    if ctx_row:
        context = ctx_row["properties"] if isinstance(ctx_row["properties"], dict) else json.loads(ctx_row["properties"])

    # 3. Policies via APPLIED_POLICY edges
    cur.execute(
        """
        SELECT n.properties, e.properties AS edge_props
        FROM pl_nodes n
        JOIN pl_edges e ON e.to_node_id = n.node_id
        WHERE e.from_node_id = %s
          AND e.edge_type = 'APPLIED_POLICY'
          AND n.node_type = 'Policy'
        """,
        (decision_id,),
    )
    policy_rows = cur.fetchall()
    policies = []
    for pr in policy_rows:
        p = pr["properties"] if isinstance(pr["properties"], dict) else json.loads(pr["properties"])
        ep = pr["edge_props"] if isinstance(pr["edge_props"], dict) else json.loads(pr["edge_props"] or "{}")
        policies.append({
            "policy_id": p.get("policy_id", "unknown"),
            "name": p.get("name", p.get("policy_id", "unknown")),
            "version": p.get("version", ""),
            "application_type": ep.get("application_type", "followed"),
        })

    cur.close()
    conn.close()

    evidence = _chain_integrity()

    return {
        "decision_id": decision_id,
        "timestamp": created_at.isoformat() if hasattr(created_at, "isoformat") else str(created_at),
        "agent_name": row["agent_name"],
        "input": context.get("input_query") or props.get("input_query", ""),
        "context": context,
        "policies": policies,
        "model": props.get("model_id", ""),
        "output": props.get("decision_value", ""),
        "confidence": props.get("confidence_score", 0.0),
        "human_override": props.get("human_override", False),
        "reasoning": props.get("reasoning_summary", ""),
        "session_id": props.get("session_id", ""),
        "evidence": evidence,
    }


# ── Diff ──────────────────────────────────────────────────────────────────────


def _get_decision_full(cur, decision_id: str) -> dict[str, Any] | None:
    """Fetch a decision with its context embedding and applied policies."""
    cur.execute(
        """
        SELECT n.node_id, n.properties, n.created_at, n.agent_name,
               snap.properties AS ctx_props,
               snap.embedding::text AS embedding_raw
        FROM pl_nodes n
        LEFT JOIN pl_edges ec ON ec.from_node_id = n.node_id
            AND ec.edge_type = 'USED_CONTEXT'
        LEFT JOIN pl_nodes snap ON snap.node_id = ec.to_node_id
            AND snap.node_type = 'ContextSnapshot'
        WHERE n.node_id = %s AND n.node_type = 'Decision'
        LIMIT 1
        """,
        (decision_id,),
    )
    row = cur.fetchone()
    if not row:
        return None

    props = row["properties"] if isinstance(row["properties"], dict) else json.loads(row["properties"])
    ctx = row["ctx_props"] if isinstance(row["ctx_props"], dict) else json.loads(row["ctx_props"] or "{}")

    # Parse pgvector text output e.g. "[0.1, 0.2, ...]"
    embedding: list[float] | None = None
    if row["embedding_raw"]:
        try:
            embedding = json.loads(row["embedding_raw"].replace("(", "[").replace(")", "]"))
        except Exception:
            pass

    # Policies
    cur.execute(
        """
        SELECT n.properties->>'policy_id' AS pid
        FROM pl_nodes n
        JOIN pl_edges e ON e.to_node_id = n.node_id
        WHERE e.from_node_id = %s
          AND e.edge_type = 'APPLIED_POLICY'
          AND n.node_type = 'Policy'
        """,
        (decision_id,),
    )
    policy_ids = {r["pid"] for r in cur.fetchall() if r["pid"]}

    return {
        "decision_id": decision_id,
        "decision_value": props.get("decision_value", ""),
        "confidence": props.get("confidence_score", 0.0),
        "model_id": props.get("model_id", ""),
        "human_override": props.get("human_override", False),
        "agent_name": row["agent_name"],
        "created_at": row["created_at"],
        "input": ctx.get("input_query", ""),
        "policy_ids": policy_ids,
        "embedding": embedding,
    }


def diff_decisions(left_id: str, right_id: str) -> dict[str, Any]:
    """Compare two decisions — changed model, policies, and output.

    Equivalent to git diff between two commits.
    """
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    left = _get_decision_full(cur, left_id)
    right = _get_decision_full(cur, right_id)

    cur.close()
    conn.close()

    if not left:
        return {"error": f"Decision {left_id} not found"}
    if not right:
        return {"error": f"Decision {right_id} not found"}

    changes: dict[str, Any] = {}

    # Output change
    if left["decision_value"] != right["decision_value"]:
        changes["output"] = {"left": left["decision_value"], "right": right["decision_value"]}

    # Model change
    if left["model_id"] != right["model_id"]:
        changes["model"] = {"left": left["model_id"], "right": right["model_id"]}

    # Confidence change
    conf_delta = round(right["confidence"] - left["confidence"], 4)
    if abs(conf_delta) > 0.01:
        changes["confidence"] = {
            "left": round(left["confidence"], 4),
            "right": round(right["confidence"], 4),
            "delta": conf_delta,
        }

    # Agent change
    if left["agent_name"] != right["agent_name"]:
        changes["agent"] = {"left": left["agent_name"], "right": right["agent_name"]}

    # Policy changes
    added = sorted(right["policy_ids"] - left["policy_ids"])
    removed = sorted(left["policy_ids"] - right["policy_ids"])
    retained = sorted(left["policy_ids"] & right["policy_ids"])
    if added or removed:
        changes["policies"] = {"added": added, "removed": removed, "retained": retained}

    # Context similarity via embedding cosine distance
    context_sim: float | None = None
    if left["embedding"] and right["embedding"]:
        context_sim = round(_cosine_similarity(left["embedding"], right["embedding"]), 4)
        if context_sim < 0.85:
            changes["context_shift"] = {
                "similarity": context_sim,
                "note": "Contexts are meaningfully different",
            }

    return {
        "left_id": left_id,
        "right_id": right_id,
        "left_timestamp": left["created_at"].isoformat() if hasattr(left["created_at"], "isoformat") else str(left["created_at"]),
        "right_timestamp": right["created_at"].isoformat() if hasattr(right["created_at"], "isoformat") else str(right["created_at"]),
        "context_similarity": context_sim,
        "summary": {
            "output_changed": "output" in changes,
            "model_changed": "model" in changes,
            "policies_changed": "policies" in changes,
            "context_shifted": "context_shift" in changes,
        },
        "changes": changes,
    }


# ── Blame ─────────────────────────────────────────────────────────────────────


def blame_decision(decision_id: str) -> dict[str, Any]:
    """Trace a decision outcome to its root cause.

    Identifies the dominant factor — low confidence, model change, policy
    change, or context shift — by comparing against the closest past decision
    with a different outcome. Equivalent to git blame.
    """
    conn = _get_conn()
    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    target = _get_decision_full(cur, decision_id)
    if not target:
        cur.close()
        conn.close()
        return {"error": f"Decision {decision_id} not found"}

    factors: list[dict[str, Any]] = []

    # Factor 1 — Human override is always the primary cause
    if target["human_override"]:
        factors.append({
            "factor": "human_override",
            "weight": 1.0,
            "explanation": "A human agent manually overrode the AI decision.",
        })

    # Factor 2 — Low confidence
    if target["confidence"] < _LOW_CONFIDENCE_THRESHOLD:
        factors.append({
            "factor": "low_confidence",
            "weight": round(1.0 - target["confidence"], 4),
            "explanation": f"Model confidence was {target['confidence']:.0%} — below the {_LOW_CONFIDENCE_THRESHOLD:.0%} threshold.",
        })

    # Factor 3 — Find closest past decision with a DIFFERENT outcome (vector ANN)
    comparison: dict[str, Any] | None = None
    if target["embedding"]:
        emb_str = json.dumps(target["embedding"])
        cur.execute(
            """
            SELECT n.node_id AS decision_id,
                   n.properties,
                   n.agent_name,
                   snap.embedding <=> %s::vector AS distance
            FROM pl_nodes snap
            JOIN pl_edges e ON e.to_node_id = snap.node_id
                AND e.edge_type = 'USED_CONTEXT'
            JOIN pl_nodes n ON n.node_id = e.from_node_id
                AND n.node_type = 'Decision'
            WHERE snap.node_type = 'ContextSnapshot'
              AND snap.embedding IS NOT NULL
              AND n.node_id != %s
              AND n.properties->>'decision_value' != %s
            ORDER BY snap.embedding <=> %s::vector
            LIMIT 1
            """,
            (emb_str, decision_id, target["decision_value"], emb_str),
        )
        similar_row = cur.fetchone()
        if similar_row and similar_row["distance"] < 0.25:  # cos distance < 0.25 = similar
            sim_props = (
                similar_row["properties"]
                if isinstance(similar_row["properties"], dict)
                else json.loads(similar_row["properties"])
            )
            context_similarity = round(1.0 - float(similar_row["distance"]), 4)
            comparison = {
                "decision_id": str(similar_row["decision_id"]),
                "decision_value": sim_props.get("decision_value", ""),
                "model_id": sim_props.get("model_id", ""),
                "confidence": sim_props.get("confidence_score", 0.0),
                "agent_name": similar_row["agent_name"],
                "context_similarity": context_similarity,
            }

            # Factor 3a — Model changed between these two similar decisions
            if target["model_id"] and comparison["model_id"] and target["model_id"] != comparison["model_id"]:
                factors.append({
                    "factor": "model_change",
                    "weight": 0.9,
                    "explanation": f"Model changed from {comparison['model_id']} to {target['model_id']}.",
                    "from": comparison["model_id"],
                    "to": target["model_id"],
                })

            # Factor 3b — Policy set changed
            sim_id = str(similar_row["decision_id"])
            cur.execute(
                """
                SELECT n.properties->>'policy_id' AS pid
                FROM pl_nodes n
                JOIN pl_edges e ON e.to_node_id = n.node_id
                WHERE e.from_node_id = %s
                  AND e.edge_type = 'APPLIED_POLICY'
                  AND n.node_type = 'Policy'
                """,
                (sim_id,),
            )
            sim_policies = {r["pid"] for r in cur.fetchall() if r["pid"]}
            added = sorted(target["policy_ids"] - sim_policies)
            removed = sorted(sim_policies - target["policy_ids"])
            if added or removed:
                factors.append({
                    "factor": "policy_change",
                    "weight": 0.85,
                    "explanation": "Applied policy set differs from the comparable past decision.",
                    "policies_added": added,
                    "policies_removed": removed,
                })

            # Factor 3c — Subtle context shift despite surface similarity
            if context_similarity > 0.80 and not any(f["factor"] in ("model_change", "policy_change") for f in factors):
                factors.append({
                    "factor": "context_shift",
                    "weight": round(1.0 - context_similarity, 4),
                    "explanation": f"Context is {context_similarity:.0%} similar to a past opposite decision — subtle input differences changed the outcome.",
                })
    else:
        factors.append({
            "factor": "insufficient_precedent",
            "weight": 0.5,
            "explanation": "No embedding available — cannot compare against historical decisions.",
        })

    cur.close()
    conn.close()

    # Sort by weight descending; primary root cause is first
    factors.sort(key=lambda f: f["weight"], reverse=True)
    root_cause = factors[0] if factors else {"factor": "unknown", "weight": 0.0, "explanation": "No causal factors identified."}

    return {
        "decision_id": decision_id,
        "output": target["decision_value"],
        "confidence": target["confidence"],
        "root_cause": root_cause,
        "all_factors": factors,
        "comparable_decision": comparison,
    }
