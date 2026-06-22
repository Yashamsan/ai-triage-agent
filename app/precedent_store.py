"""Precedent store: persistent long-term memory for triage decisions."""

import hashlib
import json
import os

import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

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


def compute_pattern_hash(
    symptoms: str, vitals: str = "", past_history: str = ""
) -> str:
    raw = f"{symptoms}|{vitals}|{past_history}"
    return hashlib.sha256(raw.lower().strip().encode()).hexdigest()


def store_trace(trace: dict, human_correction: str | None = None):
    conn = _get_conn()
    register_vector(conn)
    cur = conn.cursor()

    query_text = trace.get("query", "")
    text_for_embedding = query_text.strip() or trace.get("symptoms", "")
    embedding = _get_model().encode(text_for_embedding)

    pattern_hash = compute_pattern_hash(query_text)

    cur.execute("""
        INSERT INTO precedents
            (pattern_hash, intent, severity_decision,
             human_correction, correction_reason, full_trace, embedding)
        VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
    """, (
        pattern_hash,
        trace.get("intent") or trace.get("classification"),
        trace.get("severity_decision") or trace.get("classification"),
        human_correction or trace.get("human_correction"),
        trace.get("correction_reason"),
        json.dumps(trace),
        embedding.tolist(),
    ))
    conn.commit()
    cur.close()
    conn.close()


def find_precedent(
    symptoms: str = "",
    vitals: str = "",
    past_history: str = "",
    top_k: int = 1,
) -> list[dict]:
    conn = _get_conn()
    register_vector(conn)
    cur = conn.cursor()

    text = f"{symptoms} {vitals} {past_history}".strip()
    embedding = _get_model().encode(text)

    cur.execute("""
        SELECT pattern_hash, intent, severity_decision,
               human_correction, correction_reason,
               confidence, applied_count, full_trace,
               1 - (embedding <=> %s::vector) AS similarity
        FROM precedents
        WHERE 1 - (embedding <=> %s::vector) > 0.7
        ORDER BY similarity DESC
        LIMIT %s
    """, (embedding.tolist(), embedding.tolist(), top_k))

    rows = cur.fetchall()
    cur.close()
    conn.close()

    results = []
    for row in rows:
        results.append({
            "pattern_hash": row[0],
            "intent": row[1],
            "decision": row[2],
            "human_correction": row[3],
            "reason": row[4],
            "confidence": row[5],
            "applied_count": row[6],
            "full_trace": row[7],
            "match_type": "semantic",
            "similarity": row[8],
        })
    return results
