"""DB connection + query functions.

Connects to PostgreSQL via DATABASE_URL env var.
All functions use a short-lived connection per call (no pool needed at
this scale; swap in psycopg2.pool or asyncpg when load demands it).
"""

import json
import os
from contextlib import contextmanager
from pathlib import Path

import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres@localhost/triage_agent",
)


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL, connect_timeout=3)
    try:
        yield conn
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────

def apply_schema() -> None:
    """Run app/schema.sql against the database (idempotent — IF NOT EXISTS)."""
    sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def apply_schema_v2() -> None:
    """Run app/schema_v2.sql — pl_agents + full-text search trigger.

    Safe to call repeatedly; all statements use IF NOT EXISTS / OR REPLACE.
    No-op if schema_v2.sql is not present.
    """
    sql_path = Path(__file__).parent / "schema_v2.sql"
    if not sql_path.exists():
        return
    sql = sql_path.read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def apply_schema_v3() -> None:
    """Run app/schema_v3.sql — ProofLayer v3 tables and governance indexes.

    Adds pl_trace_steps, pl_exceptions, pl_decision_contexts, pl_policies,
    pl_policy_agent_map, pl_data_classifications, and governance indexes.
    No-op if schema_v3.sql is not present.
    """
    sql_path = Path(__file__).parent / "schema_v3.sql"
    if not sql_path.exists():
        return
    sql = sql_path.read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def apply_schema_rmf() -> None:
    """Run app/schema_rmf.sql — SDAIA-P145 Risk Management Framework tables.

    Adds rmf_contexts, rmf_risks, rmf_assessments, rmf_treatments, rmf_reviews.
    No FK into pl_agents or the SDAIA Responsible AI Policy agents table —
    standalone module, zero impact on that module's data.
    No-op if schema_rmf.sql is not present.
    """
    sql_path = Path(__file__).parent / "schema_rmf.sql"
    if not sql_path.exists():
        return
    sql = sql_path.read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


# ── FAQ queries ───────────────────────────────────────────────────────

def find_faq(intent: str, embedding: list[float]) -> dict | None:
    """Return the closest FAQ article for the given intent + embedding."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT title, content,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM   faq_articles
                WHERE  intent = %s
                ORDER  BY embedding <=> %s::vector
                LIMIT  1
                """,
                (str(embedding), intent, str(embedding)),
            )
            return cur.fetchone()


# ── Knowledge base (product_inquiry) ─────────────────────────────────
#
# Keyword-overlap boost: pure cosine similarity can rank an unrelated entry
# above a near-exact term match (confirmed on the Arabic side against this
# same KB — see app_ar/database.py for the full story). Re-ranking a wider
# candidate pool by literal keyword overlap catches those cases without
# needing a different embedding model.

# -- Retrieval ---------------------------------------------------------
#
# Formerly a keyword-overlap re-ranker sat on top of vector similarity here
# (same pattern as app_ar/database.py, which had a measured golden-set
# eval), compensating for all-MiniLM-L6-v2's weaker similarity signal.
# Removed for the same reason after switching to BAAI/bge-m3 (see
# shared/embeddings.py): a keyword-overlap crutch tuned for a weak
# embedding model actively overrides a trustworthy one's correct top-1
# pick when a single generic word happens to overlap.
#
# min_similarity=0.45 carries over the value measured against the Arabic
# KB (scripts/eval_kb_retrieval.py, 90% on its golden set) -- same
# embedding model and similarly-structured KB content, but NOT
# independently measured for English; there is no English golden set yet.
# Build one (mirroring scripts/eval_kb_retrieval.py) before relying on this
# threshold for anything high-stakes on the English side.


def find_kb_chunk(
    embedding: list[float], query_text: str = "", lang: str = "en", min_similarity: float = 0.45,
) -> dict | None:
    """Return the best knowledge-base chunk in the given language, or None if
    nothing qualifies (not filtered by intent, so a weak match is more likely).

    query_text is unused now (kept for call-site compatibility) -- see module
    notes above; pure vector similarity is the whole story since the bge-m3
    embedding swap."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT title, content, source_file, row_ref,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM   knowledge_base_chunks
                WHERE  lang = %s
                ORDER  BY embedding <=> %s::vector
                LIMIT  1
                """,
                (str(embedding), lang, str(embedding)),
            )
            best = cur.fetchone()

    if best and best["similarity"] >= min_similarity:
        return best
    return None


def delete_kb_chunks_for_file(source_file: str) -> int:
    """Delete all chunks previously ingested from source_file, so re-running
    the ingestion script on an updated spreadsheet doesn't pile up duplicates."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM knowledge_base_chunks WHERE source_file = %s", (source_file,))
            deleted = cur.rowcount
        conn.commit()
    return deleted


def insert_kb_chunk(
    source_file: str, row_ref: str, title: str, content: str,
    embedding: list[float], lang: str = "en", metadata: dict | None = None,
) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO knowledge_base_chunks
                    (source_file, row_ref, title, content, lang, metadata, embedding)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::vector)
                """,
                (source_file, row_ref, title, content, lang, json.dumps(metadata or {}), str(embedding)),
            )
        conn.commit()


def insert_faq(intent: str, title: str, content: str, embedding: list[float]) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO faq_articles (intent, title, content, embedding)
                VALUES (%s, %s, %s, %s::vector)
                """,
                (intent, title, content, str(embedding)),
            )
        conn.commit()


# ── Ticket queries ────────────────────────────────────────────────────

def create_ticket(user_message: str, intent: str, embedding: list[float]) -> int:
    """Insert a support ticket and return its id."""
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO support_tickets (user_message, intent, embedding)
                VALUES (%s, %s, %s::vector)
                RETURNING id
                """,
                (user_message, intent, str(embedding)),
            )
            ticket_id = cur.fetchone()[0]
        conn.commit()
    return ticket_id


# ── Conversation history ──────────────────────────────────────────────

def save_message(session_id: str, role: str, message: str, intent: str | None = None) -> None:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversation_history (session_id, role, message, intent)
                VALUES (%s, %s, %s, %s)
                """,
                (session_id, role, message, intent),
            )
        conn.commit()


def get_history(session_id: str, limit: int = 10) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT role, message, intent, created_at
                FROM   conversation_history
                WHERE  session_id = %s
                ORDER  BY created_at DESC
                LIMIT  %s
                """,
                (session_id, limit),
            )
            return list(reversed(cur.fetchall()))
