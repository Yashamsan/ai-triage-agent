"""DB connection — same PostgreSQL, reusable for Arabic data."""

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


def apply_schema() -> None:
    sql = (Path(__file__).parent / "schema.sql").read_text(encoding="utf-8")
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.commit()


def find_faq(intent: str, embedding: list[float]) -> dict | None:
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


# -- Retrieval ---------------------------------------------------------
#
# Formerly a keyword-overlap re-ranker sat on top of vector similarity here,
# compensating for all-MiniLM-L6-v2 not being a true multilingual model (its
# Arabic similarity ranking could be genuinely wrong, not just low-scoring).
# That heuristic measured 38% against scripts/eval_kb_retrieval.py's golden
# set -- the best of several options tried against that bad embedding
# signal (a hybrid vector+full-text design scored worse, 24-29%).
#
# Replacing the embedding model with BAAI/bge-m3 (real multilingual/Arabic
# training, see shared/embeddings.py) and re-measuring showed the
# keyword-overlap re-ranker now actively HURTS: with a trustworthy vector
# signal, pure top-1 similarity already gets the right answer in cases
# where the heuristic overrides it with a worse pick purely because of one
# generic shared word. Pure vector + threshold measured 90% (19/21) on the
# same golden set, vs. 81% keeping the heuristic layered on top -- so it's
# removed rather than kept "just in case".
#
# min_similarity=0.45 is the middle of a wide, flat plateau (0.42-0.50 all
# scored identically): real matches scored 0.55-0.79, irrelevant queries
# topped out around 0.39-0.40. Re-run scripts/eval_kb_retrieval.py before
# changing this if the KB content or embedding model changes again.


def find_kb_chunk(
    embedding: list[float], query_text: str = "", lang: str = "ar", min_similarity: float = 0.45,
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


def create_ticket(user_message: str, intent: str, embedding: list[float]) -> int:
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
