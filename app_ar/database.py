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
    sql = (Path(__file__).parent / "schema.sql").read_text()
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


def find_kb_chunk(embedding: list[float], lang: str = "ar", min_similarity: float = 0.35) -> dict | None:
    """Return the closest knowledge-base chunk in the given language, or None
    if nothing clears min_similarity (not filtered by intent, so a weak
    match is more likely).

    Threshold is lower than the English default (0.5) because all-MiniLM-L6-v2
    isn't a true multilingual model (see app_ar/embeddings.py) — measured
    against this KB, a genuine Arabic match scores ~0.42-0.43 where the
    equivalent English match scores ~0.61. 0.5 would silently reject nearly
    every real Arabic hit."""
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
            row = cur.fetchone()
            if row and row["similarity"] >= min_similarity:
                return row
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
