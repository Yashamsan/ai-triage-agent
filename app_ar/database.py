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
    conn = psycopg2.connect(DATABASE_URL)
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


def find_similar_tickets(embedding: list[float], limit: int = 3) -> list[dict]:
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT id, user_message, intent, status,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM   support_tickets
                ORDER  BY embedding <=> %s::vector
                LIMIT  %s
                """,
                (str(embedding), str(embedding), limit),
            )
            return cur.fetchall()


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
