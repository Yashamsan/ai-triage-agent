"""SQLite storage for the SDAIA-P145 RMF module.

Zero external dependencies (stdlib sqlite3). Per-call connections with
check_same_thread=False safe for FastAPI's thread pool. Set RMF_DB_PATH to
override the database location.
"""
import os
import sqlite3
from pathlib import Path
from uuid import uuid4

DB_PATH = os.environ.get(
    "RMF_DB_PATH",
    str(Path(__file__).resolve().parent.parent / "rmf_data.db"),
)
SCHEMA_PATH = Path(__file__).resolve().parent / "schema_rmf.sql"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    conn = _connect()
    try:
        conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        conn.commit()
    finally:
        conn.close()


# --- contexts (Stage 1) -----------------------------------------------------

def upsert_context(agent_id: str, data: dict) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO rmf_contexts
                (context_id, agent_id, description, data_io, automation_level,
                 human_role, lifecycle_stage, change_plan)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                description=excluded.description,
                data_io=excluded.data_io,
                automation_level=excluded.automation_level,
                human_role=excluded.human_role,
                lifecycle_stage=excluded.lifecycle_stage,
                change_plan=excluded.change_plan
            """,
            (
                str(uuid4()), agent_id,
                data.get("description", ""),
                data.get("data_io", ""),
                data.get("automation_level", ""),
                data.get("human_role", ""),
                data.get("lifecycle_stage", ""),
                data.get("change_plan", ""),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_context(agent_id: str):
    conn = _connect()
    try:
        return conn.execute(
            "SELECT * FROM rmf_contexts WHERE agent_id = ?", (agent_id,)
        ).fetchone()
    finally:
        conn.close()


# --- risks (Stage 2) ---------------------------------------------------------

def insert_risk(risk_id, agent_id, category, description, source, intent, timing, identified_by) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO rmf_risks
                (risk_id, agent_id, category, description, source, intent,
                 timing, identified_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (risk_id, agent_id, category, description, source, intent, timing, identified_by),
        )
        conn.commit()
    finally:
        conn.close()


def get_risk(risk_id: str):
    conn = _connect()
    try:
        return conn.execute(
            "SELECT * FROM rmf_risks WHERE risk_id = ?", (risk_id,)
        ).fetchone()
    finally:
        conn.close()


def list_risks(agent_id: str | None = None, category: str | None = None):
    conn = _connect()
    try:
        q = "SELECT * FROM rmf_risks"
        clauses, params = [], []
        if agent_id:
            clauses.append("agent_id = ?")
            params.append(agent_id)
        if category:
            clauses.append("category = ?")
            params.append(category)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY created_at DESC, rowid DESC"  # rowid breaks same-second ties
        return conn.execute(q, params).fetchall()
    finally:
        conn.close()


def update_risk_status(risk_id: str, status: str) -> None:
    conn = _connect()
    try:
        conn.execute("UPDATE rmf_risks SET status = ? WHERE risk_id = ?", (status, risk_id))
        conn.commit()
    finally:
        conn.close()


# --- assessments (Stage 3) ---------------------------------------------------

def insert_assessment(assessment_id, risk_id, likelihood, impact, risk_level, band, evidence, assessed_by) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO rmf_assessments
                (assessment_id, risk_id, likelihood, impact, risk_level, band,
                 evidence, assessed_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (assessment_id, risk_id, likelihood, impact, risk_level, band, evidence, assessed_by),
        )
        conn.commit()
    finally:
        conn.close()


def latest_assessment(risk_id: str):
    conn = _connect()
    try:
        return conn.execute(
            """
            SELECT * FROM rmf_assessments WHERE risk_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT 1
            """,
            (risk_id,),
        ).fetchone()
    finally:
        conn.close()


# --- treatments (Stage 4) ----------------------------------------------------

def insert_treatment(treatment_id, risk_id, strategy, rationale, controls,
                     residual_likelihood, residual_impact, residual_level,
                     approver, approval_mechanism) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO rmf_treatments
                (treatment_id, risk_id, strategy, rationale, controls,
                 residual_likelihood, residual_impact, residual_level,
                 approver, approval_mechanism)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (treatment_id, risk_id, strategy, rationale, controls,
             residual_likelihood, residual_impact, residual_level,
             approver, approval_mechanism),
        )
        conn.commit()
    finally:
        conn.close()


def latest_treatment(risk_id: str):
    conn = _connect()
    try:
        return conn.execute(
            """
            SELECT * FROM rmf_treatments WHERE risk_id = ?
            ORDER BY created_at DESC, rowid DESC LIMIT 1
            """,
            (risk_id,),
        ).fetchone()
    finally:
        conn.close()


# --- reviews (Stage 5) -------------------------------------------------------

def insert_review(review_id, agent_id, review_type, findings, incident_rca, reviewed_by) -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO rmf_reviews
                (review_id, agent_id, review_type, findings, incident_rca, reviewed_by)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (review_id, agent_id, review_type, findings, incident_rca, reviewed_by),
        )
        conn.commit()
    finally:
        conn.close()


def list_reviews(agent_id: str | None = None):
    conn = _connect()
    try:
        if agent_id:
            return conn.execute(
                "SELECT * FROM rmf_reviews WHERE agent_id = ? ORDER BY created_at DESC",
                (agent_id,),
            ).fetchall()
        return conn.execute(
            "SELECT * FROM rmf_reviews ORDER BY created_at DESC"
        ).fetchall()
    finally:
        conn.close()


def matrix_distribution():
    """Count of latest assessments per band, across the whole fleet.

    Latest assessment per risk = max rowid for that risk (rowid is
    monotonically increasing; avoids lexicographic bugs on timestamps).
    """
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT a.band, COUNT(*) AS n
            FROM rmf_assessments a
            WHERE a.rowid = (
                SELECT MAX(b.rowid) FROM rmf_assessments b
                WHERE b.risk_id = a.risk_id
            )
            GROUP BY a.band
            """
        ).fetchall()
        return {r["band"]: r["n"] for r in rows}
    finally:
        conn.close()


def register_agent_ids():
    """Distinct agent ids that appear anywhere in the RMF module."""
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT agent_id FROM rmf_risks UNION SELECT DISTINCT agent_id FROM rmf_contexts"
        ).fetchall()
        return [r["agent_id"] for r in rows]
    finally:
        conn.close()
