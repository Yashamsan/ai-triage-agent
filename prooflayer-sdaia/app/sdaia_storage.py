"""Storage abstraction for the ProofLayer SDAIA compliance module.

SQLite backend for dev/testing (in-process, ":memory:" by default).
PostgreSQL backend for production — pass a "postgresql://..." DSN, or
set the SDAIA_DSN env var, and this reuses the psycopg2-binary
dependency already required by app/database.py elsewhere in this repo.
The Postgres table shape is defined once in app/schema_sdaia.sql and
applied automatically on first connect (CREATE TABLE IF NOT EXISTS —
safe to run repeatedly).

This is a governance *scaffold* — table/field names follow a plausible
compliance-tracking structure, not a verified SDAIA data model.
"""
from __future__ import annotations

import calendar
import json
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS agents (
    agent_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    sector          TEXT NOT NULL,
    description     TEXT,
    handles_pii     INTEGER NOT NULL DEFAULT 0,
    autonomy_level  TEXT NOT NULL DEFAULT 'assisted',
    pl_agent_id     TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS risk_assessments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        TEXT NOT NULL,
    category        TEXT NOT NULL,
    level           TEXT NOT NULL,
    rationale       TEXT NOT NULL,
    factors         TEXT NOT NULL DEFAULT '{}',
    assessed_at     TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id                TEXT NOT NULL,
    decision_type           TEXT NOT NULL,
    input_summary           TEXT,
    output_summary          TEXT,
    risk_level              TEXT NOT NULL,
    requires_human_review   INTEGER NOT NULL DEFAULT 0,
    created_at              TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS incidents (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id                TEXT NOT NULL,
    severity                TEXT NOT NULL,
    category                TEXT NOT NULL,
    description             TEXT NOT NULL,
    detected_at             TEXT NOT NULL,
    resolved_at             TEXT,
    root_cause              TEXT,
    corrective_action       TEXT,
    reported_to_regulator   INTEGER NOT NULL DEFAULT 0,
    regulator_report_ref    TEXT,
    reported_at             TEXT
);

CREATE TABLE IF NOT EXISTS safety_reports (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        TEXT NOT NULL,
    report          TEXT NOT NULL,
    generated_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ethics_labels (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id        TEXT NOT NULL,
    tier            INTEGER NOT NULL,
    tier_name_ar    TEXT NOT NULL,
    tier_name_en    TEXT NOT NULL,
    score           INTEGER NOT NULL,
    computed_at     TEXT NOT NULL
);
"""

_SCHEMA_SQL_PATH = Path(__file__).with_name("schema_sdaia.sql")


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _is_postgres_dsn(db_path: str) -> bool:
    return db_path.startswith("postgresql://") or db_path.startswith("postgres://")


class SDAIAStorage:
    """Storage for agents, risk assessments, decisions, incidents,
    safety reports, and ethics labels.

    Backend is chosen by db_path:
      - ":memory:" or a file path -> SQLite (dev/testing)
      - "postgresql://..." / "postgres://..." -> PostgreSQL (production)

    If db_path is omitted, falls back to the SDAIA_DSN env var, then
    ":memory:".
    """

    def __init__(self, db_path: str | None = None):
        db_path = db_path or os.getenv("SDAIA_DSN", ":memory:")
        self.is_postgres = _is_postgres_dsn(db_path)

        if self.is_postgres:
            import psycopg2
            import psycopg2.extras

            self._psycopg2 = psycopg2
            self._extras = psycopg2.extras
            self.conn = psycopg2.connect(db_path, connect_timeout=5)
            self._init_schema_postgres()
        else:
            self.conn = sqlite3.connect(db_path)
            self.conn.row_factory = sqlite3.Row
            self.conn.executescript(SCHEMA)
            self.conn.commit()

    def _init_schema_postgres(self) -> None:
        sql = _SCHEMA_SQL_PATH.read_text(encoding="utf-8")
        cur = self.conn.cursor()
        cur.execute(sql)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # -- backend helpers -----------------------------------------------------

    def _cursor(self):
        if self.is_postgres:
            return self.conn.cursor(cursor_factory=self._extras.RealDictCursor)
        return self.conn.cursor()

    def _q(self, sql: str) -> str:
        """Translate sqlite-style '?' placeholders to psycopg2-style '%s'."""
        return sql.replace("?", "%s") if self.is_postgres else sql

    def _bool_param(self, value: bool):
        return bool(value) if self.is_postgres else int(bool(value))

    def _json_param(self, obj: dict):
        return self._extras.Json(obj) if self.is_postgres else json.dumps(obj)

    def _decode_json(self, value):
        # psycopg2 deserializes JSONB columns to Python objects already.
        return value if self.is_postgres else json.loads(value)

    def _insert_and_get_id(self, sql: str, params: tuple) -> int:
        cur = self._cursor()
        if self.is_postgres:
            cur.execute(self._q(sql) + " RETURNING id", params)
            new_id = cur.fetchone()["id"]
        else:
            cur.execute(sql, params)
            new_id = cur.lastrowid
        self.conn.commit()
        return new_id

    # -- agents ----------------------------------------------------------

    def register_agent(
        self,
        agent_id: str,
        name: str,
        sector: str,
        description: str = "",
        handles_pii: bool = False,
        autonomy_level: str = "assisted",
        pl_agent_id: str | None = None,
    ) -> dict:
        """Register (or update) an agent. pl_agent_id is an optional link to
        this repo's pl_agents(agent_id) registry for cross-referencing —
        not enforced as a foreign key here, since this module may be used
        standalone before pl_agents exists."""
        handles_pii_val = self._bool_param(handles_pii)
        cur = self._cursor()
        if self.is_postgres:
            cur.execute(
                """
                INSERT INTO agents
                    (agent_id, name, sector, description, handles_pii, autonomy_level, pl_agent_id)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (agent_id) DO UPDATE SET
                    name = EXCLUDED.name,
                    sector = EXCLUDED.sector,
                    description = EXCLUDED.description,
                    handles_pii = EXCLUDED.handles_pii,
                    autonomy_level = EXCLUDED.autonomy_level,
                    pl_agent_id = EXCLUDED.pl_agent_id
                """,
                (agent_id, name, sector, description, handles_pii_val, autonomy_level, pl_agent_id),
            )
        else:
            cur.execute(
                "INSERT OR REPLACE INTO agents "
                "(agent_id, name, sector, description, handles_pii, autonomy_level, pl_agent_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (agent_id, name, sector, description, handles_pii_val, autonomy_level, pl_agent_id, _now()),
            )
        self.conn.commit()
        return self.get_agent(agent_id)

    def get_agent(self, agent_id: str) -> dict | None:
        cur = self._cursor()
        cur.execute(self._q("SELECT * FROM agents WHERE agent_id = ?"), (agent_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_agents(self) -> list[dict]:
        cur = self._cursor()
        cur.execute("SELECT * FROM agents ORDER BY created_at")
        return [dict(r) for r in cur.fetchall()]

    # -- risk assessments --------------------------------------------------

    def save_risk_assessment(
        self, agent_id: str, category: str, level: str, rationale: str, factors: dict
    ) -> int:
        sql = self._q(
            "INSERT INTO risk_assessments (agent_id, category, level, rationale, factors, assessed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        params = (agent_id, category, level, rationale, self._json_param(factors), _now())
        return self._insert_and_get_id(sql, params)

    def get_risk_assessments(self, agent_id: str) -> list[dict]:
        cur = self._cursor()
        cur.execute(self._q("SELECT * FROM risk_assessments WHERE agent_id = ? ORDER BY id"), (agent_id,))
        out = []
        for r in cur.fetchall():
            d = dict(r)
            d["factors"] = self._decode_json(d["factors"])
            out.append(d)
        return out

    # -- decisions ---------------------------------------------------------

    def log_decision(
        self,
        agent_id: str,
        decision_type: str,
        input_summary: str,
        output_summary: str,
        risk_level: str,
        requires_human_review: bool = False,
    ) -> int:
        sql = self._q(
            "INSERT INTO decisions "
            "(agent_id, decision_type, input_summary, output_summary, risk_level, requires_human_review, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)"
        )
        params = (
            agent_id,
            decision_type,
            input_summary,
            output_summary,
            risk_level,
            self._bool_param(requires_human_review),
            _now(),
        )
        return self._insert_and_get_id(sql, params)

    def list_decisions(self, agent_id: str) -> list[dict]:
        cur = self._cursor()
        cur.execute(self._q("SELECT * FROM decisions WHERE agent_id = ? ORDER BY id"), (agent_id,))
        return [dict(r) for r in cur.fetchall()]

    # -- incidents -----------------------------------------------------------

    def log_incident(
        self,
        agent_id: str,
        severity: str,
        category: str,
        description: str,
    ) -> dict:
        sql = self._q(
            "INSERT INTO incidents (agent_id, severity, category, description, detected_at) "
            "VALUES (?, ?, ?, ?, ?)"
        )
        new_id = self._insert_and_get_id(sql, (agent_id, severity, category, description, _now()))
        return self.get_incident(new_id)

    def get_incident(self, incident_id: int) -> dict | None:
        cur = self._cursor()
        cur.execute(self._q("SELECT * FROM incidents WHERE id = ?"), (incident_id,))
        row = cur.fetchone()
        return dict(row) if row else None

    def list_incidents(
        self, agent_id: str | None = None, severity: str | None = None
    ) -> list[dict]:
        query = "SELECT * FROM incidents WHERE 1=1"
        params: list[Any] = []
        if agent_id is not None:
            query += " AND agent_id = ?"
            params.append(agent_id)
        if severity is not None:
            query += " AND severity = ?"
            params.append(severity)
        query += " ORDER BY id"
        cur = self._cursor()
        cur.execute(self._q(query), params)
        return [dict(r) for r in cur.fetchall()]

    def update_incident(self, incident_id: int, **fields) -> dict:
        if not fields:
            return self.get_incident(incident_id)
        cols = ", ".join(f"{k} = ?" for k in fields)
        params = list(fields.values()) + [incident_id]
        cur = self._cursor()
        cur.execute(self._q(f"UPDATE incidents SET {cols} WHERE id = ?"), params)
        self.conn.commit()
        return self.get_incident(incident_id)

    def mark_incident_reported(self, incident_id: int) -> dict:
        report_ref = f"SDAIA-RPT-{uuid.uuid4().hex[:10].upper()}"
        return self.update_incident(
            incident_id,
            reported_to_regulator=self._bool_param(True),
            regulator_report_ref=report_ref,
            reported_at=_now(),
        )

    def resolve_incident(self, incident_id: int, root_cause: str, corrective_action: str) -> dict:
        return self.update_incident(
            incident_id,
            root_cause=root_cause,
            corrective_action=corrective_action,
            resolved_at=_now(),
        )

    # -- safety reports ------------------------------------------------------

    def save_safety_report(self, agent_id: str, report: dict) -> int:
        sql = self._q("INSERT INTO safety_reports (agent_id, report, generated_at) VALUES (?, ?, ?)")
        return self._insert_and_get_id(sql, (agent_id, self._json_param(report), _now()))

    def get_latest_safety_report(self, agent_id: str) -> dict | None:
        cur = self._cursor()
        cur.execute(
            self._q("SELECT * FROM safety_reports WHERE agent_id = ? ORDER BY id DESC LIMIT 1"),
            (agent_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        d = dict(row)
        d["report"] = self._decode_json(d["report"])
        return d

    # -- ethics labels ---------------------------------------------------------

    def save_ethics_label(
        self, agent_id: str, tier: int, tier_name_ar: str, tier_name_en: str, score: int
    ) -> int:
        sql = self._q(
            "INSERT INTO ethics_labels (agent_id, tier, tier_name_ar, tier_name_en, score, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
        return self._insert_and_get_id(sql, (agent_id, tier, tier_name_ar, tier_name_en, score, _now()))

    def get_latest_ethics_label(self, agent_id: str) -> dict | None:
        cur = self._cursor()
        cur.execute(
            self._q("SELECT * FROM ethics_labels WHERE agent_id = ? ORDER BY id DESC LIMIT 1"),
            (agent_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None


def to_epoch(ts) -> float:
    """Convert a stored timestamp (sqlite ISO string or Postgres datetime) to epoch seconds."""
    if isinstance(ts, str):
        return calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%SZ"))
    return ts.timestamp()
