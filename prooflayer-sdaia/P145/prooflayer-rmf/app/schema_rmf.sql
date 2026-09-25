-- SDAIA-P145 National AI Risk Management Framework - ProofLayer module
-- v1.0 (2026-08-17). Standalone SQLite schema.
-- Zero impact on the Responsible AI Policy (Mar 2026) module: this schema
-- references agents by string id only; no FK into the main pl_agents tables.
--
-- Risk level = likelihood x impact (both 1-4).
-- Bands: 1-2 low, 3-6 medium, 8-12 high, 16 catastrophic.
-- NOTE: 7, 13, 14, 15 are impossible products of 1-4 x 1-4 (per the official matrix).

CREATE TABLE IF NOT EXISTS rmf_contexts (
    context_id       TEXT PRIMARY KEY,
    agent_id         TEXT NOT NULL UNIQUE,
    description      TEXT DEFAULT '',
    data_io          TEXT DEFAULT '',
    automation_level TEXT DEFAULT '',
    human_role       TEXT DEFAULT '',
    lifecycle_stage  TEXT DEFAULT '',
    change_plan      TEXT DEFAULT '',
    created_at       TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rmf_risks (
    risk_id       TEXT PRIMARY KEY,
    agent_id      TEXT NOT NULL,
    category      TEXT NOT NULL,
    description   TEXT NOT NULL,
    source        TEXT DEFAULT '',
    intent        TEXT DEFAULT '',
    timing        TEXT DEFAULT '',
    identified_by TEXT DEFAULT '',
    status        TEXT DEFAULT 'open',   -- open | assessed | treated | accepted | closed
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rmf_assessments (
    assessment_id TEXT PRIMARY KEY,
    risk_id       TEXT NOT NULL REFERENCES rmf_risks(risk_id),
    likelihood    INTEGER NOT NULL CHECK (likelihood BETWEEN 1 AND 4),
    impact        INTEGER NOT NULL CHECK (impact BETWEEN 1 AND 4),
    risk_level    INTEGER NOT NULL,
    band          TEXT NOT NULL,
    evidence      TEXT DEFAULT '',
    assessed_by   TEXT DEFAULT '',
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rmf_treatments (
    treatment_id       TEXT PRIMARY KEY,
    risk_id            TEXT NOT NULL REFERENCES rmf_risks(risk_id),
    strategy           TEXT NOT NULL CHECK (strategy IN ('avoid','mitigate','transfer','accept')),
    rationale          TEXT NOT NULL,
    controls           TEXT DEFAULT '',
    residual_likelihood INTEGER,
    residual_impact     INTEGER,
    residual_level      INTEGER,
    approver           TEXT NOT NULL,
    approval_mechanism TEXT DEFAULT '',
    created_at         TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS rmf_reviews (
    review_id    TEXT PRIMARY KEY,
    agent_id     TEXT NOT NULL,
    review_type  TEXT DEFAULT 'periodic',   -- periodic | incident | triggered
    findings     TEXT DEFAULT '',
    incident_rca TEXT DEFAULT '',
    reviewed_by  TEXT DEFAULT '',
    created_at   TEXT DEFAULT (datetime('now'))
);
