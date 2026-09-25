-- ProofLayer SDAIA-P145 National AI Risk Management Framework -- PostgreSQL schema
--
-- Ported from prooflayer-sdaia/P145/prooflayer-rmf (a standalone SQLite
-- prototype with its own port/dashboard/tests -- all 9 of its tests passed
-- before this port). This file targets the same Postgres database as the
-- rest of ProofLayer so the module can live as a tab in the same admin UI,
-- but stays a standalone module in the sense that matters: no foreign key
-- into pl_agents or the agents (SDAIA Responsible AI Policy) table --
-- agents are referenced by plain string id, exactly as the prototype's own
-- schema comment specified. Zero impact on the Responsible AI Policy
-- (Mar 2026) module's tables or behavior.
--
-- Risk level = likelihood x impact (both 1-4).
-- Bands: 1-2 low, 3-6 medium, 8-12 high, 16 catastrophic.
-- 7, 13, 14, 15 are impossible products of 1-4 x 1-4 and are not valid levels.

CREATE TABLE IF NOT EXISTS rmf_contexts (
    context_id       TEXT PRIMARY KEY,
    agent_id         TEXT NOT NULL UNIQUE,
    description      TEXT DEFAULT '',
    data_io          TEXT DEFAULT '',
    automation_level TEXT DEFAULT '',
    human_role       TEXT DEFAULT '',
    lifecycle_stage  TEXT DEFAULT '',
    change_plan      TEXT DEFAULT '',
    created_at       TIMESTAMPTZ NOT NULL DEFAULT NOW()
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
    status        TEXT NOT NULL DEFAULT 'open',   -- open | assessed | treated | accepted | closed
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rmf_assessments (
    assessment_id TEXT PRIMARY KEY,
    risk_id       TEXT NOT NULL REFERENCES rmf_risks(risk_id) ON DELETE CASCADE,
    likelihood    INTEGER NOT NULL CHECK (likelihood BETWEEN 1 AND 4),
    impact        INTEGER NOT NULL CHECK (impact BETWEEN 1 AND 4),
    risk_level    INTEGER NOT NULL,
    band          TEXT NOT NULL,
    evidence      TEXT DEFAULT '',
    assessed_by   TEXT DEFAULT '',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rmf_treatments (
    treatment_id        TEXT PRIMARY KEY,
    risk_id              TEXT NOT NULL REFERENCES rmf_risks(risk_id) ON DELETE CASCADE,
    strategy             TEXT NOT NULL CHECK (strategy IN ('avoid','mitigate','transfer','accept')),
    rationale            TEXT NOT NULL,
    controls             TEXT DEFAULT '',
    residual_likelihood  INTEGER,
    residual_impact      INTEGER,
    residual_level       INTEGER,
    approver             TEXT NOT NULL,
    approval_mechanism   TEXT DEFAULT '',
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS rmf_reviews (
    review_id    TEXT PRIMARY KEY,
    agent_id     TEXT NOT NULL,
    review_type  TEXT NOT NULL DEFAULT 'periodic',   -- periodic | incident | triggered
    findings     TEXT DEFAULT '',
    incident_rca TEXT DEFAULT '',
    reviewed_by  TEXT DEFAULT '',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_rmf_risks_agent ON rmf_risks(agent_id);
CREATE INDEX IF NOT EXISTS idx_rmf_risks_category ON rmf_risks(category);
CREATE INDEX IF NOT EXISTS idx_rmf_assessments_risk ON rmf_assessments(risk_id);
CREATE INDEX IF NOT EXISTS idx_rmf_treatments_risk ON rmf_treatments(risk_id);
CREATE INDEX IF NOT EXISTS idx_rmf_reviews_agent ON rmf_reviews(agent_id);
