-- ProofLayer SDAIA Compliance Module — PostgreSQL schema (production)
--
-- NOTE: This schema is an illustrative governance scaffold, not a
-- verified transcription of SDAIA's actual published requirements.
-- Section numbers referenced in comments throughout this module
-- (e.g. "7.2", "8.8") are placeholders for wherever your org's real
-- compliance mapping lives — replace with your verified references
-- before using this for an actual regulatory submission.
--
-- The demo/test suite runs against SQLite (app/sdaia_storage.py).
-- This file defines the equivalent shape for a Postgres deployment.

CREATE TABLE IF NOT EXISTS agents (
    agent_id        TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    sector          TEXT NOT NULL,
    description     TEXT,
    handles_pii     BOOLEAN NOT NULL DEFAULT FALSE,
    autonomy_level  TEXT NOT NULL DEFAULT 'assisted',   -- assisted | supervised | autonomous
    -- Optional cross-reference to this repo's pl_agents(agent_id) registry
    -- (see app/schema_v2.sql). Left as a plain column, not a FK, so this
    -- module can be applied standalone before pl_agents exists. If you want
    -- referential integrity once both schemas are present, add it manually:
    --   ALTER TABLE agents ADD CONSTRAINT fk_agents_pl_agent
    --     FOREIGN KEY (pl_agent_id) REFERENCES pl_agents(agent_id);
    pl_agent_id     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS risk_assessments (
    id              SERIAL PRIMARY KEY,
    agent_id        TEXT NOT NULL REFERENCES agents(agent_id),
    category        TEXT NOT NULL,          -- one of the 7 risk categories
    level           TEXT NOT NULL,          -- MINIMAL | LIMITED | HIGH | CRITICAL
    rationale       TEXT NOT NULL,
    factors         JSONB NOT NULL DEFAULT '{}',
    assessed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS decisions (
    id                      SERIAL PRIMARY KEY,
    agent_id                TEXT NOT NULL REFERENCES agents(agent_id),
    decision_type           TEXT NOT NULL,
    input_summary           TEXT,
    output_summary          TEXT,
    risk_level              TEXT NOT NULL,
    requires_human_review   BOOLEAN NOT NULL DEFAULT FALSE,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS incidents (
    id                      SERIAL PRIMARY KEY,
    agent_id                TEXT NOT NULL REFERENCES agents(agent_id),
    severity                TEXT NOT NULL,      -- MINIMAL | LIMITED | HIGH | CRITICAL
    category                TEXT NOT NULL,
    description             TEXT NOT NULL,
    detected_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    resolved_at             TIMESTAMPTZ,
    root_cause              TEXT,
    corrective_action       TEXT,
    reported_to_regulator   BOOLEAN NOT NULL DEFAULT FALSE,
    regulator_report_ref    TEXT,
    reported_at             TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS safety_reports (
    id              SERIAL PRIMARY KEY,
    agent_id        TEXT NOT NULL REFERENCES agents(agent_id),
    report          JSONB NOT NULL,
    generated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS ethics_labels (
    id              SERIAL PRIMARY KEY,
    agent_id        TEXT NOT NULL REFERENCES agents(agent_id),
    tier            INTEGER NOT NULL,       -- 1..5
    tier_name_ar    TEXT NOT NULL,
    tier_name_en    TEXT NOT NULL,
    score           INTEGER NOT NULL,
    computed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_risk_agent ON risk_assessments(agent_id);
CREATE INDEX IF NOT EXISTS idx_decisions_agent ON decisions(agent_id);
CREATE INDEX IF NOT EXISTS idx_incidents_agent ON incidents(agent_id);
CREATE INDEX IF NOT EXISTS idx_incidents_severity ON incidents(severity);
CREATE INDEX IF NOT EXISTS idx_safety_reports_agent ON safety_reports(agent_id);
CREATE INDEX IF NOT EXISTS idx_labels_agent ON ethics_labels(agent_id);
