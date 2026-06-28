-- ProofLayer v3 Schema Migration
-- Run after schema.sql and schema_v2.sql
-- Idempotent: all statements use IF NOT EXISTS / ADD COLUMN IF NOT EXISTS

-- ── pl_agents extensions ──────────────────────────────────────────────────────

ALTER TABLE pl_agents
    ADD COLUMN IF NOT EXISTS agent_group       TEXT,
    ADD COLUMN IF NOT EXISTS data_classification TEXT DEFAULT 'internal';

-- ── pl_nodes extensions ───────────────────────────────────────────────────────

-- PII flag and agent group denormalised onto Decision nodes for fast filtering
ALTER TABLE pl_nodes
    ADD COLUMN IF NOT EXISTS contains_pii  BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS agent_group   TEXT;

-- ── pl_edges extensions ───────────────────────────────────────────────────────

ALTER TABLE pl_edges
    ADD COLUMN IF NOT EXISTS edge_metadata JSONB DEFAULT '{}';

-- ── pl_trace_steps ────────────────────────────────────────────────────────────
-- One row per Thought→Action→Observation step recorded inside a decision.

CREATE TABLE IF NOT EXISTS pl_trace_steps (
    step_id         BIGSERIAL PRIMARY KEY,
    decision_node_id BIGINT    NOT NULL REFERENCES pl_nodes(node_id) ON DELETE CASCADE,
    step_order      INT        NOT NULL DEFAULT 0,
    node_type       TEXT       NOT NULL,   -- classifier / reflect / tool_runner / …
    thought         TEXT,
    action          TEXT,
    observation     TEXT,
    confidence      FLOAT,
    latency_ms      FLOAT,
    extra           JSONB      DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trace_steps_decision
    ON pl_trace_steps (decision_node_id, step_order);

-- ── pl_exceptions ────────────────────────────────────────────────────────────
-- Ghost Knowledge: human narrative behind every policy exception or override.

CREATE TABLE IF NOT EXISTS pl_exceptions (
    exception_id      BIGSERIAL PRIMARY KEY,
    decision_node_id  BIGINT    NOT NULL REFERENCES pl_nodes(node_id) ON DELETE CASCADE,
    human_narrative   TEXT      NOT NULL,
    approver          TEXT,
    approval_channel  TEXT,                -- slack / email / in-person / ticket
    policy_violated   TEXT,
    justification     TEXT,
    severity          TEXT      DEFAULT 'low',  -- low / medium / high / critical
    created_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_exceptions_decision
    ON pl_exceptions (decision_node_id);
CREATE INDEX IF NOT EXISTS idx_exceptions_severity
    ON pl_exceptions (severity, created_at DESC);

-- ── pl_decision_contexts ──────────────────────────────────────────────────────
-- Event Clock: immutable snapshot of system state at decision time.

CREATE TABLE IF NOT EXISTS pl_decision_contexts (
    context_id        BIGSERIAL PRIMARY KEY,
    decision_node_id  BIGINT    NOT NULL REFERENCES pl_nodes(node_id) ON DELETE CASCADE,
    model_version     TEXT,
    active_policies   JSONB     DEFAULT '[]',
    risk_scores       JSONB     DEFAULT '{}',
    feature_flags     JSONB     DEFAULT '{}',
    system_load       JSONB     DEFAULT '{}',
    captured_at       TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_decision_contexts_decision
    ON pl_decision_contexts (decision_node_id);

-- ── pl_policies ───────────────────────────────────────────────────────────────
-- Policy-as-Code: versioned policy definitions.

CREATE TABLE IF NOT EXISTS pl_policies (
    policy_id    BIGSERIAL  PRIMARY KEY,
    policy_code  TEXT       NOT NULL,        -- e.g. "pwd_reset_v2"
    version      TEXT       NOT NULL DEFAULT '1.0',
    name         TEXT       NOT NULL,
    description  TEXT,
    rules        JSONB      DEFAULT '{}',
    effective_from TIMESTAMPTZ DEFAULT NOW(),
    effective_to   TIMESTAMPTZ,
    created_by   TEXT,
    UNIQUE (policy_code, version)
);

-- ── pl_policy_agent_map ───────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pl_policy_agent_map (
    map_id       BIGSERIAL PRIMARY KEY,
    policy_id    BIGINT    NOT NULL REFERENCES pl_policies(policy_id) ON DELETE CASCADE,
    agent_name   TEXT      NOT NULL,
    applied_from TIMESTAMPTZ DEFAULT NOW(),
    applied_to   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_policy_agent_map_agent
    ON pl_policy_agent_map (agent_name, applied_from);

-- ── pl_data_classifications ───────────────────────────────────────────────────
-- PDPL / PII audit trail: what data types does each agent process?

CREATE TABLE IF NOT EXISTS pl_data_classifications (
    classification_id BIGSERIAL PRIMARY KEY,
    agent_name        TEXT      NOT NULL,
    data_types        JSONB     DEFAULT '[]',   -- ["name", "email", "phone", …]
    pii_types         JSONB     DEFAULT '[]',   -- PDPL categories
    retention_days    INT       DEFAULT 365,
    encryption_tier   TEXT      DEFAULT 'standard',  -- standard / enhanced / restricted
    processing_basis  TEXT,                     -- PDPL legal basis
    data_controller   TEXT,
    created_at        TIMESTAMPTZ DEFAULT NOW(),
    updated_at        TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_data_class_agent
    ON pl_data_classifications (agent_name);

-- ── Governance indexes ────────────────────────────────────────────────────────

CREATE INDEX IF NOT EXISTS idx_nodes_contains_pii
    ON pl_nodes (contains_pii, created_at DESC)
    WHERE contains_pii = TRUE;

CREATE INDEX IF NOT EXISTS idx_nodes_agent_group
    ON pl_nodes (agent_group, node_type, created_at DESC)
    WHERE agent_group IS NOT NULL;

-- New edge type index for cross-agent references
CREATE INDEX IF NOT EXISTS idx_edges_cross_agent
    ON pl_edges (edge_type, valid_from DESC)
    WHERE edge_type = 'CROSS_AGENT_REFERENCE';
