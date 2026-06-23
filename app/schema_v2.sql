-- ProofLayer schema v2 — agent registry + full-text search
-- Apply on top of schema_prooflayer.sql

-- ── Agent Registry ────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS pl_agents (
    agent_id      UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name    VARCHAR(255) NOT NULL,
    agent_version VARCHAR(50)  NOT NULL DEFAULT '1.0',
    model_id      VARCHAR(255) NOT NULL DEFAULT '',
    description   TEXT         NOT NULL DEFAULT '',
    registered_at TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    last_seen     TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    metadata      JSONB        NOT NULL DEFAULT '{}'
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_pl_agents_name_version
    ON pl_agents (agent_name, agent_version);

CREATE INDEX IF NOT EXISTS idx_pl_agents_last_seen
    ON pl_agents (last_seen DESC);

-- ── Extend pl_nodes with agent + search ───────────────────────────────────────

ALTER TABLE pl_nodes
    ADD COLUMN IF NOT EXISTS agent_name VARCHAR(255);

ALTER TABLE pl_nodes
    ADD COLUMN IF NOT EXISTS search_vector TSVECTOR;

CREATE INDEX IF NOT EXISTS idx_pl_nodes_agent
    ON pl_nodes (agent_name) WHERE agent_name IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_pl_nodes_type_created
    ON pl_nodes (node_type, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_pl_nodes_search
    ON pl_nodes USING GIN(search_vector);

-- ── Full-text search trigger ──────────────────────────────────────────────────
-- Indexes all text values inside the properties JSONB blob plus node_type
-- so queries like "mortgage Ahmed rejected" hit the right Decision nodes.

CREATE OR REPLACE FUNCTION pl_nodes_search_update()
RETURNS TRIGGER LANGUAGE plpgsql AS $$
BEGIN
    NEW.search_vector := to_tsvector(
        'english',
        COALESCE(NEW.node_type, '') || ' ' ||
        COALESCE(NEW.agent_name, '') || ' ' ||
        COALESCE(
            (
                SELECT string_agg(val, ' ')
                FROM jsonb_each_text(NEW.properties) AS kv(key, val)
                WHERE val IS NOT NULL
            ),
            ''
        )
    );
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_pl_nodes_search ON pl_nodes;
CREATE TRIGGER trg_pl_nodes_search
    BEFORE INSERT OR UPDATE OF properties, node_type, agent_name
    ON pl_nodes
    FOR EACH ROW EXECUTE FUNCTION pl_nodes_search_update();

-- Backfill existing rows
UPDATE pl_nodes
SET search_vector = to_tsvector(
    'english',
    COALESCE(node_type, '') || ' ' ||
    COALESCE(agent_name, '') || ' ' ||
    COALESCE(
        (
            SELECT string_agg(val, ' ')
            FROM jsonb_each_text(properties) AS kv(key, val)
            WHERE val IS NOT NULL
        ),
        ''
    )
);

-- ── Decision-value index for fast filter queries ───────────────────────────────

CREATE INDEX IF NOT EXISTS idx_pl_nodes_decision_value
    ON pl_nodes ((properties->>'decision_value'))
    WHERE node_type = 'Decision';
