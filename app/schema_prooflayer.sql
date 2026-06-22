-- ProofLayer Context Graph — Core Schema
-- Extends existing triage agent schema with generic graph storage
-- Requires ltree extension for hierarchy queries

CREATE EXTENSION IF NOT EXISTS ltree;
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ══════════════════════════════════════════════════════════
-- Nodes — every entity, decision, policy, precedent, etc.
-- ══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS pl_nodes (
    node_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_type   VARCHAR(50) NOT NULL,   -- Decision | Policy | Precedent | Entity | ContextSnapshot
    properties  JSONB NOT NULL DEFAULT '{}',
    path        LTREE,                   -- hierarchy navigation (e.g. "triage.severity.high")
    embedding   VECTOR(384),             -- pgvector for semantic search
    valid_from  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to    TIMESTAMPTZ,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- ══════════════════════════════════════════════════════════
-- Edges — typed, directed relationships between nodes
-- ══════════════════════════════════════════════════════════

CREATE TABLE IF NOT EXISTS pl_edges (
    edge_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    from_node_id UUID NOT NULL REFERENCES pl_nodes(node_id) ON DELETE CASCADE,
    to_node_id   UUID NOT NULL REFERENCES pl_nodes(node_id) ON DELETE CASCADE,
    edge_type    VARCHAR(50) NOT NULL,   -- CAUSED | INFLUENCED | APPLIED_POLICY | PRECEDENT_FOR | GRANTED_EXCEPTION | ABOUT | USED_CONTEXT | OVERTURNED_BY
    properties   JSONB NOT NULL DEFAULT '{}',
    valid_from   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_to     TIMESTAMPTZ,
    CONSTRAINT unique_edge UNIQUE (from_node_id, to_node_id, edge_type, valid_from)
);

-- ══════════════════════════════════════════════════════════
-- Indexes
-- ══════════════════════════════════════════════════════════

-- Lookup by type + temporal filter
CREATE INDEX IF NOT EXISTS idx_pl_nodes_type
    ON pl_nodes (node_type, valid_from DESC);

-- Precedent matching by pattern_signature
CREATE INDEX IF NOT EXISTS idx_pl_precedent_pattern
    ON pl_nodes USING gin (properties jsonb_path_ops)
    WHERE node_type = 'Precedent';

-- pgvector ANN search for semantic precedent matching
CREATE INDEX IF NOT EXISTS idx_pl_nodes_embedding
    ON pl_nodes USING hnsw (embedding vector_cosine_ops);

-- ltree path index for hierarchy queries
CREATE INDEX IF NOT EXISTS idx_pl_nodes_path
    ON pl_nodes USING gist (path);

-- Edge lookups
CREATE INDEX IF NOT EXISTS idx_pl_edges_from
    ON pl_edges (from_node_id, edge_type);

CREATE INDEX IF NOT EXISTS idx_pl_edges_to
    ON pl_edges (to_node_id, edge_type);

-- ══════════════════════════════════════════════════════════
-- Seed: Policy nodes (governing rules the agent follows)
-- ══════════════════════════════════════════════════════════

INSERT INTO pl_nodes (node_type, properties, path) VALUES
('Policy', '{
    "policy_id": "severity_v1",
    "policy_version": 1,
    "policy_text": "password_reset → FAQ lookup; billing → FAQ lookup; technical_support → FAQ lookup; escalation → ticket creation",
    "effective_from": "2026-06-01T00:00:00Z",
    "jurisdiction": "internal"
}'::jsonb, 'triage.policies.severity'),
('Policy', '{
    "policy_id": "escalation_thresholds_v1",
    "policy_version": 1,
    "policy_text": "needs_escalation=true → create_support_ticket and route to human agent",
    "effective_from": "2026-06-01T00:00:00Z",
    "jurisdiction": "internal"
}'::jsonb, 'triage.policies.escalation'),
('Policy', '{
    "policy_id": "reflection_override_v1",
    "policy_version": 1,
    "policy_text": "LLM-as-Judge reflection may override original classification when safety risk detected",
    "effective_from": "2026-06-22T00:00:00Z",
    "jurisdiction": "internal"
}'::jsonb, 'triage.policies.reflection')
ON CONFLICT DO NOTHING;

-- ══════════════════════════════════════════════════════════
-- Seed: Entity nodes (core domain objects)
-- ══════════════════════════════════════════════════════════

INSERT INTO pl_nodes (node_type, properties, path) VALUES
('Entity', '{
    "entity_type": "system",
    "name": "ai-triage-agent",
    "description": "Customer support triage agent with LangGraph, RAG, observability"
}'::jsonb, 'triage.entities.system.agent'),
('Entity', '{
    "entity_type": "system",
    "name": "postgresql-db",
    "description": "PostgreSQL with pgvector for FAQ, tickets, conversation history"
}'::jsonb, 'triage.entities.system.database'),
('Entity', '{
    "entity_type": "system",
    "name": "langfuse-v3",
    "description": "Self-hosted LangFuse observability stack"
}'::jsonb, 'triage.entities.system.observability')
ON CONFLICT DO NOTHING;
