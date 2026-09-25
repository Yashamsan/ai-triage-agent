-- AI Triage Agent — Arabic/English schema
-- Requires pgvector extension (vector dim = 1024, BAAI/bge-m3 — see
-- shared/embeddings.py; migrate_vector_dim() below handles the one-time
-- resize from the old 384-dim all-MiniLM-L6-v2 columns)

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS faq_articles (
    id          SERIAL PRIMARY KEY,
    intent      VARCHAR(50)  NOT NULL,
    title       TEXT         NOT NULL,
    content     TEXT         NOT NULL,
    embedding   vector(1024),
    lang        VARCHAR(10)  DEFAULT 'en',
    created_at  TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS support_tickets (
    id           SERIAL PRIMARY KEY,
    user_message TEXT         NOT NULL,
    intent       VARCHAR(50),
    status       VARCHAR(20)  DEFAULT 'open',
    embedding    vector(1024),
    lang         VARCHAR(10)  DEFAULT 'en',
    created_at   TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversation_history (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(100) NOT NULL,
    role        VARCHAR(20)  NOT NULL,
    message     TEXT         NOT NULL,
    intent      VARCHAR(50),
    created_at  TIMESTAMP    DEFAULT NOW()
);

-- Private company knowledge base (product_inquiry intent) — same physical
-- table as app/schema.sql's; declared here too (IF NOT EXISTS) so either
-- pipeline can apply its own schema.sql standalone and still get it.
CREATE TABLE IF NOT EXISTS knowledge_base_chunks (
    id          SERIAL PRIMARY KEY,
    source_file TEXT         NOT NULL,
    row_ref     TEXT,
    title       TEXT,
    content     TEXT         NOT NULL,
    lang        VARCHAR(10)  DEFAULT 'en',
    metadata    JSONB        DEFAULT '{}',
    embedding   vector(1024),
    created_at  TIMESTAMP    DEFAULT NOW()
);
ALTER TABLE knowledge_base_chunks ADD COLUMN IF NOT EXISTS lang VARCHAR(10) DEFAULT 'en';

-- Same migration helper as app/schema.sql (duplicated, not imported, so this
-- file stays applicable standalone per the comment above knowledge_base_chunks).
-- See app/schema.sql for the full explanation.
CREATE OR REPLACE FUNCTION migrate_vector_dim(
    p_table TEXT, p_column TEXT, p_dim INT
) RETURNS void AS $$
DECLARE
    current_dim INT;
BEGIN
    SELECT atttypmod INTO current_dim
    FROM pg_attribute
    WHERE attrelid = p_table::regclass
      AND attname = p_column
      AND NOT attisdropped;

    IF current_dim IS DISTINCT FROM p_dim THEN
        EXECUTE format('ALTER TABLE %I DROP COLUMN IF EXISTS %I', p_table, p_column);
        EXECUTE format('ALTER TABLE %I ADD COLUMN %I vector(%s)', p_table, p_column, p_dim);
    END IF;
END;
$$ LANGUAGE plpgsql;

SELECT migrate_vector_dim('faq_articles', 'embedding', 1024);
SELECT migrate_vector_dim('support_tickets', 'embedding', 1024);
SELECT migrate_vector_dim('knowledge_base_chunks', 'embedding', 1024);

-- HNSW indexes
CREATE INDEX IF NOT EXISTS faq_embedding_hnsw
    ON faq_articles USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS ticket_embedding_hnsw
    ON support_tickets USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS history_session_idx
    ON conversation_history (session_id);

CREATE INDEX IF NOT EXISTS kb_chunk_embedding_hnsw
    ON knowledge_base_chunks USING hnsw (embedding vector_cosine_ops);
