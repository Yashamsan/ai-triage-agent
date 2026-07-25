-- AI Triage Agent — Arabic/English schema
-- Requires pgvector extension (vector dim = 384)
-- Compatible with both all-MiniLM-L6-v2 and paraphrase-multilingual-MiniLM-L12-v2

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS faq_articles (
    id          SERIAL PRIMARY KEY,
    intent      VARCHAR(50)  NOT NULL,
    title       TEXT         NOT NULL,
    content     TEXT         NOT NULL,
    embedding   vector(384),
    lang        VARCHAR(10)  DEFAULT 'en',
    created_at  TIMESTAMP    DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS support_tickets (
    id           SERIAL PRIMARY KEY,
    user_message TEXT         NOT NULL,
    intent       VARCHAR(50),
    status       VARCHAR(20)  DEFAULT 'open',
    embedding    vector(384),
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
    embedding   vector(384),
    created_at  TIMESTAMP    DEFAULT NOW()
);
ALTER TABLE knowledge_base_chunks ADD COLUMN IF NOT EXISTS lang VARCHAR(10) DEFAULT 'en';

-- HNSW indexes
CREATE INDEX IF NOT EXISTS faq_embedding_hnsw
    ON faq_articles USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS ticket_embedding_hnsw
    ON support_tickets USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS history_session_idx
    ON conversation_history (session_id);

CREATE INDEX IF NOT EXISTS kb_chunk_embedding_hnsw
    ON knowledge_base_chunks USING hnsw (embedding vector_cosine_ops);
