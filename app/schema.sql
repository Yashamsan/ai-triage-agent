-- AI Triage Agent — PostgreSQL schema
-- Requires pgvector extension (vector dim = 384, all-MiniLM-L6-v2)

CREATE EXTENSION IF NOT EXISTS vector;

-- FAQ knowledge base — one row per article, indexed by intent
CREATE TABLE IF NOT EXISTS faq_articles (
    id          SERIAL PRIMARY KEY,
    intent      VARCHAR(50)  NOT NULL,
    title       TEXT         NOT NULL,
    content     TEXT         NOT NULL,
    embedding   vector(384),
    created_at  TIMESTAMP    DEFAULT NOW()
);

-- Support tickets — created when a message needs human follow-up
CREATE TABLE IF NOT EXISTS support_tickets (
    id           SERIAL PRIMARY KEY,
    user_message TEXT         NOT NULL,
    intent       VARCHAR(50),
    status       VARCHAR(20)  DEFAULT 'open',
    embedding    vector(384),
    created_at   TIMESTAMP    DEFAULT NOW()
);

-- Conversation history — persists per session_id for memory
CREATE TABLE IF NOT EXISTS conversation_history (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(100) NOT NULL,
    role        VARCHAR(20)  NOT NULL,  -- 'user' | 'assistant'
    message     TEXT         NOT NULL,
    intent      VARCHAR(50),
    created_at  TIMESTAMP    DEFAULT NOW()
);

-- Private company knowledge base (product_inquiry intent). Populated by
-- scripts/ingest_knowledge_base.py from a local, gitignored source file —
-- never from anything committed to this repo. One row per source-row/chunk.
CREATE TABLE IF NOT EXISTS knowledge_base_chunks (
    id          SERIAL PRIMARY KEY,
    source_file TEXT         NOT NULL,
    row_ref     TEXT,                    -- e.g. "Sheet1!row12", traces a hit back to its source row
    title       TEXT,
    content     TEXT         NOT NULL,   -- text shown back as the answer (clean, not the embedding text)
    lang        VARCHAR(10)  DEFAULT 'en',  -- 'en' | 'ar' — same convention as faq_articles.lang
    metadata    JSONB        DEFAULT '{}',
    embedding   vector(384),
    created_at  TIMESTAMP    DEFAULT NOW()
);
-- Safe if the table already existed before this column was added.
ALTER TABLE knowledge_base_chunks ADD COLUMN IF NOT EXISTS lang VARCHAR(10) DEFAULT 'en';

-- HNSW indexes for fast cosine similarity search
CREATE INDEX IF NOT EXISTS faq_embedding_hnsw
    ON faq_articles USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS ticket_embedding_hnsw
    ON support_tickets USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS history_session_idx
    ON conversation_history (session_id);

CREATE INDEX IF NOT EXISTS kb_chunk_embedding_hnsw
    ON knowledge_base_chunks USING hnsw (embedding vector_cosine_ops);
