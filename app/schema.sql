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

-- HNSW indexes for fast cosine similarity search
CREATE INDEX IF NOT EXISTS faq_embedding_hnsw
    ON faq_articles USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS ticket_embedding_hnsw
    ON support_tickets USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS history_session_idx
    ON conversation_history (session_id);
