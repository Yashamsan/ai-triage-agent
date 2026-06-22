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

-- HNSW indexes
CREATE INDEX IF NOT EXISTS faq_embedding_hnsw
    ON faq_articles USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS ticket_embedding_hnsw
    ON support_tickets USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS history_session_idx
    ON conversation_history (session_id);
