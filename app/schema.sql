-- AI Triage Agent — PostgreSQL schema
-- Requires pgvector extension (vector dim = 1024, BAAI/bge-m3 — see
-- shared/embeddings.py; migrate_vector_dim() below handles the one-time
-- resize from the old 384-dim all-MiniLM-L6-v2 columns)

CREATE EXTENSION IF NOT EXISTS vector;

-- FAQ knowledge base — one row per article, indexed by intent
CREATE TABLE IF NOT EXISTS faq_articles (
    id          SERIAL PRIMARY KEY,
    intent      VARCHAR(50)  NOT NULL,
    title       TEXT         NOT NULL,
    content     TEXT         NOT NULL,
    embedding   vector(1024),
    created_at  TIMESTAMP    DEFAULT NOW()
);

-- Support tickets — created when a message needs human follow-up
CREATE TABLE IF NOT EXISTS support_tickets (
    id           SERIAL PRIMARY KEY,
    user_message TEXT         NOT NULL,
    intent       VARCHAR(50),
    status       VARCHAR(20)  DEFAULT 'open',
    embedding    vector(1024),
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
    embedding   vector(1024),
    created_at  TIMESTAMP    DEFAULT NOW()
);
-- Safe if the table already existed before this column was added.
ALTER TABLE knowledge_base_chunks ADD COLUMN IF NOT EXISTS lang VARCHAR(10) DEFAULT 'en';

-- Arabic light-normalization, SQL equivalent of app_ar/database.py's
-- _normalize_arabic(): strip diacritics and tatweel, collapse alef/hamza
-- variants to bare alef, alef-maksura to ya, ta-marbuta to ha. IMMUTABLE
-- so it can back a GENERATED STORED tsvector column below. English text
-- passes through unchanged (none of these substitutions touch Latin script).
CREATE OR REPLACE FUNCTION ar_normalize(input TEXT) RETURNS TEXT AS $$
    SELECT translate(
        regexp_replace(
            regexp_replace(coalesce(input, ''), '[ؐ-ًؚ-ٟۖ-ٰۭ]', '', 'g'),
            'ـ', '', 'g'
        ),
        'إأآاىة', 'اااايه'
    );
$$ LANGUAGE sql IMMUTABLE;

-- Full-text search column, hybrid partner to the vector embedding search.
-- Retrieval-quality eval (scripts/eval_kb_retrieval.py) showed pure vector
-- similarity alone -- against a non-Arabic-native embedding model -- only
-- got 38% of a realistic golden query set right, including missing several
-- near-exact title matches entirely. This tsvector, combined with the
-- embedding search via Reciprocal Rank Fusion in find_kb_chunk(), is the
-- literal-term-match side of that combination -- 'simple' config (no
-- language-specific stemming dictionary needed) over pre-normalized text.
ALTER TABLE knowledge_base_chunks ADD COLUMN IF NOT EXISTS search_vector tsvector
    GENERATED ALWAYS AS (
        to_tsvector('simple', ar_normalize(coalesce(title, '') || ' ' || coalesce(content, '')))
    ) STORED;

CREATE INDEX IF NOT EXISTS idx_kb_chunks_search_vector
    ON knowledge_base_chunks USING GIN (search_vector);

-- Precedent memory (long-term behavioral memory across triage decisions).
-- Populated by shared/precedent_store.py — declared here (not previously
-- tracked in any schema.sql) so a fresh DB reproduces it instead of relying
-- on undocumented manual state.
CREATE TABLE IF NOT EXISTS precedents (
    id                 SERIAL PRIMARY KEY,
    pattern_hash       VARCHAR(64)   NOT NULL,
    intent             VARCHAR(50),
    severity_decision  VARCHAR(50),
    human_correction   VARCHAR(50),
    correction_reason  TEXT,
    confidence         REAL          DEFAULT 0.0,
    success_rate       REAL          DEFAULT 0.0,
    applied_count      INTEGER       DEFAULT 0,
    full_trace         JSONB,
    embedding          vector(1024),
    created_at         TIMESTAMPTZ   DEFAULT NOW(),
    expires_at         TIMESTAMPTZ
);
CREATE INDEX IF NOT EXISTS idx_precedent_pattern ON precedents (pattern_hash);
-- idx_precedent_vector is created further down, AFTER migrate_vector_dim()
-- has a chance to drop/recreate this table's embedding column — an index
-- created here would just be dropped along with the pre-migration column.

-- One-time embedding-dimension migration helper: BAAI/bge-m3 replaced
-- all-MiniLM-L6-v2 (384-dim -> 1024-dim, see shared/embeddings.py). A
-- vector column can't be resized in place while it holds data of the old
-- dimension, so this drops and re-adds the column when its current
-- dimension doesn't match target_dim -- existing embeddings in that column
-- are lost (expected: every embedding-bearing table gets re-populated by
-- its own ingestion/seed path right after a model swap) but every other
-- column and row is untouched. Safe to re-run: a no-op once already at
-- target_dim. Used below for this file's tables, and from schema_v2.sql
-- for pl_nodes.embedding (defined here because schema.sql always applies
-- first — see apply_schema()'s call order in app/database.py).
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
SELECT migrate_vector_dim('precedents', 'embedding', 1024);

-- HNSW indexes for fast cosine similarity search
CREATE INDEX IF NOT EXISTS faq_embedding_hnsw
    ON faq_articles USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS ticket_embedding_hnsw
    ON support_tickets USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS history_session_idx
    ON conversation_history (session_id);

CREATE INDEX IF NOT EXISTS kb_chunk_embedding_hnsw
    ON knowledge_base_chunks USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS idx_precedent_vector
    ON precedents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 10);
