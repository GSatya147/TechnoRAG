-- Enable the pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS articles (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(), -- uuid instead of bigserial, to avoid conflicts while parallelism in cloud run
    url             TEXT NOT NULL UNIQUE,                       -- url unique to avoid duplicates
    content_hash    TEXT NOT NULL,                              -- hash not unique, same url is allowed to get a new hash without conflicts and to handle hash collisions
    published_at    TIMESTAMPTZ,                                -- tz captures timezone, and it can be null due to inconsistencies of parser or source itself
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    source          TEXT NOT NULL,
    ingest_status   TEXT NOT NULL DEFAULT 'pending'             -- check instead of enum, enum needs alter by extra params and system which iterates over time - check better
        CHECK (ingest_status IN ('pending', 'chunked', 'failed')),
    title           TEXT NOT NULL,
    author          TEXT
);

CREATE TABLE IF NOT EXISTS article_nodes(
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id      UUID NOT NULL 
        REFERENCES articles(id) ON DELETE CASCADE,              -- cascade to avoid orphaned article nodes in case original article row removed
    embedding       vector(1024) NOT NULL,                      -- voyage outputs 1024 dims
    sparse_tsv      TSVECTOR GENERATED
        ALWAYS AS (to_tsvector('english', chunk_text)) STORED,  -- generated, sparse vector always in-sync with chunk text, english config for BM25 to catch morphological terms
    node_type       TEXT NOT NULL CHECK (node_type IN ('summary', 'chunk')),
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    tokens_count    INT NOT NULL,
    metadata        JSONB NOT NULL DEFAULT '{}'
);

-- higher m, better recall
-- production defaults
CREATE INDEX article_nodes_embedding_hnsw_idx
ON article_nodes
USING hnsw (embedding vector_cosine_ops)
WITH (m=16, ef_construction = 64);

-- gin over gist, gin is read-optimised and our system is heavily read-oriented
-- gist is preferred for write 
CREATE INDEX article_nodes_sparse_tsv_gin_idx
ON article_nodes
USING gin (sparse_tsv);

-- B-Tree on article_id
-- when cascade-delete, postgres tries full table scan to remove the orphaned article_nodes
CREATE INDEX articles_nodes_article_id_idx
ON article_nodes (article_id);

-- B-Tree on metadata published_at
-- avoids full table scan for temporal layer while filtering
CREATE INDEX articles_nodes_published_at_idx
ON article_nodes ((metadata ->> 'published_at')); 


