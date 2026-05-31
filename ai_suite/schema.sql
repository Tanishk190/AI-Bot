-- DocuMind database schema
-- Run: psql -U documind_user -d documind -f schema.sql

CREATE TABLE IF NOT EXISTS sessions (
    id          SERIAL PRIMARY KEY,
    session_id  VARCHAR(64) UNIQUE NOT NULL,
    created_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,
    session_id   INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    filename     VARCHAR(255) NOT NULL,
    content_hash VARCHAR(64),
    raw_text     TEXT,
    tool         VARCHAR(20) NOT NULL DEFAULT 'chat',
    uploaded_at  TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id          SERIAL PRIMARY KEY,
    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text        TEXT NOT NULL,
    text_hash   VARCHAR(64) NOT NULL,
    source      VARCHAR(255) NOT NULL,
    embedding   JSONB
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_text_hash ON chunks(text_hash);

CREATE TABLE IF NOT EXISTS chat_history (
    id         SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    role       VARCHAR(10) NOT NULL,
    content    TEXT NOT NULL,
    sources    JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS pii_results (
    id            SERIAL PRIMARY KEY,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    input_preview TEXT,
    result        JSONB NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS sentiment_results (
    id            SERIAL PRIMARY KEY,
    session_id    INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    input_preview TEXT,
    result        JSONB NOT NULL,
    created_at    TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ocr_results (
    id         SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    result     JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS classifier_results (
    id         SERIAL PRIMARY KEY,
    session_id INTEGER NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    criteria   TEXT NOT NULL,
    result     JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);
