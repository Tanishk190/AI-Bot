-- Migration: Add user accounts with roles
-- Run: psql -U postgres -d documind -f migrations/002_add_users.sql

CREATE TABLE IF NOT EXISTS users (
    id            SERIAL PRIMARY KEY,
    email         VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    name          VARCHAR(100) NOT NULL,
    role          VARCHAR(20) NOT NULL DEFAULT 'staff'
                  CHECK (role IN ('admin', 'staff', 'readonly')),
    created_at    TIMESTAMP DEFAULT NOW()
);

-- Link sessions to users
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS user_id INTEGER REFERENCES users(id) ON DELETE CASCADE;
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id);
