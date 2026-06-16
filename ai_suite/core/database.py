"""PostgreSQL database layer for DocuMind."""
import json
import os
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json
from werkzeug.security import generate_password_hash, check_password_hash


_pool = None


def get_pool():
    global _pool
    if _pool is None:
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise RuntimeError("Missing DATABASE_URL in .env")
        _pool = pool.SimpleConnectionPool(1, 5, database_url)
    return _pool


@contextmanager
def get_connection():
    conn = get_pool().getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        get_pool().putconn(conn)


def init_db():
    schema_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "schema.sql")
    with open(schema_path, "r") as f:
        sql = f.read()
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql)


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(email: str, password: str, name: str, role: str = "staff") -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, password_hash, name, role) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (email.lower().strip(), generate_password_hash(password), name.strip(), role)
            )
            return cur.fetchone()[0]


def authenticate_user(email: str, password: str) -> dict | None:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("SELECT * FROM users WHERE email = %s", (email.lower().strip(),))
            user = cur.fetchone()
            if user and check_password_hash(user["password_hash"], password):
                return dict(user)
            return None


def get_user_by_id(user_id: int) -> dict | None:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT id, email, name, role, assigned_to, created_at FROM users WHERE id = %s",
                (user_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None


def list_users() -> list[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT u.id, u.email, u.name, u.role, u.assigned_to, u.created_at, "
                "a.name as assigned_to_name "
                "FROM users u LEFT JOIN users a ON u.assigned_to = a.id "
                "ORDER BY u.created_at"
            )
            return [dict(row) for row in cur.fetchall()]


def update_user(user_id: int, name: str | None = None, role: str | None = None,
                password: str | None = None, assigned_to: int | None = -1) -> bool:
    fields = []
    values = []
    if name is not None:
        fields.append("name = %s")
        values.append(name.strip())
    if role is not None:
        fields.append("role = %s")
        values.append(role)
    if password is not None:
        fields.append("password_hash = %s")
        values.append(generate_password_hash(password))
    if assigned_to != -1:  # -1 = not provided, None = clear assignment
        fields.append("assigned_to = %s")
        values.append(assigned_to)
    if not fields:
        return False
    values.append(user_id)
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE users SET {', '.join(fields)} WHERE id = %s", values)
            return cur.rowcount > 0


def get_assigned_session(user_id: int) -> int | None:
    """For read-only users: get the session of the user they're assigned to view."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT s.id FROM sessions s "
                "JOIN users u ON s.user_id = u.id "
                "WHERE u.id = (SELECT assigned_to FROM users WHERE id = %s) "
                "LIMIT 1",
                (user_id,)
            )
            row = cur.fetchone()
            return row[0] if row else None


def delete_user(user_id: int) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM users WHERE id = %s RETURNING id", (user_id,))
            return cur.fetchone() is not None


def get_user_count() -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM users")
            return cur.fetchone()[0]


def ensure_admin_exists():
    """Create default admin from env vars if no users exist."""
    if get_user_count() > 0:
        return
    email = os.getenv("ADMIN_EMAIL", "admin@documind.local")
    password = os.getenv("ADMIN_PASSWORD", "admin")
    name = os.getenv("ADMIN_NAME", "Admin")
    create_user(email, password, name, role="admin")
    print(f"Default admin created: {email}")


# ── Sessions ──────────────────────────────────────────────────────────────────

def get_or_create_session(session_id: str, user_id: int | None = None) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            if user_id is not None:
                # Check if user already has a session
                cur.execute("SELECT id FROM sessions WHERE user_id = %s LIMIT 1", (user_id,))
                row = cur.fetchone()
                if row:
                    # Update session_id to current one
                    cur.execute(
                        "UPDATE sessions SET session_id = %s WHERE id = %s",
                        (session_id, row[0])
                    )
                    return row[0]
                # Create new session for user
                cur.execute(
                    "INSERT INTO sessions (session_id, user_id) VALUES (%s, %s) RETURNING id",
                    (session_id, user_id)
                )
                return cur.fetchone()[0]
            # Fallback: no user (should not happen with auth enabled)
            cur.execute(
                "INSERT INTO sessions (session_id) VALUES (%s) "
                "ON CONFLICT (session_id) DO UPDATE SET session_id = EXCLUDED.session_id "
                "RETURNING id",
                (session_id,)
            )
            return cur.fetchone()[0]


def delete_session(session_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))


# ── Documents & Chunks ────────────────────────────────────────────────────────

def insert_document(session_db_id: int, filename: str, raw_text: str,
                    content_hash: str, tool: str = "chat") -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO documents (session_id, filename, raw_text, content_hash, tool) "
                "VALUES (%s, %s, %s, %s, %s) RETURNING id",
                (session_db_id, filename, raw_text, content_hash, tool)
            )
            return cur.fetchone()[0]


def insert_chunks(document_id: int, chunks_data: list[dict]):
    if not chunks_data:
        return
    with get_connection() as conn:
        with conn.cursor() as cur:
            args = [
                (document_id, c["chunk_index"], c["text"], c["text_hash"], c["source"], c.get("page"))
                for c in chunks_data
            ]
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_index, text, text_hash, source, page) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                args
            )


def update_chunk_embedding(text_hash: str, embedding: list[float]):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chunks SET embedding = %s WHERE text_hash = %s",
                (Json(embedding), text_hash)
            )


def get_embedding_by_hash(text_hash: str) -> list[float] | None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT embedding FROM chunks WHERE text_hash = %s AND embedding IS NOT NULL LIMIT 1",
                (text_hash,)
            )
            row = cur.fetchone()
            if row and row[0]:
                return row[0] if isinstance(row[0], list) else json.loads(row[0])
            return None


def get_chunks_by_session(session_db_id: int, document_ids: list[int] | None = None) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            if document_ids:
                cur.execute(
                    "SELECT c.id, c.chunk_index, c.text, c.text_hash, c.source, c.page, c.embedding "
                    "FROM chunks c "
                    "JOIN documents d ON c.document_id = d.id "
                    "WHERE d.session_id = %s AND d.tool = 'chat' AND d.id = ANY(%s) "
                    "ORDER BY d.id, c.chunk_index",
                    (session_db_id, document_ids)
                )
            else:
                cur.execute(
                    "SELECT c.id, c.chunk_index, c.text, c.text_hash, c.source, c.page, c.embedding "
                    "FROM chunks c "
                    "JOIN documents d ON c.document_id = d.id "
                    "WHERE d.session_id = %s AND d.tool = 'chat' "
                    "ORDER BY d.id, c.chunk_index",
                    (session_db_id,)
                )
            return [dict(row) for row in cur.fetchall()]


def get_session_documents(session_db_id: int) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT d.id, d.filename, d.uploaded_at, "
                "(SELECT count(*) FROM chunks c WHERE c.document_id = d.id) as chunk_count "
                "FROM documents d "
                "WHERE d.session_id = %s AND d.tool = 'chat' "
                "ORDER BY d.uploaded_at DESC",
                (session_db_id,)
            )
            return [dict(row) for row in cur.fetchall()]


def delete_document(doc_id: int, session_db_id: int) -> bool:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM documents WHERE id = %s AND session_id = %s RETURNING id",
                (doc_id, session_db_id)
            )
            return cur.fetchone() is not None


def clear_chat_history(session_db_id: int):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM chat_history WHERE session_id = %s", (session_db_id,))


def get_session_document_count(session_db_id: int) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM documents WHERE session_id = %s AND tool = 'chat'",
                (session_db_id,)
            )
            return cur.fetchone()[0]


# ── Chat History ──────────────────────────────────────────────────────────────

def save_chat_message(session_db_id: int, role: str, content: str, sources: list | None = None):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_history (session_id, role, content, sources) "
                "VALUES (%s, %s, %s, %s)",
                (session_db_id, role, content, Json(sources) if sources else None)
            )


def get_chat_history(session_db_id: int) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute(
                "SELECT role, content, sources, created_at FROM chat_history "
                "WHERE session_id = %s ORDER BY created_at",
                (session_db_id,)
            )
            return [dict(row) for row in cur.fetchall()]


# ── Analysis Results ──────────────────────────────────────────────────────────

def save_pii_result(session_db_id: int, input_preview: str, result: dict):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO pii_results (session_id, input_preview, result) VALUES (%s, %s, %s)",
                (session_db_id, input_preview[:500], Json(result))
            )


def save_sentiment_result(session_db_id: int, input_preview: str, result: dict):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO sentiment_results (session_id, input_preview, result) VALUES (%s, %s, %s)",
                (session_db_id, input_preview[:500], Json(result))
            )


def save_ocr_result(session_db_id: int, result: dict):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO ocr_results (session_id, result) VALUES (%s, %s)",
                (session_db_id, Json(result))
            )


def save_classifier_result(session_db_id: int, criteria: str, result: dict):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO classifier_results (session_id, criteria, result) VALUES (%s, %s, %s)",
                (session_db_id, criteria, Json(result))
            )
