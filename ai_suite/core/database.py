"""PostgreSQL database layer for DocuMind."""
import json
import os
from contextlib import contextmanager

import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor, Json


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


# ── Sessions ──────────────────────────────────────────────────────────────────

def get_or_create_session(session_id: str) -> int:
    with get_connection() as conn:
        with conn.cursor() as cur:
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
                (document_id, c["chunk_index"], c["text"], c["text_hash"], c["source"])
                for c in chunks_data
            ]
            cur.executemany(
                "INSERT INTO chunks (document_id, chunk_index, text, text_hash, source) "
                "VALUES (%s, %s, %s, %s, %s)",
                args
            )


def update_chunk_embedding(text_hash: str, embedding: list[float]):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE chunks SET embedding = %s WHERE text_hash = %s AND embedding IS NULL",
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
                    "SELECT c.id, c.chunk_index, c.text, c.text_hash, c.source, c.embedding "
                    "FROM chunks c "
                    "JOIN documents d ON c.document_id = d.id "
                    "WHERE d.session_id = %s AND d.tool = 'chat' AND d.id = ANY(%s) "
                    "ORDER BY d.id, c.chunk_index",
                    (session_db_id, document_ids)
                )
            else:
                cur.execute(
                    "SELECT c.id, c.chunk_index, c.text, c.text_hash, c.source, c.embedding "
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
