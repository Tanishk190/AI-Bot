"""AI Suite - Document Intelligence Web App with OpenAI GPT-4o."""
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request, session, redirect, url_for, g
from functools import wraps
import hashlib
import os
import uuid

load_dotenv()

try:
    from core.llm import generate_completion, generate_chat_completion, build_rag_prompt, parse_llm_json
    from core.rag import (
        load_documents, load_chat_documents, retrieve_relevant_chunks_db,
        embed_session_chunks, Chunk, _text_hash,
    )
    from core.pii import extract_pii, format_pii_for_display
    from core.ocr import extract_ocr_text
    from core.classifier import classify_documents
    from core.database import (
        init_db, get_or_create_session, delete_session,
        insert_document, insert_chunks, get_session_document_count,
        get_session_documents, delete_document,
        save_chat_message, get_chat_history, clear_chat_history,
        save_pii_result, save_sentiment_result,
        save_ocr_result, save_classifier_result,
        create_user, authenticate_user, get_user_by_id,
        list_users, update_user, delete_user, ensure_admin_exists,
        get_assigned_session,
    )
except ModuleNotFoundError:
    from ai_suite.core.llm import generate_completion, generate_chat_completion, build_rag_prompt, parse_llm_json
    from ai_suite.core.rag import (
        load_documents, load_chat_documents, retrieve_relevant_chunks_db,
        embed_session_chunks, Chunk, _text_hash,
    )
    from ai_suite.core.pii import extract_pii, format_pii_for_display
    from ai_suite.core.ocr import extract_ocr_text
    from ai_suite.core.classifier import classify_documents
    from ai_suite.core.database import (
        init_db, get_or_create_session, delete_session,
        insert_document, insert_chunks, get_session_document_count,
        get_session_documents, delete_document,
        save_chat_message, get_chat_history, clear_chat_history,
        save_pii_result, save_sentiment_result,
        save_ocr_result, save_classifier_result,
        create_user, authenticate_user, get_user_by_id,
        list_users, update_user, delete_user, ensure_admin_exists,
        get_assigned_session,
    )


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
app.secret_key = os.getenv("SECRET_KEY", uuid.uuid4().hex)


# ── Auth helpers ──────────────────────────────────────────────────────────────

def _get_current_user() -> dict | None:
    user_id = session.get("user_id")
    if not user_id:
        return None
    if not hasattr(g, "_current_user"):
        g._current_user = get_user_by_id(user_id)
    return g._current_user


def _get_session_id() -> str:
    if "sid" not in session:
        session["sid"] = uuid.uuid4().hex
    return session["sid"]


def _get_session_db_id() -> int:
    user = _get_current_user()
    if not user:
        return get_or_create_session(_get_session_id())
    # Read-only users view their assigned user's session
    if user["role"] == "readonly" and user.get("assigned_to"):
        assigned_session = get_assigned_session(user["id"])
        if assigned_session:
            return assigned_session
    return get_or_create_session(_get_session_id(), user_id=user["id"])


def require_role(*roles):
    """Decorator to restrict endpoints to specific roles."""
    def decorator(f):
        @wraps(f)
        def wrapper(*args, **kwargs):
            user = _get_current_user()
            if not user:
                return jsonify({"error": "Authentication required."}), 401
            if user["role"] not in roles:
                return jsonify({"error": "You don't have permission for this action."}), 403
            return f(*args, **kwargs)
        return wrapper
    return decorator


TOOLS = [
    {
        "id": "chat",
        "title": "AI Chat",
        "badge": "RAG Enabled",
        "icon": "MSG",
    },
    {
        "id": "ocr",
        "title": "OCR",
        "badge": "Image to Text",
        "icon": "OCR",
    },
    {
        "id": "pii",
        "title": "PII Extractor",
        "badge": "LLM Powered",
        "icon": "ID",
    },
    {
        "id": "sentiment",
        "title": "Sentiment",
        "page_title": "Sentiment Analysis",
        "badge": "LLM Powered",
        "icon": "SEN",
    },
    {
        "id": "classify",
        "title": "Classifier",
        "page_title": "Document Classifier",
        "badge": "RAG + LLM",
        "icon": "TAG",
    },
]


# ── Auth routes ───────────────────────────────────────────────────────────────

@app.before_request
def require_login():
    allowed = {"login", "static"}
    if request.endpoint in allowed:
        return
    if not session.get("user_id"):
        return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("index"))
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        password = request.form.get("password") or ""
        user = authenticate_user(email, password)
        if user:
            session["user_id"] = user["id"]
            session["sid"] = uuid.uuid4().hex
            return redirect(url_for("index"))
        return render_template("login.html", error="Invalid email or password.")
    return render_template("login.html", error=None)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
def index():
    user = _get_current_user()
    return render_template("index.html", tools=TOOLS, user=user)


# ── User info API ─────────────────────────────────────────────────────────────

@app.get("/api/me")
def current_user_info():
    user = _get_current_user()
    if not user:
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify({
        "id": user["id"],
        "email": user["email"],
        "name": user["name"],
        "role": user["role"],
    })


# ── Admin routes ──────────────────────────────────────────────────────────────

@app.route("/admin")
def admin_page():
    user = _get_current_user()
    if not user or user["role"] != "admin":
        return redirect(url_for("index"))
    users = list_users()
    for u in users:
        u["created_at"] = u["created_at"].isoformat() if u.get("created_at") else None
    return render_template("admin.html", users=users, current_user=user)


@app.post("/api/admin/users")
@require_role("admin")
def api_create_user():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip()
    password = (payload.get("password") or "").strip()
    name = (payload.get("name") or "").strip()
    role = (payload.get("role") or "staff").strip()
    assigned_to = payload.get("assigned_to")  # user ID or None

    if not email or not password or not name:
        return jsonify({"error": "Email, password, and name are required."}), 400
    if role not in ("admin", "staff", "readonly"):
        return jsonify({"error": "Role must be admin, staff, or readonly."}), 400

    try:
        user_id = create_user(email, password, name, role)
        if role == "readonly" and assigned_to:
            update_user(user_id, assigned_to=int(assigned_to))
        return jsonify({"id": user_id, "message": f"User {email} created."})
    except Exception as exc:
        if "duplicate key" in str(exc).lower() or "unique" in str(exc).lower():
            return jsonify({"error": "A user with this email already exists."}), 409
        return jsonify({"error": str(exc)}), 500


@app.put("/api/admin/users/<int:user_id>")
@require_role("admin")
def api_update_user(user_id):
    payload = request.get_json(silent=True) or {}
    name = payload.get("name")
    role = payload.get("role")
    password = payload.get("password") or None
    assigned_to = payload.get("assigned_to", -1)  # -1 = not provided

    if role and role not in ("admin", "staff", "readonly"):
        return jsonify({"error": "Role must be admin, staff, or readonly."}), 400

    if assigned_to != -1:
        assigned_to = int(assigned_to) if assigned_to else None
    updated = update_user(user_id, name=name, role=role, password=password, assigned_to=assigned_to)
    if not updated:
        return jsonify({"error": "User not found."}), 404
    return jsonify({"message": "User updated."})


@app.delete("/api/admin/users/<int:user_id>")
@require_role("admin")
def api_delete_user(user_id):
    user = _get_current_user()
    if user["id"] == user_id:
        return jsonify({"error": "You cannot delete yourself."}), 400
    deleted = delete_user(user_id)
    if not deleted:
        return jsonify({"error": "User not found."}), 404
    return jsonify({"message": "User deleted."})


# ── Chat routes ───────────────────────────────────────────────────────────────

@app.post("/api/chat/index")
@require_role("admin", "staff")
def index_chat_documents():
    """Index documents for RAG."""
    files = request.files.getlist("documents")
    if not files:
        return jsonify({"error": "Upload at least one document."}), 400

    try:
        session_db_id = _get_session_db_id()
        chunks = load_chat_documents(files)
        if not chunks:
            return jsonify({"error": "No readable text found in uploaded documents."}), 400

        by_source: dict[str, list[Chunk]] = {}
        for chunk in chunks:
            by_source.setdefault(chunk.source, []).append(chunk)

        warnings = []
        all_sources = {os.path.basename(f.filename) for f in files if f and f.filename}

        for source, source_chunks in by_source.items():
            raw_text = " ".join(c.text for c in source_chunks)
            content_hash = hashlib.sha256(raw_text.encode()).hexdigest()
            doc_id = insert_document(session_db_id, source, raw_text, content_hash, tool="chat")
            insert_chunks(doc_id, [
                {
                    "chunk_index": c.index,
                    "text": c.text,
                    "text_hash": _text_hash(c.text),
                    "source": c.source,
                }
                for c in source_chunks
            ])

        empty_files = all_sources - set(by_source.keys())
        for name in empty_files:
            warnings.append(f"{name}: no text extracted. If this is a scanned PDF, run OCR first.")

        embed_session_chunks(session_db_id)

        indexed_files = [
            {"name": source, "chunks": len(source_chunks)}
            for source, source_chunks in by_source.items()
        ]

        return jsonify({
            "message": "Documents indexed successfully.",
            "files": indexed_files,
            "total_chunks": len(chunks),
            "warnings": warnings,
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.get("/api/chat/documents")
def list_chat_documents():
    """List all indexed documents for the current session."""
    session_db_id = _get_session_db_id()
    docs = get_session_documents(session_db_id)
    for doc in docs:
        doc["uploaded_at"] = doc["uploaded_at"].isoformat() if doc.get("uploaded_at") else None
    return jsonify({"documents": docs})


@app.delete("/api/chat/documents/<int:doc_id>")
@require_role("admin", "staff")
def remove_chat_document(doc_id):
    """Delete an indexed document and its chunks."""
    session_db_id = _get_session_db_id()
    deleted = delete_document(doc_id, session_db_id)
    if not deleted:
        return jsonify({"error": "Document not found."}), 404
    return jsonify({"message": "Document deleted."})


@app.get("/api/chat/history")
def chat_history():
    """Return chat history for the current session."""
    session_db_id = _get_session_db_id()
    messages = get_chat_history(session_db_id)
    for msg in messages:
        msg["created_at"] = msg["created_at"].isoformat() if msg.get("created_at") else None
    return jsonify({"messages": messages})


@app.delete("/api/chat/history")
@require_role("admin", "staff")
def clear_history():
    """Clear chat history for the current session."""
    session_db_id = _get_session_db_id()
    clear_chat_history(session_db_id)
    return jsonify({"message": "Chat history cleared."})


CHAT_CONTEXT_WINDOW = 6


@app.post("/api/chat/message")
@require_role("admin", "staff")
def chat_message():
    """Generate chat response using RAG with multi-turn context."""
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    document_ids = payload.get("document_ids")

    if not question:
        return jsonify({"error": "Enter a question first."}), 400

    session_db_id = _get_session_db_id()
    doc_count = get_session_document_count(session_db_id)
    if doc_count == 0:
        return jsonify({"error": "Upload and index documents before asking."}), 400

    try:
        save_chat_message(session_db_id, "user", question)

        relevant = retrieve_relevant_chunks_db(
            question, session_db_id, k=3, document_ids=document_ids
        )
        if not relevant:
            answer = "I could not find relevant context in the uploaded documents."
            save_chat_message(session_db_id, "assistant", answer)
            return jsonify({"answer": answer, "sources": []})

        context_blocks = [
            f"Source: {chunk.source}, chunk {chunk.index}\n{chunk.text}"
            for chunk in relevant
        ]
        system_p, user_p = build_rag_prompt(question, context_blocks)

        history = get_chat_history(session_db_id)
        prior = history[:-1][-CHAT_CONTEXT_WINDOW:]

        messages = []
        for msg in prior:
            role = "assistant" if msg["role"] == "assistant" else "user"
            messages.append({"role": role, "content": msg["content"]})
        messages.append({"role": "user", "content": user_p})

        answer = generate_chat_completion(messages, system_prompt=system_p)
        sources = [{"source": chunk.source, "chunk": chunk.index} for chunk in relevant]

        save_chat_message(session_db_id, "assistant", answer, sources)

        return jsonify({"answer": answer, "sources": sources})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


# ── PII, OCR, Sentiment, Classifier routes ────────────────────────────────────

@app.post("/api/pii/extract")
@require_role("admin", "staff")
def pii_extract():
    """Extract PII using custom system prompt."""
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    system_prompt = (payload.get("system_prompt") or "").strip()

    if not text:
        return jsonify({"error": "Input text is required."}), 400
    if not system_prompt:
        return jsonify({"error": "System prompt is required."}), 400

    try:
        pii_data = extract_pii(text, system_prompt)
        session_db_id = _get_session_db_id()
        save_pii_result(session_db_id, text, pii_data)
        return jsonify({"data": pii_data, "formatted": format_pii_for_display(pii_data)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


@app.post("/api/ocr/extract")
@require_role("admin", "staff")
def ocr_extract():
    """Extract text from uploaded images using OCR."""
    images = request.files.getlist("images")
    if not images:
        return jsonify({"error": "Upload at least one image."}), 400

    try:
        results = extract_ocr_text(images)
        if not results:
            return jsonify({"error": "No readable text found in uploaded images."}), 400

        errors = [item.get("error") for item in results if item.get("error")]
        combined_text = "\n\n".join(
            f"{item['filename']}\n{item['text']}".strip() for item in results if item.get("text")
        ).strip()

        if not combined_text:
            if errors:
                return jsonify({"error": f"OCR failed: {errors[0]}"}), 503
            return jsonify({"error": "No readable text found in uploaded images."}), 400

        response_data = {"text": combined_text, "results": results, "errors": errors}
        session_db_id = _get_session_db_id()
        save_ocr_result(session_db_id, response_data)
        return jsonify(response_data)
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"OCR failed: {str(exc)}"}), 500


SENTIMENT_CHAR_LIMIT = 3000


@app.post("/api/sentiment/analyze")
@require_role("admin", "staff")
def sentiment_analyze():
    """Analyze sentiment of text or documents."""
    text = (request.form.get("text") or "").strip()
    documents = request.files.getlist("documents")

    if not text and not documents:
        return jsonify({"error": "Provide text or upload documents"}), 400
    if text and documents:
        return jsonify({"error": "Provide either text or documents, not both."}), 400

    try:
        combined_text = ""
        source = None

        if documents:
            chunks = load_documents(documents)
            if chunks:
                combined_text = " ".join(chunk.text for chunk in chunks)
                source = "document"
            else:
                return jsonify({"error": "No readable text found. If scanned PDF, run OCR first."}), 400

        if text:
            combined_text = text
            source = "text"

        normalized_text = " ".join(combined_text.split())
        original_length = len(normalized_text)
        truncated = original_length > SENTIMENT_CHAR_LIMIT
        normalized_text = normalized_text[:SENTIMENT_CHAR_LIMIT]

        system_prompt = (
            "You are a sentiment analysis assistant for legal document review. Analyze the sentiment of the given text. "
            "Return ONLY a valid JSON object with these exact fields:\n"
            "{\n"
            "  \"sentiment\": \"Positive\" | \"Negative\" | \"Neutral\",\n"
            "  \"confidence\": integer 0-100,\n"
            "  \"tone\": single word descriptor,\n"
            "  \"reasoning\": 1-2 sentence explanation\n"
            "}\n"
            "This is for legal eDiscovery context — flag hostile or adversarial tone especially."
        )

        response = generate_completion(normalized_text, system_prompt=system_prompt)
        sentiment_data = parse_llm_json(response)
        if not isinstance(sentiment_data, dict):
            raise ValueError("LLM response must be a JSON object.")

        result = {**sentiment_data, "source": source, "char_count": len(normalized_text)}
        if truncated:
            result["warning"] = f"Text truncated from {original_length} to {SENTIMENT_CHAR_LIMIT} chars."

        session_db_id = _get_session_db_id()
        save_sentiment_result(session_db_id, normalized_text, result)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Sentiment analysis failed: {str(exc)}"}), 500


@app.post("/api/classify/documents")
@require_role("admin", "staff")
def classify_docs():
    """Classify documents against user-defined criteria."""
    documents = request.files.getlist("documents")
    criteria = (request.form.get("criteria") or "").strip()
    low_threshold = float(request.form.get("low_threshold", 0.25))
    high_threshold = float(request.form.get("high_threshold", 0.65))

    if not documents or not any(d for d in documents if d and d.filename):
        return jsonify({"error": "Upload at least one document."}), 400
    if not criteria:
        return jsonify({"error": "Classification criteria is required."}), 400

    try:
        results = classify_documents(
            documents, criteria,
            low_threshold=low_threshold, high_threshold=high_threshold,
        )
        if not results:
            return jsonify({"error": "No readable text found in uploaded documents."}), 400

        session_db_id = _get_session_db_id()
        save_classifier_result(session_db_id, criteria, results)
        return jsonify({"criteria": criteria, "results": results, "total_documents": len(results)})
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Classification failed: {str(exc)}"}), 500


def _run_startup():
    """Initialize the database schema and seed the default admin.

    Called at import time so it runs under gunicorn (where the ``__main__``
    block below never executes). Guarded so a transient DB hiccup at boot
    doesn't crash the worker — schema creation is idempotent and retried on
    the next start.
    """
    try:
        init_db()
        ensure_admin_exists()
    except Exception as exc:  # noqa: BLE001
        print(f"Startup DB initialization failed: {exc}")


_run_startup()


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
