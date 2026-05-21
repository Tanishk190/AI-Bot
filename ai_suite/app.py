"""AI Suite - Document Intelligence Web App with OpenAI GPT-4o."""
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
import os

load_dotenv()

try:
    from core.llm import generate_completion, build_rag_prompt
    from core.rag import load_documents, retrieve_relevant_chunks, Chunk
    from core.pii import extract_pii, format_pii_for_display
except ModuleNotFoundError:
    from ai_suite.core.llm import generate_completion, build_rag_prompt
    from ai_suite.core.rag import load_documents, retrieve_relevant_chunks, Chunk
    from ai_suite.core.pii import extract_pii, format_pii_for_display


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024

CHAT_CHUNKS = []

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


@app.route("/")
def index():
    return render_template("index.html", tools=TOOLS)


@app.post("/api/chat/index")
def index_chat_documents():
    """Index documents for RAG."""
    global CHAT_CHUNKS
    
    files = request.files.getlist("documents")
    if not files:
        return jsonify({"error": "Upload at least one document."}), 400
    
    try:
        chunks = load_documents(files)
        if not chunks:
            return jsonify({"error": "No readable text found in uploaded documents."}), 400
        
        CHAT_CHUNKS = chunks
        indexed_files = _group_by_source(chunks)
        
        return jsonify({
            "message": "Documents indexed successfully.",
            "files": indexed_files,
            "total_chunks": len(CHAT_CHUNKS),
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.post("/api/chat/message")
def chat_message():
    """Generate chat response using RAG."""
    payload = request.get_json(silent=True) or {}
    question = (payload.get("message") or "").strip()
    
    if not question:
        return jsonify({"error": "Enter a question first."}), 400
    
    if not CHAT_CHUNKS:
        return jsonify({"error": "Upload and index documents before asking."}), 400
    
    try:
        relevant = retrieve_relevant_chunks(question, CHAT_CHUNKS, k=3)
        if not relevant:
            return jsonify({
                "answer": "I could not find relevant context in the uploaded documents.",
                "sources": [],
            })
        
        context_blocks = [
            f"Source: {chunk.source}, chunk {chunk.index}\n{chunk.text}"
            for chunk in relevant
        ]
        prompt = build_rag_prompt(question, context_blocks)
        
        answer = generate_completion(prompt)
        sources = [{"source": chunk.source, "chunk": chunk.index} for chunk in relevant]
        
        return jsonify({"answer": answer, "sources": sources})
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


def _group_by_source(chunks: list) -> list[dict]:
    """Group chunks by source file."""
    by_source = {}
    for chunk in chunks:
        if chunk.source not in by_source:
            by_source[chunk.source] = 0
        by_source[chunk.source] += 1
    
    return [{"name": source, "chunks": count} for source, count in by_source.items()]


@app.post("/api/pii/extract")
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
        return jsonify({
            "data": pii_data,
            "formatted": format_pii_for_display(pii_data)
        })
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
