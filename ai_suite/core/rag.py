"""RAG pipeline for document retrieval and chunking."""
from dataclasses import dataclass
import os
import json
from openai import OpenAI, APIError, APIConnectionError


@dataclass
class Chunk:
    """Represents a document chunk."""
    text: str
    source: str
    index: int


# Lazy load embeddings client only when needed
_embeddings_client = None
_embeddings_cache = {}
_embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

def _get_embeddings_client():
    """Lazy load OpenAI client for embeddings."""
    global _embeddings_client
    if _embeddings_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY. Set it to enable embeddings.")
        _embeddings_client = OpenAI(api_key=api_key)
    return _embeddings_client


def _get_embedding(text: str):
    """Get embedding for text with caching."""
    if text in _embeddings_cache:
        return _embeddings_cache[text]

    try:
        client = _get_embeddings_client()
        response = client.embeddings.create(
            model=_embedding_model_name,
            input=text,
        )
        embedding = response.data[0].embedding
        _embeddings_cache[text] = embedding
        return embedding
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to OpenAI embeddings API: {str(exc)}") from exc
    except APIError as exc:
        raise RuntimeError(f"OpenAI embeddings API error: {str(exc)}") from exc


def load_documents(uploaded_files: list) -> list[Chunk]:
    """
    Load and parse documents from uploaded files.
    Supports: TXT, PDF, DOCX
    """
    chunks = []
    
    for file_storage in uploaded_files:
        if not file_storage or not file_storage.filename:
            continue
            
        filename = os.path.basename(file_storage.filename)
        text = _extract_text(file_storage, filename)
        
        if text:
            file_chunks = chunk_text(text, filename)
            chunks.extend(file_chunks)
    
    return chunks


def chunk_text(text: str, source: str, chunk_size: int = 512, overlap: int = 100) -> list[Chunk]:
    """
    Split text into overlapping chunks using multiple strategies.
    First tries paragraph breaks, then sentence breaks, then character breaks.
    """
    if not text or not text.strip():
        return []
    
    text = text.strip()
    
    # Try splitting by double newlines (paragraphs)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    # If only 1 "paragraph", try splitting by single newlines
    if len(paragraphs) == 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]
    
    # If still not enough, try sentence breaks
    if len(paragraphs) == 1:
        import re
        paragraphs = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    
    chunks = []
    current = ""
    idx = 1
    
    for para in paragraphs:
        # If single paragraph is too large, split it
        if len(para) > chunk_size:
            if current:
                chunks.append(Chunk(source=source, index=idx, text=current.strip()))
                idx += 1
            # Split large paragraph into smaller pieces
            for i in range(0, len(para), chunk_size - overlap):
                chunk_text = para[i : i + chunk_size]
                if chunk_text.strip():
                    chunks.append(Chunk(source=source, index=idx, text=chunk_text.strip()))
                    idx += 1
            current = ""
        elif len(current) + len(para) <= chunk_size:
            current += " " + para
        else:
            if current:
                chunks.append(Chunk(source=source, index=idx, text=current.strip()))
                idx += 1
                current = current[-overlap:] + " " + para
            else:
                current = para
    
    if current:
        chunks.append(Chunk(source=source, index=idx, text=current.strip()))
    
    return chunks


def retrieve_relevant_chunks(
    question: str, chunks: list[Chunk], k: int = 3
) -> list[Chunk]:
    """
    Retrieve relevant chunks using semantic similarity (preferred) or BM25 fallback.
    
    Semantic search understands meaning, not just keywords.
    Examples: "author" matches "writer", "expensive" matches "costly"
    """
    if not chunks:
        return []
    
    # Try semantic search first
    try:
        import numpy as np

        question_embedding = _get_embedding(question)
        scored = []
        for chunk in chunks:
            chunk_embedding = _get_embedding(chunk.text)
            # Cosine similarity
            similarity = np.dot(question_embedding, chunk_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
            )
            scored.append((chunk, similarity))

        if scored:
            scored.sort(key=lambda x: x[1], reverse=True)
            return [chunk for chunk, _ in scored[:k]]
    except Exception as e:
        print(f"⚠️  Semantic search failed: {e}. Falling back to BM25.")
    
    # Fallback to BM25 (simple word matching)
    return _bm25_retrieval(question, chunks, k)


def _bm25_retrieval(question: str, chunks: list[Chunk], k: int) -> list[Chunk]:
    """Fallback BM25-style retrieval: word overlap."""
    query_words = set(question.lower().split())
    scored = []
    
    for chunk in chunks:
        chunk_words = set(chunk.text.lower().split())
        overlap = len(query_words & chunk_words)
        if overlap > 0:
            scored.append((chunk, overlap))
    
    scored.sort(key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in scored[:k]]


def _extract_text(file_storage, filename: str) -> str:
    """Extract text from uploaded file."""
    try:
        if filename.lower().endswith(".txt"):
            return file_storage.read().decode("utf-8", errors="ignore")
        elif filename.lower().endswith(".pdf"):
            return _extract_pdf(file_storage)
        elif filename.lower().endswith(".docx"):
            return _extract_docx(file_storage)
    except Exception as e:
        print(f"Error extracting text from {filename}: {e}")
        return ""
    
    return ""


def _extract_pdf(file_storage) -> str:
    """Extract text from PDF."""
    text = ""
    try:
        from PyPDF2 import PdfReader
        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        pdf = PdfReader(file_storage)
        for page in pdf.pages:
            extracted = page.extract_text() or ""
            text += extracted
        if text.strip():
            return text
    except ImportError:
        pass
    except Exception as e:
        print(f"Error extracting text with PyPDF2: {e}")

    try:
        from pypdf import PdfReader
        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        try:
            pdf = PdfReader(file_storage, strict=False)
        except TypeError:
            pdf = PdfReader(file_storage)
        text = ""
        for page in pdf.pages:
            extracted = page.extract_text() or ""
            text += extracted
        return text
    except ImportError:
        return "[PDF extraction requires PyPDF2 or pypdf]"
    except Exception as e:
        print(f"Error extracting text with pypdf: {e}")
        return text


def _extract_docx(file_storage) -> str:
    """Extract text from DOCX."""
    try:
        from docx import Document
        doc = Document(file_storage)
        return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        return "[DOCX extraction requires python-docx]"
