"""RAG pipeline for document retrieval and chunking."""
from dataclasses import dataclass
import os
import json


@dataclass
class Chunk:
    """Represents a document chunk."""
    text: str
    source: str
    index: int


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
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    
    chunks = []
    current = ""
    idx = 1

    for para in paragraphs:
        if len(current) + len(para) <= chunk_size:
            current += " " + para
        else:
            if current:
                chunks.append(Chunk(source=source, index=idx, text=current.strip()))
                idx += 1
                # Keep overlap from end of last chunk
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
    Simple BM25-style retrieval: return chunks with highest word overlap.
    For production, use semantic search (sentence-transformers + ChromaDB).
    """
    if not chunks:
        return []
    
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
    try:
        from PyPDF2 import PdfReader
        pdf = PdfReader(file_storage)
        text = ""
        for page in pdf.pages:
            text += page.extract_text()
        return text
    except ImportError:
        return "[PDF extraction requires PyPDF2]"


def _extract_docx(file_storage) -> str:
    """Extract text from DOCX."""
    try:
        from docx import Document
        doc = Document(file_storage)
        return "\n".join([para.text for para in doc.paragraphs])
    except ImportError:
        return "[DOCX extraction requires python-docx]"
