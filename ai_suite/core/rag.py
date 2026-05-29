"""RAG pipeline for document retrieval and chunking."""
from dataclasses import dataclass
import os
import re
from openai import OpenAI, APIError, APIConnectionError


@dataclass
class Chunk:
    text: str
    source: str
    index: int


_embeddings_client = None
_embeddings_cache = {}
_embedding_model_name = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")


def _get_embeddings_client():
    global _embeddings_client
    if _embeddings_client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY.")
        _embeddings_client = OpenAI(api_key=api_key)
    return _embeddings_client


def _get_embedding(text: str):
    if text in _embeddings_cache:
        return _embeddings_cache[text]
    try:
        client = _get_embeddings_client()
        response = client.embeddings.create(model=_embedding_model_name, input=text)
        embedding = response.data[0].embedding
        _embeddings_cache[text] = embedding
        return embedding
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to OpenAI embeddings API: {str(exc)}") from exc
    except APIError as exc:
        raise RuntimeError(f"OpenAI embeddings API error: {str(exc)}") from exc


def load_documents(uploaded_files: list) -> list[Chunk]:
    chunks = []
    for file_storage in uploaded_files:
        if not file_storage or not file_storage.filename:
            continue
        filename = os.path.basename(file_storage.filename)
        text = _extract_text(file_storage, filename)
        if text:
            chunks.extend(chunk_text(text, filename))
    return chunks


def load_chat_documents(uploaded_files: list) -> list[Chunk]:
    chunks = []
    for file_storage in uploaded_files:
        if not file_storage or not file_storage.filename:
            continue
        filename = os.path.basename(file_storage.filename)
        text = _extract_text(file_storage, filename)
        if text:
            chunks.extend(chunk_text_structured(text, filename))
    return chunks


def chunk_text(text: str, source: str, chunk_size: int = 500, overlap: int = 85) -> list[Chunk]:
    if not text or not text.strip():
        return []

    text = text.strip()
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    if len(paragraphs) == 1:
        paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    if len(paragraphs) == 1:
        paragraphs = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]

    chunks = []
    current = ""
    idx = 1

    for para in paragraphs:
        if len(para) > chunk_size:
            if current:
                chunks.append(Chunk(source=source, index=idx, text=current.strip()))
                idx += 1
            for i in range(0, len(para), chunk_size - overlap):
                piece = para[i: i + chunk_size]
                if piece.strip():
                    chunks.append(Chunk(source=source, index=idx, text=piece.strip()))
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


def chunk_text_structured(
    text: str, source: str, chunk_size: int = 500, overlap: int = 85
) -> list[Chunk]:
    if not text or not text.strip():
        return []

    sections = _split_sections(text)
    chunks = []
    idx = 1

    for heading, body in sections:
        if not body:
            continue

        # Keep table-like blocks atomic
        if _is_table_block(body):
            chunks.append(Chunk(source=source, index=idx, text=_format_chunk(heading, body)))
            idx += 1
            continue

        clauses = _split_clauses(body)
        current = ""

        for clause in clauses:
            clause = clause.strip()
            if not clause:
                continue

            if len(clause) > chunk_size:
                if current:
                    chunks.append(Chunk(source=source, index=idx, text=_format_chunk(heading, current)))
                    idx += 1
                    current = ""
                step = max(chunk_size - overlap, 1)
                for i in range(0, len(clause), step):
                    piece = clause[i: i + chunk_size].strip()
                    if piece:
                        chunks.append(Chunk(source=source, index=idx, text=_format_chunk(heading, piece)))
                        idx += 1
                continue

            if not current:
                current = clause
            elif len(current) + len(clause) + 1 <= chunk_size:
                current = f"{current} {clause}"
            else:
                chunks.append(Chunk(source=source, index=idx, text=_format_chunk(heading, current)))
                idx += 1
                overlap_text = current[-overlap:].strip()
                current = f"{overlap_text} {clause}".strip() if overlap_text else clause

        if current:
            chunks.append(Chunk(source=source, index=idx, text=_format_chunk(heading, current)))
            idx += 1

    return chunks


def _is_table_block(text: str) -> bool:
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    if len(lines) < 3:
        return False
    tab_lines = sum(1 for l in lines if "\t" in l)
    pipe_lines = sum(1 for l in lines if "|" in l)
    if tab_lines >= 2 or pipe_lines >= 2:
        return True
    date_pattern = re.compile(r"\d{1,2}-[A-Za-z]+-\d{4}")
    date_lines = sum(1 for l in lines if date_pattern.search(l))
    if date_lines >= 2:
        return True
    lengths = [len(l) for l in lines]
    avg = sum(lengths) / len(lengths)
    return avg < 120 and (max(lengths) - min(lengths)) < 100


def _format_chunk(heading: str | None, text: str) -> str:
    if heading:
        return f"Section: {heading}\n{text}".strip()
    return text.strip()


def _split_sections(text: str) -> list[tuple[str | None, str]]:
    lines = text.splitlines()
    sections = []
    current_heading = None
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current_lines.append("")
            continue

        if _is_heading(stripped):
            if current_lines and any(part.strip() for part in current_lines):
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = stripped
            current_lines = []
            continue

        current_lines.append(stripped)

    if current_lines and any(part.strip() for part in current_lines):
        sections.append((current_heading, "\n".join(current_lines).strip()))

    if not sections:
        return [(None, text.strip())]

    return sections


def _is_heading(line: str) -> bool:
    if len(line) > 140:
        return False
    heading_patterns = [
        r"^(section|article)\s+[0-9ivxlcdm]+([.-]\d+)*\b.*",
        r"^(schedule|appendix|exhibit)\s+[a-z0-9ivxlcdm]+.*",
        r"^\d+(\.\d+)*\s+[A-Z].{3,}$",
        r"^[A-Z][A-Z0-9 \-]{4,}$",
        r"^[A-Z][A-Za-z0-9 ,\-]{0,80}:$",
    ]
    for pattern in heading_patterns:
        if re.match(pattern, line.strip(), re.IGNORECASE):
            return True
    return False


def _split_clauses(text: str) -> list[str]:
    lines = text.splitlines()
    clauses = []
    current = ""
    clause_marker = re.compile(
        r"^\s*(\([a-z0-9]+\)|[a-z0-9]+\)|[a-z]\.|[0-9]+\.|[0-9]+\.[0-9]+)\s+",
        re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()
        if not stripped:
            if current:
                clauses.append(current.strip())
                current = ""
            continue

        if clause_marker.match(stripped):
            if current:
                clauses.append(current.strip())
            current = stripped
        else:
            current = f"{current} {stripped}".strip() if current else stripped

    if current:
        clauses.append(current.strip())

    if not clauses:
        return [text.strip()]

    refined = []
    for clause in clauses:
        if ";" in clause:
            parts = [p.strip() for p in clause.split(";") if p.strip()]
            refined.extend(parts)
        else:
            refined.append(clause)

    return refined


def retrieve_relevant_chunks(question: str, chunks: list[Chunk], k: int = 3) -> list[Chunk]:
    if not chunks:
        return []

    try:
        import numpy as np
        question_embedding = _get_embedding(question)
        scored = []
        for chunk in chunks:
            chunk_embedding = _get_embedding(chunk.text)
            similarity = np.dot(question_embedding, chunk_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(chunk_embedding)
            )
            # boost if query words appear in chunk
            query_words = set(question.lower().split())
            chunk_words = set(chunk.text.lower().split())
            keyword_boost = 0.1 if query_words & chunk_words else 0
            scored.append((chunk, similarity + keyword_boost))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in scored[:k]]
    except Exception as e:
        print(f"⚠️  Semantic search failed: {e}. Falling back to BM25.")

    return _bm25_retrieval(question, chunks, k)


def _bm25_retrieval(question: str, chunks: list[Chunk], k: int) -> list[Chunk]:
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


def _extract_pdf(file_storage) -> str:
    text = ""
    try:
        import      pdfplumber
        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        with pdfplumber.open(file_storage.stream) as pdf:
            page_texts = []
            for page in pdf.pages:
                page_text = page.extract_text(x_tolerance=1, y_tolerance=1) or ""
                tables = page.extract_tables() or []
                table_blocks = []
                for table in tables:
                    rows = []
                    for row in table:
                        cells = [(cell or "").strip().replace("\n", " ") for cell in row]
                        rows.append("\t".join(cells).strip())
                    if rows:
                        table_blocks.append("\n".join(rows))
                if table_blocks:
                    table_text = "\n\n".join(table_blocks)
                    page_text = f"{page_text}\n\n{table_text}".strip() if page_text else table_text
                if page_text:
                    page_texts.append(page_text)
            text = "\n\n".join(page_texts).strip()
            if text:
                return text
    except ImportError:
        pass
    except Exception as e:
        print(f"Error extracting text with pdfplumber: {e}")

    try:
        from PyPDF2 import PdfReader
        try:
            file_storage.stream.seek(0)
        except Exception:
            pass
        pdf = PdfReader(file_storage)
        for page in pdf.pages:
            text += page.extract_text() or ""
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
            try:
                extracted = page.extract_text(extraction_mode="layout")
            except TypeError:
                extracted = page.extract_text()
            text += extracted or ""
        return text
    except ImportError:
        return "[PDF extraction requires PyPDF2 or pypdf]"
    except Exception as e:
        print(f"Error extracting text with pypdf: {e}")
        return text


def _extract_docx(file_storage) -> str:
    try:
        from docx import Document
        doc = Document(file_storage)
        parts = [para.text for para in doc.paragraphs if para.text.strip()]
        for table in doc.tables:
            rows = []
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                rows.append("\t".join(cells).strip())
            if rows:
                parts.append("TABLE:")
                parts.extend(rows)
        return "\n".join([p for p in parts if p.strip()])
    except ImportError:
        return "[DOCX extraction requires python-docx]"
