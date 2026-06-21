"""Tests for document chunking — chunk creation, sequential indexing, and the
heading/table detection used by the structured chat chunker."""
from core.rag import (
    Chunk, chunk_text, chunk_text_structured, _is_heading, _is_table_block,
)


def test_empty_text_returns_no_chunks():
    assert chunk_text("", "doc.txt") == []
    assert chunk_text("   \n  ", "doc.txt") == []


def test_chunk_indices_are_sequential():
    text = "alpha beta gamma. delta epsilon zeta. eta theta iota kappa lambda."
    chunks = chunk_text(text, "doc.txt", chunk_size=20, overlap=5)
    assert len(chunks) >= 2
    assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))


def test_chunk_carries_source_and_default_page():
    chunks = chunk_text("some readable content here for the chunker", "report.pdf")
    assert chunks
    assert all(c.source == "report.pdf" for c in chunks)
    assert all(c.page is None for c in chunks)  # page is set later, per PDF page


def test_chunk_dataclass_defaults():
    c = Chunk(text="t", source="s", index=1)
    assert c.page is None


def test_structured_chunker_prefixes_heading():
    text = "SCOPE OF AUDIT\nThe scope covers the financial statements.\n\n" \
           "CONFIDENTIALITY\nAll information is kept strictly confidential."
    chunks = chunk_text_structured(text, "engagement.pdf")
    assert chunks
    assert [c.index for c in chunks] == list(range(1, len(chunks) + 1))
    assert any("Section:" in c.text for c in chunks)


def test_is_heading_detection():
    assert _is_heading("SCOPE OF AUDIT") is True
    assert _is_heading("Confidentiality:") is True
    assert _is_heading("This is an ordinary sentence of prose.") is False


def test_is_table_block_detection():
    table = "Name\tAge\tCity\nAlice\t30\tNYC\nBob\t25\tLA"
    assert _is_table_block(table) is True
    prose = "This is a single ordinary line with no tabular structure."
    assert _is_table_block(prose) is False
