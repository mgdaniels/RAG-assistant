"""Tests for document loading and chunking."""
from collections import Counter

from ragassistant.ingest import split_text, chunk_documents, load_documents


def test_short_text_is_one_chunk():
    assert split_text("just a little text", 800, 150) == ["just a little text"]


def test_empty_text_yields_no_chunks():
    assert split_text("   \n\n  ", 800, 150) == []


def test_chunks_respect_size_budget():
    # 20 paragraphs of ~250 chars each, well above one chunk.
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 50 for i in range(20))
    chunks = split_text(text, chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 1
    # Content fits chunk_size; the overlap prefix may push a chunk slightly over.
    for c in chunks:
        assert len(c) <= 400 + 80


def test_overlap_creates_continuity():
    text = " ".join(f"sentence{i}." for i in range(200))
    chunks = split_text(text, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    shared = [bool(set(a.split()) & set(b.split())) for a, b in zip(chunks, chunks[1:])]
    assert any(shared), "consecutive chunks should share words via the overlap"


def test_no_content_lost_on_clean_boundaries():
    text = ". ".join(f"fact number {i} is unique" for i in range(30)) + "."
    chunks = split_text(text, chunk_size=120, chunk_overlap=20)
    joined = " ".join(chunks)
    for i in range(30):
        assert f"fact number {i} is unique" in joined


def test_overlap_must_be_smaller_than_size():
    try:
        split_text("x" * 100, chunk_size=50, chunk_overlap=50)
    except ValueError:
        return
    raise AssertionError("expected ValueError when overlap >= chunk_size")


def test_corpus_chunks_have_clean_metadata():
    chunks = chunk_documents()  # uses the real Northwind corpus
    assert len(chunks) >= len(load_documents())
    assert len({c.id for c in chunks}) == len(chunks)        # ids are unique
    for c in chunks:
        assert c.source.endswith(".md")
        assert c.title
        assert c.metadata["chunk_index"] == c.chunk_index
    # At least one document is large enough to split into multiple chunks.
    assert max(Counter(c.source for c in chunks).values()) >= 2
