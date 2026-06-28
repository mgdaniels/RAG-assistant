"""Tests for retrieval. The embedder is stubbed, so the model is not required."""
from ragassistant.ingest import Chunk
from ragassistant.store import VectorStore
from ragassistant import retrieve as retrieve_mod
from ragassistant.retrieve import retrieve, format_results, RetrievedChunk


class _StubEmbedder:
    def __init__(self, vector):
        self._vector = vector

    def embed_query(self, text):
        return self._vector


def _seed_store(tmp_path):
    store = VectorStore(persist_dir=tmp_path, collection_name="test")
    chunks = [
        Chunk(id="a::0", text="cats are great", source="a.md", title="A", chunk_index=0),
        Chunk(id="b::0", text="dogs are loyal", source="b.md", title="B", chunk_index=0),
    ]
    store.add(chunks, [[1.0, 0.0], [0.0, 1.0]])
    return store


def test_retrieve_returns_typed_results(tmp_path, monkeypatch):
    store = _seed_store(tmp_path)
    monkeypatch.setattr(retrieve_mod, "get_embedder", lambda: _StubEmbedder([0.95, 0.05]))
    results = retrieve("anything", k=1, store=store)
    assert len(results) == 1
    r = results[0]
    assert isinstance(r, RetrievedChunk)
    assert r.source == "a.md"
    assert r.title == "A"
    assert r.score > 0.5
    assert r.citation == "A (a.md)"


def test_retrieve_orders_by_similarity(tmp_path, monkeypatch):
    store = _seed_store(tmp_path)
    monkeypatch.setattr(retrieve_mod, "get_embedder", lambda: _StubEmbedder([0.0, 1.0]))
    results = retrieve("anything", k=2, store=store)
    assert [r.source for r in results] == ["b.md", "a.md"]   # dogs first
    assert results[0].score >= results[1].score


def test_format_results_handles_empty():
    assert "No results" in format_results("q", [])


def test_format_results_shows_scores_and_sources():
    rc = RetrievedChunk(text="hello world", source="a.md", title="A", chunk_index=0, score=0.812)
    out = format_results("q", [rc])
    assert "0.812" in out
    assert "a.md" in out
