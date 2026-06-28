"""Tests for the vector store, using synthetic vectors (no embedding model required)."""
from ragassistant.ingest import Chunk
from ragassistant.store import VectorStore


def _chunk(cid: str, text: str, source: str) -> Chunk:
    idx = int(cid.split("::")[1])
    return Chunk(id=cid, text=text, source=source, title=source, chunk_index=idx)


def test_store_returns_nearest_first(tmp_path):
    store = VectorStore(persist_dir=tmp_path, collection_name="test")
    chunks = [
        _chunk("a::0", "about cats", "a.md"),
        _chunk("b::0", "about dogs", "b.md"),
        _chunk("c::0", "about boats", "c.md"),
    ]
    embeddings = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    store.add(chunks, embeddings)
    assert store.count() == 3

    hits = store.query([0.9, 0.1, 0.0], k=2)
    assert len(hits) == 2
    assert hits[0]["metadata"]["source"] == "a.md"   # nearest
    assert hits[0]["score"] > hits[1]["score"]        # sorted by similarity
    assert hits[0]["text"] == "about cats"            # documents round-trip


def test_store_upsert_is_idempotent(tmp_path):
    store = VectorStore(persist_dir=tmp_path, collection_name="test")
    chunk = _chunk("a::0", "v1", "a.md")
    store.add([chunk], [[1.0, 0.0]])
    store.add([chunk], [[1.0, 0.0]])  # same id again
    assert store.count() == 1          # not duplicated


def test_store_persists_across_instances(tmp_path):
    VectorStore(persist_dir=tmp_path, collection_name="test").add(
        [_chunk("a::0", "hello", "a.md")], [[1.0, 0.0]]
    )
    # A fresh instance pointed at the same directory sees the data.
    reopened = VectorStore(persist_dir=tmp_path, collection_name="test")
    assert reopened.count() == 1
