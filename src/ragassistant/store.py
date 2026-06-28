"""Vector store backed by a persistent Chroma collection.

All Chroma-specific code lives in this module, behind a small
``add`` / ``query`` / ``count`` interface. Keeping the backend isolated here
means it can be replaced (for example with a FAISS index) without changing the
retrieval or generation code.
"""
from __future__ import annotations

from pathlib import Path

from .config import settings
from .ingest import Chunk


class VectorStore:
    """A persistent Chroma collection of chunk embeddings."""

    def __init__(self, persist_dir: Path | None = None, collection_name: str | None = None):
        # Imported here to keep Chroma out of lightweight code paths such as
        # `rag --help`.
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        persist_dir = Path(persist_dir or settings.chroma_dir)
        persist_dir.mkdir(parents=True, exist_ok=True)

        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # Cosine space matches the L2-normalised embeddings. Chroma reports
        # distance = 1 - cosine_similarity.
        self._collection = self._client.get_or_create_collection(
            name=collection_name or settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Insert or update chunks. Re-adding an existing id overwrites it (upsert)."""
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        self._collection.upsert(
            ids=[c.id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )

    def query(self, embedding: list[float], k: int | None = None) -> list[dict]:
        """Return the ``k`` most similar chunks, nearest first."""
        k = k or settings.top_k
        res = self._collection.query(
            query_embeddings=[embedding],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits: list[dict] = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            # Convert distance back to a similarity score for callers.
            hits.append(
                {"text": doc, "metadata": meta, "distance": dist, "score": 1.0 - dist}
            )
        return hits

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self._collection.count()
