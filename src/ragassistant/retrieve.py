"""Query-time retrieval.

Embeds a question with the same model used for the documents and returns the
most similar chunks from the vector store as typed, attributable results.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .embeddings import get_embedder
from .store import VectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned for a query, with its similarity score and source."""

    text: str
    source: str
    title: str
    chunk_index: int
    score: float  # cosine similarity in [-1, 1]; higher is more similar

    @property
    def citation(self) -> str:
        """A short, human-readable source reference."""
        return f"{self.title} ({self.source})"


def retrieve(
    question: str,
    k: int | None = None,
    store: VectorStore | None = None,
) -> list[RetrievedChunk]:
    """Embed ``question`` and return the ``k`` most similar chunks, nearest first."""
    k = k or settings.top_k
    store = store or VectorStore()
    query_vector = get_embedder().embed_query(question)
    hits = store.query(query_vector, k=k)
    return [
        RetrievedChunk(
            text=h["text"],
            source=h["metadata"]["source"],
            title=h["metadata"]["title"],
            chunk_index=h["metadata"]["chunk_index"],
            score=h["score"],
        )
        for h in hits
    ]


def format_results(question: str, results: list[RetrievedChunk], snippet_chars: int = 200) -> str:
    """Render retrieved chunks as readable text for the ``search`` command."""
    if not results:
        return "No results — has the index been built? Run `rag ingest` first."
    lines = [f'Top {len(results)} chunks for: "{question}"', ""]
    for rank, r in enumerate(results, start=1):
        snippet = " ".join(r.text.split())[:snippet_chars]
        lines.append(f"{rank}. [score {r.score:.3f}] {r.citation}  (chunk {r.chunk_index})")
        lines.append(f"    {snippet}...")
        lines.append("")
    return "\n".join(lines).rstrip()
