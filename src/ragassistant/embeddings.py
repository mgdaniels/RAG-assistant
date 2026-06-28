"""Text embeddings via a local sentence-transformers model.

Embeddings are L2-normalised so that cosine similarity is equivalent to a dot
product, which matches the metric used by the vector store. The model is loaded
lazily and cached, because loading it (and downloading it on first use) is the
expensive step and only needs to happen once per process.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .config import settings


def _l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Scale each row vector to unit length. Zero vectors are left unchanged."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return vectors / norms


class Embedder:
    """Encodes text into normalised vectors. The model loads on first use."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.embed_model
        self._model = None

    @property
    def model(self):
        if self._model is None:
            # Imported here so the dependency and the model download are
            # deferred until embeddings are actually required.
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """The dimensionality of the embedding vectors."""
        return self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts into normalised vectors."""
        if not texts:
            return []
        vectors = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        # encode(..., normalize_embeddings=True) is equivalent; normalisation is
        # done explicitly here to keep the metric obvious.
        return _l2_normalize(np.asarray(vectors, dtype=np.float32)).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self.embed_texts([text])[0]


@lru_cache(maxsize=1)
def get_embedder() -> Embedder:
    """Return a process-wide cached embedder so the model loads at most once."""
    return Embedder()
