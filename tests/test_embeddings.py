"""Tests for the embedding helpers (normalisation only; the model is not loaded)."""
import numpy as np

from ragassistant.embeddings import _l2_normalize


def test_normalize_gives_unit_vectors():
    vectors = np.array([[3.0, 4.0], [1.0, 0.0], [-2.0, 2.0]])
    out = _l2_normalize(vectors)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0)


def test_normalize_preserves_direction():
    # A normalised vector points the same way as the original.
    v = np.array([[3.0, 4.0]])
    out = _l2_normalize(v)
    assert np.allclose(out, [[0.6, 0.8]])


def test_normalize_handles_zero_vector():
    out = _l2_normalize(np.array([[0.0, 0.0]]))
    assert np.all(np.isfinite(out))  # no NaN/inf from divide-by-zero
