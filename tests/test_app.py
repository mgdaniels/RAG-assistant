"""Tests for the Gradio handler. Skipped if gradio is not installed; the pipeline is stubbed."""
import pytest

pytest.importorskip("gradio")

import app as appmod
from ragassistant.generate import Answer
from ragassistant.retrieve import RetrievedChunk


class _FakeStore:
    def __init__(self, count=3):
        self._count = count

    def count(self):
        return self._count


def test_demo_builds():
    assert appmod.demo is not None


def test_respond_rejects_empty_question(monkeypatch):
    monkeypatch.setattr(appmod, "get_store", lambda: _FakeStore())
    answer, _, _ = appmod.respond("   ", 4)
    assert "enter a question" in answer.lower()


def test_respond_warns_on_empty_index(monkeypatch):
    monkeypatch.setattr(appmod, "get_store", lambda: _FakeStore(count=0))
    answer, _, _ = appmod.respond("anything", 4)
    assert "rag ingest" in answer


def test_respond_renders_answer_sources_and_chunks(monkeypatch):
    monkeypatch.setattr(appmod, "get_store", lambda: _FakeStore(count=3))
    rc = RetrievedChunk("25 days of leave", "leave-policy.md", "Leave", 0, 0.712)
    monkeypatch.setattr(
        appmod,
        "answer_question",
        lambda q, k, store: Answer("You get 25 days.", ["Leave (leave-policy.md)"], [rc]),
    )
    answer, sources, chunks = appmod.respond("how much leave?", 4)
    assert "25 days" in answer
    assert "leave-policy.md" in sources
    assert "0.712" in chunks
