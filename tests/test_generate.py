"""Tests for answer generation. The model call is stubbed."""
from ragassistant.retrieve import RetrievedChunk
from ragassistant import generate as gen
from ragassistant.generate import (
    format_context,
    build_messages,
    answer_question,
    REFUSAL,
)


def _chunk(source, title, score, text="some policy text"):
    return RetrievedChunk(text=text, source=source, title=title, chunk_index=0, score=score)


def test_context_lists_sources_and_text():
    ctx = format_context([_chunk("leave-policy.md", "Leave", 0.6, "25 days of leave")])
    assert "leave-policy.md" in ctx
    assert "25 days of leave" in ctx
    assert "[1]" in ctx


def test_system_prompt_enforces_grounding():
    msgs = build_messages("how much leave?", [_chunk("a.md", "A", 0.5)])
    system = msgs[0]["content"].lower()
    assert "only" in system
    assert "don't know" in system
    assert "how much leave?" in msgs[1]["content"]


def test_low_score_refuses_without_calling_llm(monkeypatch):
    monkeypatch.setattr(gen, "retrieve", lambda *a, **k: [_chunk("a.md", "A", 0.05)])

    def _boom(messages):
        raise AssertionError("the LLM must not be called on a low-score refusal")

    monkeypatch.setattr(gen, "call_llm", _boom)
    answer = answer_question("an unrelated question", min_score=0.25)
    assert answer.text == REFUSAL
    assert answer.sources == []


def test_no_chunks_refuses(monkeypatch):
    monkeypatch.setattr(gen, "retrieve", lambda *a, **k: [])
    assert answer_question("anything").text == REFUSAL


def test_good_score_calls_llm_and_dedupes_sources(monkeypatch):
    chunks = [
        _chunk("leave-policy.md", "Leave", 0.70),
        _chunk("benefits.md", "Benefits", 0.50),
        _chunk("leave-policy.md", "Leave", 0.45),  # duplicate source
    ]
    monkeypatch.setattr(gen, "retrieve", lambda *a, **k: chunks)
    monkeypatch.setattr(gen, "call_llm", lambda messages: "You get 25 days. (leave-policy.md)")

    answer = answer_question("how much leave?", min_score=0.25)
    assert "25 days" in answer.text
    assert answer.sources == ["Leave (leave-policy.md)", "Benefits (benefits.md)"]
