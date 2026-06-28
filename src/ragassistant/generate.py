"""Answer generation.

Builds a prompt from retrieved context and calls an OpenAI-compatible chat model
to answer the question. The model is instructed to use only the supplied context
and to cite its sources. If no retrieved chunk is sufficiently relevant, the
assistant declines instead of answering. Because the call goes through the
OpenAI-compatible interface, the same code targets any such endpoint (for
example Google Gemini or a local Ollama server) via configuration.
"""
from __future__ import annotations

from dataclasses import dataclass

from .config import settings
from .retrieve import RetrievedChunk, retrieve

REFUSAL = "I couldn't find an answer to that in the Northwind handbook."

SYSTEM_PROMPT = (
    "You are an assistant that answers questions about Northwind Robotics' "
    "internal handbook. Use ONLY the numbered context passages provided to "
    "answer the question. If the answer is not contained in the context, say "
    "you don't know rather than guessing. Be concise, and cite the source "
    "document(s) you relied on, e.g. (leave-policy.md)."
)


class ConfigurationError(RuntimeError):
    """Raised when the LLM provider is not configured (e.g. a missing API key)."""


@dataclass(frozen=True)
class Answer:
    text: str
    sources: list[str]               # source documents the context came from
    retrieved: list[RetrievedChunk]  # the chunks used, for transparency


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render retrieved chunks as numbered, source-labelled passages."""
    return "\n\n".join(
        f"[{i}] {c.title} ({c.source})\n{c.text}" for i, c in enumerate(chunks, start=1)
    )


def build_messages(question: str, chunks: list[RetrievedChunk]) -> list[dict]:
    """Build the system and user chat messages for a question and its context."""
    user = (
        f"Context passages:\n\n{format_context(chunks)}\n\n"
        f"Question: {question}\n\n"
        "Answer using only the context above, and cite the source document(s)."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def call_llm(messages: list[dict]) -> str:
    """Send ``messages`` to the configured chat model and return the reply text."""
    if not settings.llm_api_key:
        raise ConfigurationError(
            "No LLM API key set. Add LLM_API_KEY to your .env file "
            "(get a free Gemini key at https://aistudio.google.com), or point "
            "LLM_BASE_URL at a local Ollama server and set LLM_API_KEY=ollama."
        )
    # Imported lazily so the dependency is only required when generating.
    from openai import OpenAI

    client = OpenAI(api_key=settings.llm_api_key, base_url=settings.llm_base_url)
    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=0,  # deterministic output for factual answers
    )
    return (response.choices[0].message.content or "").strip()


def answer_question(
    question: str,
    k: int | None = None,
    store=None,
    min_score: float | None = None,
) -> Answer:
    """Retrieve context for ``question``, then either answer or decline."""
    min_score = settings.min_relevance_score if min_score is None else min_score
    chunks = retrieve(question, k=k, store=store)

    # Decline when nothing is relevant enough. This avoids an unsupported answer
    # and an unnecessary model call.
    if not chunks or chunks[0].score < min_score:
        return Answer(text=REFUSAL, sources=[], retrieved=chunks)

    text = call_llm(build_messages(question, chunks))
    sources = _unique([c.citation for c in chunks])
    return Answer(text=text, sources=sources, retrieved=chunks)


def _unique(items: list[str]) -> list[str]:
    """Return ``items`` de-duplicated, preserving first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
