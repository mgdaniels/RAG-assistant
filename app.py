"""Gradio web interface.

A thin front end over ``ragassistant.generate.answer_question``. It displays the
grounded answer, the cited sources, and the retrieved chunks (with similarity
scores) the answer was based on. It contains no retrieval or generation logic of
its own.

Usage:
    pip install -e ".[ui]"
    python app.py
"""
from __future__ import annotations

import os
from functools import lru_cache

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # disable telemetry

import gradio as gr

from ragassistant.config import settings
from ragassistant.generate import ConfigurationError, answer_question
from ragassistant.store import VectorStore

EXAMPLES = [
    "How many days of paid leave do I get in my first year?",
    "How quickly must I report a lost laptop?",
    "What is the mileage rate for using my own car?",
    "Who do I escalate a SEV1 incident to?",
    "What is the company's pension match?",
    "What is the capital of France?",  # off-topic: expected to be declined
]


@lru_cache(maxsize=1)
def get_store() -> VectorStore:
    """Open the persistent store once and reuse it across requests."""
    return VectorStore()


def _render_chunks(retrieved) -> str:
    """Render retrieved chunks as Markdown for the context panel."""
    if not retrieved:
        return "_No chunks retrieved._"
    blocks = []
    for rank, c in enumerate(retrieved, start=1):
        snippet = " ".join(c.text.split())[:240]
        blocks.append(f"**{rank}. {c.citation}** — score {c.score:.3f}\n\n> {snippet}…")
    return "\n\n".join(blocks)


def respond(question: str, k: int):
    """Answer a question and return (answer, sources, retrieved-context)."""
    question = (question or "").strip()
    if not question:
        return "Please enter a question.", "", ""

    store = get_store()
    if store.count() == 0:
        return "The index is empty — run `rag ingest` first to build it.", "", ""

    try:
        answer = answer_question(question, k=int(k), store=store)
    except ConfigurationError as exc:
        return f"⚠️ {exc}", "", ""
    except Exception as exc:  # show errors in the UI instead of the console
        return f"Something went wrong: {exc}", "", ""

    sources = ", ".join(answer.sources) if answer.sources else "—"
    return answer.text, sources, _render_chunks(answer.retrieved)


def build_demo() -> gr.Blocks:
    """Construct the Gradio interface."""
    with gr.Blocks(title="Northwind RAG Assistant") as demo:
        gr.Markdown(
            "# Northwind RAG Assistant\n"
            "Ask a natural-language question about the Northwind Robotics handbook. "
            "Answers are grounded in the retrieved documents and cite their sources; "
            "off-topic questions are declined rather than guessed."
        )
        with gr.Row():
            question = gr.Textbox(
                label="Question",
                placeholder="e.g. How many days of leave do I get?",
                scale=4,
            )
            k = gr.Slider(1, 8, value=settings.top_k, step=1, label="Chunks to retrieve (k)", scale=1)
        ask_btn = gr.Button("Ask", variant="primary")

        gr.Markdown("### Answer")
        answer = gr.Markdown()
        sources = gr.Textbox(label="Sources consulted", interactive=False)
        with gr.Accordion("Retrieved context (what the answer was grounded in)", open=False):
            chunks = gr.Markdown()

        gr.Examples(EXAMPLES, inputs=question)

        ask_btn.click(respond, inputs=[question, k], outputs=[answer, sources, chunks])
        question.submit(respond, inputs=[question, k], outputs=[answer, sources, chunks])
    return demo


demo = build_demo()

if __name__ == "__main__":
    # Pass share=True to demo.launch() for a temporary public URL.
    demo.launch()
