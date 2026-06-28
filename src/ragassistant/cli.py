"""Command-line interface.

Sub-commands:
  ingest   build the vector index from the documents in the corpus
  ask      answer a question using retrieval-augmented generation
  search   show the chunks retrieved for a question (no model call)
"""
from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rag",
        description="Answer natural-language questions over a document set (RAG).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Load, chunk, embed and index the documents.")

    p_ask = sub.add_parser("ask", help="Ask a question over the indexed documents.")
    p_ask.add_argument("question", help="The natural-language question to answer.")
    p_ask.add_argument("-k", "--top-k", type=int, default=None, help="How many chunks to retrieve.")

    p_search = sub.add_parser("search", help="Show the chunks retrieved for a question (no LLM call).")
    p_search.add_argument("question", help="The question to retrieve chunks for.")
    p_search.add_argument("-k", "--top-k", type=int, default=None, help="How many chunks to retrieve.")

    args = parser.parse_args(argv)

    if args.command == "ingest":
        # Imported lazily so `rag --help` stays fast and heavy dependencies load
        # only when the command actually runs.
        from .config import settings
        from .ingest import chunk_documents
        from .embeddings import get_embedder
        from .store import VectorStore

        chunks = chunk_documents()
        print(f"Chunked {len(chunks)} pieces from the corpus. Embedding (first run downloads the model)...")
        embedder = get_embedder()
        embeddings = embedder.embed_texts([c.text for c in chunks])
        store = VectorStore()
        store.add(chunks, embeddings)
        print(
            f"Indexed {store.count()} chunks into collection "
            f"'{settings.collection_name}' (dim={len(embeddings[0])}). Ready to query."
        )
    elif args.command == "search":
        from .retrieve import retrieve, format_results

        results = retrieve(args.question, k=args.top_k)
        print(format_results(args.question, results))
    elif args.command == "ask":
        from .generate import answer_question, ConfigurationError

        try:
            answer = answer_question(args.question, k=args.top_k)
        except ConfigurationError as exc:
            print(f"Configuration error: {exc}")
            return
        print(answer.text)
        if answer.sources:
            print("\nSources consulted: " + ", ".join(answer.sources))


if __name__ == "__main__":
    main()
