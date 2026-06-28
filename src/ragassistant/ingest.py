"""Document loading and chunking.

Loads Markdown files from the corpus directory and splits each document into
overlapping, size-bounded chunks suitable for embedding and retrieval. Splitting
is structure-aware: it keeps whole paragraphs together where possible, falls back
to sentence boundaries for oversized paragraphs, and splits on character count
only as a last resort. Each chunk records its source file, document title, and
position so that results can be attributed to their origin.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from .config import settings


@dataclass(frozen=True)
class Chunk:
    """A unit of text together with the metadata needed to attribute it."""

    id: str            # stable, deterministic id, e.g. "leave-policy::0"
    text: str
    source: str        # source filename, e.g. "leave-policy.md"
    title: str         # document title (first H1), e.g. "Annual Leave and Sickness Policy"
    chunk_index: int   # 0-based position of this chunk within its document

    @property
    def metadata(self) -> dict:
        # Chroma metadata must be a flat mapping of str/int/float/bool values.
        return {"source": self.source, "title": self.title, "chunk_index": self.chunk_index}


@dataclass(frozen=True)
class Document:
    source: str
    title: str
    text: str


# --- Loading -----------------------------------------------------------------

def load_documents(docs_dir: Path | None = None) -> list[Document]:
    """Load all Markdown files in ``docs_dir`` (sorted for deterministic order)."""
    docs_dir = Path(docs_dir or settings.docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    documents: list[Document] = []
    for path in sorted(docs_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        documents.append(
            Document(source=path.name, title=_extract_title(text, path.stem), text=text)
        )
    return documents


def _extract_title(text: str, fallback: str) -> str:
    """Return the document's first Markdown H1, or ``fallback`` if there is none."""
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


# --- Splitting ---------------------------------------------------------------

def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text into overlapping chunks of about ``chunk_size`` characters.

    Boundaries are chosen in order of preference: paragraph, then sentence, then
    a hard character split. Adjacent chunks overlap by roughly ``chunk_overlap``
    characters to preserve context across boundaries.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    atoms = _atomise(text, chunk_size)
    return _pack_with_overlap(atoms, chunk_size, chunk_overlap)


def _atomise(text: str, chunk_size: int) -> list[str]:
    """Split text into ordered pieces, each no longer than ``chunk_size``."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    atoms: list[str] = []
    for para in paragraphs:
        if len(para) <= chunk_size:
            atoms.append(para)
            continue
        for sentence in _split_sentences(para):       # paragraph too long
            if len(sentence) <= chunk_size:
                atoms.append(sentence)
            else:                                      # sentence too long
                atoms.extend(
                    sentence[i:i + chunk_size] for i in range(0, len(sentence), chunk_size)
                )
    return atoms


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation (. ! ?) followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def _pack_with_overlap(atoms: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Merge pieces into chunks up to ``chunk_size``, overlapping adjacent chunks."""
    chunks: list[str] = []
    current = ""
    for atom in atoms:
        candidate = f"{current}\n\n{atom}" if current else atom
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # Begin the next chunk with the tail of the previous one for continuity.
        overlap = _tail(current, chunk_overlap)
        current = f"{overlap}\n\n{atom}" if overlap else atom
    if current:
        chunks.append(current)
    return chunks


def _tail(text: str, n: int) -> str:
    """Return up to the last ``n`` characters, trimmed to start at a word boundary."""
    if n <= 0 or not text:
        return ""
    tail = text[-n:]
    space = tail.find(" ")
    return tail[space + 1:] if 0 <= space < len(tail) - 1 else tail


# --- Chunking: load, split, and attach metadata ------------------------------

def chunk_documents(
    docs_dir: Path | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Chunk]:
    """Load the corpus and return all chunks with ids and metadata attached."""
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    chunks: list[Chunk] = []
    for doc in load_documents(docs_dir):
        for i, piece in enumerate(split_text(doc.text, chunk_size, chunk_overlap)):
            stem = Path(doc.source).stem
            chunks.append(
                Chunk(id=f"{stem}::{i}", text=piece, source=doc.source,
                      title=doc.title, chunk_index=i)
            )
    return chunks


def chunking_report(chunks: list[Chunk]) -> str:
    """Return a human-readable summary of a set of chunks."""
    if not chunks:
        return "No chunks produced — is data/docs/ empty?"
    sizes = [len(c.text) for c in chunks]
    by_source: dict[str, int] = {}
    for c in chunks:
        by_source[c.source] = by_source.get(c.source, 0) + 1
    lines = [
        f"{len(chunks)} chunks from {len(by_source)} documents",
        f"chunk size (chars): min {min(sizes)}, mean {sum(sizes)//len(sizes)}, max {max(sizes)}",
        "chunks per document:",
        *[f"  {src:<26} {n}" for src, n in sorted(by_source.items())],
    ]
    return "\n".join(lines)
