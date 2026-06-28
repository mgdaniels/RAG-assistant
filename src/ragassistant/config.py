"""Application configuration.

Settings are read from environment variables (loaded from a local ``.env`` file
when present) with defaults, so the application runs without any configuration
and can be customised without code changes. Every module imports the shared
``settings`` instance defined at the bottom of this file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load variables from a .env file if present. Existing environment variables
# take precedence over the file.
load_dotenv()

# Project root, resolved relative to this file so that configured paths are
# independent of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    # LLM provider: any OpenAI-compatible endpoint (default: Google Gemini).
    llm_base_url: str = os.getenv(
        "LLM_BASE_URL",
        "https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    llm_model: str = os.getenv("LLM_MODEL", "gemini-2.5-flash")
    llm_api_key: str = os.getenv("LLM_API_KEY", "")

    # Embedding model (runs locally; no API key required).
    embed_model: str = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    # Vector store location and collection name.
    chroma_dir: Path = PROJECT_ROOT / os.getenv("CHROMA_DIR", ".chroma")
    collection_name: str = os.getenv("COLLECTION_NAME", "handbook")

    # Document corpus directory.
    docs_dir: Path = PROJECT_ROOT / os.getenv("DOCS_DIR", "data/docs")

    # Chunking parameters, in characters.
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # Number of chunks retrieved per query.
    top_k: int = int(os.getenv("TOP_K", "4"))

    # Minimum cosine similarity for the best chunk. If the top result scores
    # below this, the assistant declines rather than answer without support.
    min_relevance_score: float = float(os.getenv("MIN_RELEVANCE_SCORE", "0.25"))


settings = Settings()
