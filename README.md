# RAG Assistant

A small, self-contained **retrieval-augmented generation (RAG)** tool that answers
natural-language questions over a set of documents, using an LLM API together with
embedding-based vector search.

It ships with a demo corpus — the internal handbook of a fictional company,
*Northwind Robotics* — so you can try it immediately. Ask something like
*"How many days of paid leave do I get in my first year?"* and it returns an answer
grounded in the relevant document, with the source cited. Questions the documents
don't cover are declined rather than answered from guesswork.

## Requirements

- Python 3.10 or newer
- An API key for an OpenAI-compatible LLM (a free Google Gemini key works; see
  [Configuration](#configuration)), or a local [Ollama](https://ollama.com) install (change the config in the .env to include your own key or swap to llama)

The embedding model runs locally and is downloaded automatically on first use
(~90 MB); no embedding API key is required.

## Installation

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

**Windows (PowerShell)**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
```

> If PowerShell blocks the activation script, run
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` first, or skip
> activation and call the tools directly, e.g. `.\.venv\Scripts\rag.exe ingest`.

## Configuration

Copy the example environment file and add your API key:

```bash
cp .env.example .env        # Windows PowerShell: copy .env.example .env
```

Open `.env` and set `LLM_API_KEY` to a Gemini key from
<https://aistudio.google.com> (free, no card required). `.env` is gitignored, so
your key stays local.

To run fully offline with Ollama instead, install it, run `ollama pull llama3.1`,
and set in `.env`:

```
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
LLM_API_KEY=ollama
```

All settings and their defaults:

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_BASE_URL` | Gemini OpenAI endpoint | Base URL of the OpenAI-compatible API |
| `LLM_MODEL` | `gemini-2.5-flash` | Chat model name |
| `LLM_API_KEY` | _(empty)_ | API key for the LLM provider |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `TOP_K` | `4` | Chunks retrieved per query |
| `CHUNK_SIZE` | `800` | Maximum chunk size (characters) |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks (characters) |
| `MIN_RELEVANCE_SCORE` | `0.25` | Minimum similarity to answer; below this it declines |
| `CHROMA_DIR` | `.chroma` | Where the index is stored |
| `COLLECTION_NAME` | `handbook` | Chroma collection name |

## Usage

**1. Build the index** (run once, and again whenever the documents change):

```bash
rag ingest
```

**2. Ask a question:**

```bash
rag ask "How many days of paid leave do I get in my first year?"
```

The answer is printed with the sources it consulted. Off-topic questions are
declined. Use `-k` to change how many chunks are retrieved, e.g. `rag ask "..." -k 6`.

**3. Inspect retrieval** without calling the model (useful for tuning):

```bash
rag search "expensing a client dinner" -k 4
```

### Web UI (optional)

A Gradio interface provides the same functionality in the browser:

```bash
pip install -e ".[ui]"
python app.py
```

Open the printed URL (usually <http://127.0.0.1:7860>). It shows the answer, the
cited sources, and an expandable panel with the retrieved chunks and their
similarity scores.

## Running the tests

```bash
pip install -e ".[dev]"
pytest
```

The tests stub the embedding model and the LLM, so they run quickly and need no
network access or API key.

## How it works

```
INGEST  (rag ingest)                   QUERY  (rag ask)
────────────────────                   ────────────────
data/docs/*.md                         a question
      │                                      │
      ▼                                      ▼
  split into overlapping chunks        embed the query
      │                                      │
      ▼                                      ▼
  embed chunks  ───►  Chroma  ◄──── similarity search (top-k)
                    (on disk)               │
                                            ▼
                                  build a grounded prompt
                                            │
                                            ▼
                                    LLM (Gemini / Ollama)
                                            │
                                            ▼
                                   answer + cited sources
```

Documents are split into overlapping, size-bounded chunks and embedded with a
local sentence-transformers model; the vectors are stored in a persistent Chroma
collection. At query time the question is embedded the same way, the most similar
chunks are retrieved, and an OpenAI-compatible chat model is asked to answer using
only that retrieved context and to cite its sources. If nothing retrieved is
relevant enough, the assistant declines instead of guessing.

## Project structure

```
rag-assistant/
├── README.md
├── LICENSE
├── pyproject.toml
├── .env.example
├── data/docs/               # the demo corpus (Markdown)
├── src/ragassistant/
│   ├── config.py            # environment-based settings
│   ├── ingest.py            # document loading and chunking
│   ├── embeddings.py        # local embedding model
│   ├── store.py             # Chroma vector store
│   ├── retrieve.py          # top-k similarity search
│   ├── generate.py          # grounded answer generation
│   └── cli.py               # `rag` command-line interface
├── app.py                   # optional Gradio web UI
├── conftest.py
└── tests/
```

## License

MIT — see [LICENSE](LICENSE).
