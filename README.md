# BubbleHead RAG Pipeline

BubbleHead is a local retrieval-augmented generation (RAG) application built on Ollama, ChromaDB, BM25 reranking, and an iterative gap-analysis loop. The current repo ships a FastAPI backend, a static HTML/JavaScript frontend, and a CLI for batch ingestion and ad hoc queries.

## Highlights

- Hybrid retrieval with ChromaDB vector search and BM25 reranking
- Gap analysis loop that retries low-confidence queries with refined wording
- Token-budget-aware chunking and retrieval safeguards
- Multi-format ingestion for PDF, DOCX, PPTX, TXT, HTML, and CSV files
- Local-first runtime with Ollama models and on-disk Chroma persistence
- Browser UI plus CLI entry points for ingestion and querying

## Architecture

```text
retrieve_node -> generate_node -> gap_analysis_node
                                      |
                                   PASS -> end
                                   RETRY -> retrieve_node
```

## Requirements

- Python 3.11 or 3.12
- [Ollama](https://ollama.com/) installed locally
- A pulled embedding model: `nomic-embed-text`
- A pulled generation model: `mistral:latest`

## Quick Start

1. Clone the repository and move into it.
   ```bash
   git clone <your-repo-url>
   cd BubbleHead
   ```
2. Create and activate a virtual environment.
   ```bash
   python -m venv .venv
   ```
   Windows:
   ```bash
   .venv\Scripts\activate
   ```
   macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```
3. Install Python dependencies.
   ```bash
   pip install -r requirements.txt
   ```
4. Start Ollama and pull the models used by `config.py`.
   ```bash
   ollama serve
   ```
   In another terminal:
   ```bash
   ollama pull nomic-embed-text
   ollama pull mistral:latest
   ```

## Running The App

### Web UI

Start the FastAPI server with either helper script or directly:

```bash
# Windows
start_ui.bat

# macOS / Linux
./start_ui.sh

# direct
python ui.py
```

Open `http://localhost:7860` after the server reports that the pipeline is ready.

### CLI

Ingest every supported document under a directory:

```bash
python main.py ingest ./data
```

Run a query against the stored collection:

```bash
python main.py query "What are the main findings?"
```

## API Surface

The FastAPI app in `ui.py` exposes:

- `GET /api/status` for warm-up state
- `POST /api/ingest` for single-file ingestion
- `POST /api/query` for question answering
- `GET /api/collection` for collection stats

## Configuration

Project defaults live in `config.py`.

```python
OLLAMA_BASE_URL = "http://localhost:11434"
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:latest"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
TOP_K_CANDIDATES = 10
TOP_K_FINAL = 6
TOKEN_BUDGET = 5000

GAP_CONFIDENCE_THRESHOLD = 0.6
GAP_MAX_ITERATIONS = 2
```

## Project Layout

```text
BubbleHead/
|-- config.py
|-- main.py
|-- ui.py
|-- frontend/
|   `-- index.html
|-- ingestion/
|   |-- Chunker.py
|   |-- Embedder.py
|   `-- parsers/Parser.py
|-- retrieval/
|   |-- Retriever.py
|   `-- gap_analysis_agent.py
|-- pipeline/
|   |-- generator.py
|   `-- pipeline.py
`-- data/
    `-- chroma/
```

## Retrieval Flow

1. Documents are parsed into sections, chunked, validated, embedded, and stored in ChromaDB.
2. Queries are embedded with the `search_query:` prefix and matched against stored vectors.
3. The retriever reranks candidates with BM25 and trims the final context to the configured token budget.
4. The generator produces an answer, and the gap-analysis step decides whether retrieval should retry.

## Troubleshooting

**Ollama connection errors**

Make sure `ollama serve` is running and the configured models are pulled locally.

**Pipeline warming up**

The UI loads before heavy pipeline imports finish. Retry once `GET /api/status` reports `"ready": true`.

**Missing parser dependencies**

Reinstall with `pip install -r requirements.txt`. PDF fallback parsing requires `pdfplumber`.

**Collection issues**

Delete `data/chroma/` and re-ingest if the local collection becomes inconsistent.

## Development

Suggested local checks:

```bash
black --check .
flake8 .
python -m compileall .
```
