# BubbleHead RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) pipeline with iterative gap analysis, BM25 reranking, and a Gradio UI for testing.

## Features

- **Advanced Retrieval**: ChromaDB vector store with BM25 reranking
- **Gap Analysis**: Iterative quality evaluation with confidence scoring
- **Token Budget Management**: Automatic token counting and budget enforcement
- **Multiple Document Formats**: PDF, DOCX, PPTX, TXT, HTML, CSV support
- **LangGraph Pipeline**: State-based orchestration with conditional routing
- **Interactive UI**: Gradio interface for document ingestion and querying
- **GPU Support**: Automatic GPU acceleration via Ollama

## Architecture

```
retrieve_node → generate_node → gap_analysis_node
                                      ↓
                                   [PASS] → END
                                   [RETRY] → retrieve_node (loop)
```

## Prerequisites

- Python 3.11+
- [Ollama](https://ollama.ai/) installed and running
- NVIDIA GPU (optional, for acceleration)

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd RAG
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Pull required Ollama models**
   ```bash
   ollama pull nomic-embed-text
   ollama pull kimi-k2.5:cloud
   ```

## Configuration

Edit `config.py` to customize:

```python
# Models
EMBED_MODEL = 'nomic-embed-text'
LLM_MODEL = 'kimi-k2.5:cloud'

# Retrieval
TOP_K_CANDIDATES = 10
TOP_K_FINAL = 6
TOKEN_BUDGET = 5000

# Gap Analysis
GAP_CONFIDENCE_THRESHOLD = 0.6
GAP_MAX_ITERATIONS = 2
```

## Usage

### Option 1: Gradio UI (Recommended)

```bash
# Windows
start_ui.bat

# Linux/Mac
./start_ui.sh
```

Then open http://localhost:7860 in your browser.

**UI Features:**
- **Ingest Tab**: Upload and process documents
- **Query Tab**: Ask questions and get answers
- **Collection Info Tab**: View database statistics

### Option 2: Command Line

**Ingest documents:**
```bash
python main.py ingest <directory_path>
```

**Query:**
```bash
python main.py query "What is the main topic?"
```

## Project Structure

```
RAG/
├── config.py                 # Configuration constants
├── main.py                   # CLI entry point
├── ui.py                     # Gradio interface
├── requirements.txt          # Python dependencies
├── ingestion/
│   ├── Chunker.py           # Document chunking
│   ├── Embedder.py          # Embedding & storage
│   └── parsers/
│       └── Parser.py        # Multi-format parsing
├── retrieval/
│   ├── Retriever.py         # BM25 + vector retrieval
│   └── gap_analysis_agent.py # Quality evaluation
├── pipeline/
│   ├── generator.py         # Answer generation
│   └── pipeline.py          # LangGraph orchestration
└── data/
    └── chroma/              # Vector database (gitignored)
```

## How It Works

### 1. Document Ingestion
- Parse documents (PDF, DOCX, PPTX, etc.)
- Chunk text (512 tokens, 50 overlap)
- Generate embeddings with `nomic-embed-text`
- Store in ChromaDB

### 2. Retrieval
- Embed query with `search_query:` prefix
- Retrieve top 10 candidates from ChromaDB
- Rerank with BM25
- Select top 6 within token budget (5000 tokens)

### 3. Generation
- Format chunks as numbered passages
- Generate answer with `kimi-k2.5:cloud`
- Use system prompt for structured responses

### 4. Gap Analysis
- Evaluate answer quality
- Confidence scoring (0.0-1.0)
- If confidence < 0.6: retry with refined query
- Max 2 iterations

## GPU Acceleration

Ollama automatically uses your GPU if available. To verify:

```bash
ollama ps
```

Look for "GPU" in the PROCESSOR column.

## Development

### Running Tests
```bash
pytest tests/
```

### Code Style
```bash
black .
flake8 .
```

## Troubleshooting

**Issue: "No module named 'gradio'"**
- Solution: Activate venv and run `pip install -r requirements.txt`

**Issue: Ollama connection error**
- Solution: Ensure Ollama is running: `ollama serve`

**Issue: GPU not detected**
- Solution: Restart Ollama service or reinstall with GPU support

**Issue: ChromaDB errors**
- Solution: Delete `data/chroma/` and re-ingest documents

## Performance Tips

1. **Adjust chunk size** for your documents (config.py)
2. **Tune gap analysis threshold** (lower = more retries)
3. **Increase token budget** for longer contexts
4. **Use GPU** for 5-10x faster inference

## License

MIT License - see LICENSE file for details

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Powered by [Ollama](https://ollama.ai/)
- Vector store: [ChromaDB](https://www.trychroma.com/)
- UI: [Gradio](https://gradio.app/)

## Contact

For issues and questions, please open a GitHub issue.
