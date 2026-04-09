# BubbleHead RAG Testing UI

A simple web interface for testing the BubbleHead RAG pipeline.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Ollama

Make sure Ollama is running with the required models:

```bash
# Start Ollama server
ollama serve

# Pull required models (in another terminal)
ollama pull nomic-embed-text
ollama pull mistral:7b
```

### 3. Launch the UI

```bash
cd RAG
python ui.py
```

The UI will be available at: **http://localhost:7860**

## Features

### 📥 Ingest Documents Tab
- Upload documents (PDF, DOCX, PPTX, TXT, HTML, CSV)
- Automatic parsing, chunking, and embedding
- Real-time ingestion statistics

### 🔍 Query RAG Tab
- Ask questions about your documents
- View generated answers with citations
- Pipeline status information
- Example questions provided

### 📊 Collection Info Tab
- View collection statistics
- Check total chunks stored
- Verify system readiness

## Usage Flow

1. **Ingest Documents**
   - Go to "Ingest Documents" tab
   - Upload a document
   - Click "Ingest Document"
   - Wait for confirmation

2. **Query the System**
   - Go to "Query RAG" tab
   - Enter your question
   - Click "Ask Question"
   - View the generated answer

3. **Check Statistics**
   - Go to "Collection Info" tab
   - Click "Refresh Stats"
   - View collection details

## Pipeline Features

The UI demonstrates the full BubbleHead RAG pipeline:

- **Retrieval**: ChromaDB vector search with cosine similarity
- **BM25 Reranking**: Hybrid retrieval for better results
- **Gap Analysis**: Iterative query refinement (up to 4 iterations)
- **Token Budget**: Enforces 5000-token limit
- **Answer Generation**: Mistral 7B with citation support

## Troubleshooting

### "Collection not found" error
- Ingest at least one document first

### "Connection refused" error
- Make sure Ollama is running: `ollama serve`

### Slow responses
- First query may be slow (model loading)
- Subsequent queries should be faster

### Empty answers
- Check that documents were ingested successfully
- Verify Ollama models are downloaded

## CLI Alternative

You can also use the command-line interface:

```bash
# Ingest documents
python main.py ingest ./data

# Query the system
python main.py query "What is BubbleHead?"
```

## Configuration

Edit `config.py` to customize:
- Chunk size and overlap
- Top-K candidates and final results
- Token budget
- Gap analysis threshold and iterations
- Model names

## Logs

The UI logs all operations to the console. Check the terminal for:
- Ingestion progress
- Query processing steps
- Error messages
- Performance metrics
