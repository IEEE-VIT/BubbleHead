"""
config.py - BubbleHead RAG Configuration

Central configuration for the BubbleHead RAG pipeline.

Constants:
-----------
CHROMA_PATH : str
    Path to ChromaDB persistent storage directory

CHROMA_COLLECTION : str
    Name of the ChromaDB collection for document embeddings

OLLAMA_BASE_URL : str
    Base URL for the Ollama API server

EMBED_MODEL : str
    Embedding model name for nomic-embed-text via Ollama

LLM_MODEL : str
    Language model name for generation and gap analysis

LLM_TEMPERATURE : float
    Temperature setting for LLM generation (0.0-1.0, lower = more deterministic)

CHUNK_SIZE : int
    Maximum token size per document chunk

CHUNK_OVERLAP : int
    Number of overlapping tokens between consecutive chunks

TOP_K_CANDIDATES : int
    Number of initial candidates to retrieve from ChromaDB (before BM25 reranking)

TOP_K_FINAL : int
    Number of chunks to select after BM25 reranking (before token budget)

TOKEN_BUDGET : int
    Maximum total tokens allowed for all retrieved chunks combined

GAP_CONFIDENCE_THRESHOLD : float
    Minimum confidence score (0.0-1.0) required to pass gap analysis

GAP_MAX_ITERATIONS : int
    Maximum number of retrieval iterations in gap analysis retry loop
"""

# ── VECTOR DATABASE ───────────────────────────────────────────────────────
CHROMA_PATH = './data/chroma'
CHROMA_COLLECTION = 'bubblehead'

# ── OLLAMA CONFIGURATION ──────────────────────────────────────────────────
OLLAMA_BASE_URL = 'http://localhost:11434'
EMBED_MODEL = 'nomic-embed-text'
LLM_MODEL = 'kimi-k2.5:cloud'
LLM_TEMPERATURE = 0.3

# ── CHUNKING PARAMETERS ───────────────────────────────────────────────────
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# ── RETRIEVAL PARAMETERS ──────────────────────────────────────────────────
TOP_K_CANDIDATES = 10
TOP_K_FINAL = 6
TOKEN_BUDGET = 5000

# ── GAP ANALYSIS PARAMETERS ───────────────────────────────────────────────
GAP_CONFIDENCE_THRESHOLD = 0.6
GAP_MAX_ITERATIONS = 2
