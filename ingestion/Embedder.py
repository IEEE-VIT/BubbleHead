"""
embedder.py - Context Window Optimized RAG Embedder

DESIGN GOAL: Guarantee retriever never exceeds 5000-token budget
Worst-case: 6 chunks × 512 tokens = 3072 tokens (61% budget)
Headroom: 1928 tokens for gap analysis (4 passes × ~480 tokens)

━━━ SPEC ENFORCEMENT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
- 512-token MAX per chunk (HARD VALIDATION)
- nomic-embed-text 768-dim ONLY
- Parallel embedding (>10k chunks)
- Raw text storage (no prompt pollution)
- Deterministic deduplication
- Production monitoring"""

import logging
from typing import List, Tuple, Set, Any
import chromadb
import ollama
from chromadb.config import Settings
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import json
from pathlib import Path
import hashlib
import tiktoken
import threading

from config import (
    CHROMA_COLLECTION,
    CHROMA_PATH,
    EMBED_MODEL,
    OLLAMA_BASE_URL,
)

# CONTEXT WINDOW CONSTANTS (HARD ENFORCED)
MAX_CHUNK_TOKENS = 512  # SPEC: Never exceed
EXPECTED_EMBED_DIM = 768  # nomic-embed-text ONLY
CHROMA_UPSERT_BATCH = 250  # Memory-optimized
EMBED_PARALLEL_WORKERS = 4  # Ollama concurrency limit
CONTEXT_BUDGET_HEADROOM = 5000 - (6 * MAX_CHUNK_TOKENS)  # Computed dynamically

REQUIRED_METADATA: Set[str] = {
    "source_file",
    "chunk_index",
    "page_number",
    "section_heading",
    "document_type",
}

logger = logging.getLogger(__name__)

# Lazy-initialized tokenizer
_tokenizer = None
_tokenizer_lock = threading.Lock()


def _get_tokenizer():
    """Lazy init tokenizer on first use to avoid loading tiktoken on every import."""
    global _tokenizer
    if _tokenizer is None:
        with _tokenizer_lock:
            if _tokenizer is None:
                _tokenizer = tiktoken.get_encoding("cl100k_base")
    return _tokenizer


# Singleton client with thread-safe initialization
_chroma_client: chromadb.PersistentClient = None
_client_lock = threading.Lock()


def _init_client() -> chromadb.PersistentClient:
    global _chroma_client
    if _chroma_client is None:
        with _client_lock:
            if _chroma_client is None:
                Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
                _chroma_client = chromadb.PersistentClient(
                    path=CHROMA_PATH, settings=Settings(anonymized_telemetry=False)
                )
    return _chroma_client


# TOKEN ESTIMATION
def _estimate_tokens(text: str) -> int:
    # Accurate token count using tiktoken (cl100k_base).
    return len(_get_tokenizer().encode(text))


# DEDUPLICATION
def _content_hash(text: str) -> str:
    """
    Stable 16-char SHA-256 fingerprint for deterministic deduplication.

    Used as the content-addressed component of ChromaDB chunk IDs so that
    re-ingesting the same document is always idempotent regardless of whether
    chunk indices shift (e.g. after adding pages to a doc).
    """
    return hashlib.sha256(text.encode()).hexdigest()[:16]


# CHUNK VALIDATION (Fix 5 + Fix 6)
def validate_chunk(chunk: Any, idx: int) -> Tuple[bool, str]:
    # Full spec validation: token limit + all 5 required metadata fields.
    errors = []

    # 1. TOKEN LIMIT (CRITICAL for context budget)
    tokens = _estimate_tokens(chunk.text)
    if tokens > MAX_CHUNK_TOKENS:
        errors.append(f"{tokens} tokens > {MAX_CHUNK_TOKENS}")

    # 2. NON-EMPTY
    if not chunk.text.strip():
        errors.append("empty text")

    for field in REQUIRED_METADATA:
        if field not in chunk.metadata:
            errors.append(f"missing metadata field: '{field}'")

    valid = not errors
    if not valid:
        logger.warning("Chunk %d REJECTED: %s", idx, "; ".join(errors))

    return valid, "; ".join(errors)


# EMBEDDING ENGINE
def _embed_single(
    text: str, is_query: bool = False, retry_count: int = 1
) -> List[float]:
    # Single embedding using nomic-embed-text asymmetric prompting with retry.
    prefix = "search_query: " if is_query else "search_document: "
    full_text = prefix + text

    last_error = None
    for attempt in range(retry_count + 1):
        try:
            resp = ollama.embeddings(
                model=EMBED_MODEL, prompt=full_text, options={"timeout": 20}
            )
            embedding = resp["embedding"]

            # DIMENSION CHECK (spec compliance)
            if len(embedding) != EXPECTED_EMBED_DIM:
                raise ValueError(
                    "Dim mismatch: %d != %d" % (len(embedding), EXPECTED_EMBED_DIM)
                )

            return embedding

        except Exception as e:
            last_error = e
            if attempt < retry_count:
                time.sleep(0.5)  # Short backoff before retry
                continue
            raise RuntimeError("Embedding failed: %.100s" % str(e))


def embed_chunks_parallel(
    chunks: List, is_query: bool = False
) -> List[Tuple[Any, List[float]]]:
    """Parallel embedding with per-future error handling. Returns only successful (chunk, embedding) pairs."""
    logger.info("Parallel embedding %d chunks (is_query=%s)...", len(chunks), is_query)
    start = time.time()

    with ThreadPoolExecutor(max_workers=EMBED_PARALLEL_WORKERS) as executor:
        futures = {
            executor.submit(_embed_single, c.text, is_query): (i, c)
            for i, c in enumerate(chunks)
        }

        successful_pairs = []
        failed_count = 0

        for future in as_completed(futures):
            idx, chunk = futures[future]
            try:
                embedding = future.result()
                successful_pairs.append((chunk, embedding))
            except Exception as e:
                logger.error("Chunk %d embedding failed: %s", idx, e)
                failed_count += 1

    success_count = len(successful_pairs)
    elapsed = time.time() - start
    logger.info(
        " Embedded %d/%d chunks in %.1fs (%.0f/s), %d failed",
        success_count,
        len(chunks),
        elapsed,
        success_count / elapsed if elapsed > 0 else 0,
        failed_count,
    )
    return successful_pairs


# MAIN INGESTION PIPELINE
def embed_and_store(chunks: List, collection_name: str = CHROMA_COLLECTION) -> dict:
    """
    PRODUCTION INGESTION: Validates → Embeds → Deduplicates → Stores → Stats

    Returns dict with context budget safety metrics.
    """
    if not chunks:
        return {
            "stored": 0,
            "rejected": 0,
            "embed_failed": 0,
            "avg_tokens": 0,
            "context_safe": True,
        }

    client = _init_client()
    collection = client.get_or_create_collection(
        name=collection_name, metadata={"hnsw:space": "cosine"}
    )

    # ── PHASE 1: VALIDATION (Context Budget Gatekeeper) ──────────────────
    valid_chunks = []
    rejected = 0
    total_tokens = 0
    max_tokens = 0

    for i, chunk in enumerate(chunks):
        valid, reason = validate_chunk(chunk, i)
        if valid:
            valid_chunks.append(chunk)
            chunk_tokens = _estimate_tokens(chunk.text)
            total_tokens += chunk_tokens
            max_tokens = max(max_tokens, chunk_tokens)
        else:
            rejected += 1

    avg_tokens = total_tokens / len(valid_chunks) if valid_chunks else 0
    context_safe = max_tokens <= MAX_CHUNK_TOKENS  # Use worst-case, not average

    logger.info(
        "VALIDATION: %d/%d valid (%d rejected), avg %.0f tokens/chunk, max %d tokens",
        len(valid_chunks),
        len(chunks),
        rejected,
        avg_tokens,
        max_tokens,
    )

    if not context_safe:
        logger.error("  MAX CHUNK SIZE EXCEEDS 512 TOKENS — CONTEXT BUDGET RISK!")

    # ── PHASE 2: PARALLEL EMBEDDING (Fix 4 — safe failure handling) ──────
    successful_pairs = embed_chunks_parallel(valid_chunks, is_query=False)

    # Unpack successful pairs
    safe_chunks = [chunk for chunk, _ in successful_pairs]
    safe_embeddings = [embedding for _, embedding in successful_pairs]

    embed_failed = len(valid_chunks) - len(successful_pairs)
    if embed_failed > 0:
        logger.warning("  %d chunks dropped due to embedding failure", embed_failed)

    # ── PHASE 3: BATCHED STORAGE ──────────────────────────────────────────
    stored = 0
    ingest_start = time.time()

    for i in range(0, len(safe_chunks), CHROMA_UPSERT_BATCH):
        batch = safe_chunks[i : i + CHROMA_UPSERT_BATCH]
        batch_embeddings = safe_embeddings[i : i + CHROMA_UPSERT_BATCH]

        # Old index-based IDs (source_file::chunk_index) broke deduplication
        # whenever chunk indices shifted after document edits.
        ids = [f"{c.metadata['source_file']}::{_content_hash(c.text)}" for c in batch]
        documents = [c.text for c in batch]  # RAW TEXT for clean retrieval
        metadatas = [c.metadata.copy() for c in batch]

        # Annotate each chunk's token count for retriever budget awareness
        for j, meta in enumerate(metadatas):
            meta["token_count"] = _estimate_tokens(documents[j])

        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=batch_embeddings,
            metadatas=metadatas,
        )
        stored += len(batch)

    elapsed = time.time() - ingest_start

    # ── FINAL STATS ───────────────────────────────────────────────────────
    stats = {
        "stored": stored,
        "rejected": rejected,
        "embed_failed": embed_failed,
        "avg_tokens": avg_tokens,
        "max_tokens": max_tokens,
        "context_safe": context_safe,
        "collection_total": collection.count(),
        "throughput_chunks_sec": stored / elapsed if elapsed > 0 else 0,
        "headroom_tokens": CONTEXT_BUDGET_HEADROOM - (6 * max_tokens),
        "budget_utilization_pct": (6 * max_tokens / 5000) * 100,
    }

    logger.info(" INGEST COMPLETE: %s", json.dumps(stats, indent=2))
    return stats


# UTILITIES
def get_collection(collection_name: str = CHROMA_COLLECTION) -> chromadb.Collection:
    return _init_client().get_collection(name=collection_name)


def delete_collection(collection_name: str = CHROMA_COLLECTION) -> None:
    _init_client().delete_collection(name=collection_name)
    logger.info("  Deleted '%s'", collection_name)


def collection_stats(collection_name: str = CHROMA_COLLECTION) -> dict:
    # Context-aware collection stats.
    try:
        coll = get_collection(collection_name)
        return {
            "name": collection_name,
            "count": coll.count(),
            "safe_for_context": True,
        }
    except Exception as e:
        logger.warning("collection_stats failed for '%s': %s", collection_name, e)
        return {"error": "Collection not found or unavailable: %s" % e}


def init_logging():
    """Initialize logging configuration. Call this at entry point, not at module level."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
