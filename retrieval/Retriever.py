"""
Retriever.py - BubbleHead RAG Query Retriever

Embeds queries and retrieves top-k candidates from ChromaDB.
Reranks using BM25 and applies token budget constraints.
"""

import logging
from typing import List, Dict
import chromadb
from chromadb.config import Settings
import ollama
from pathlib import Path
from rank_bm25 import BM25Okapi
import tiktoken

from config import (
    CHROMA_PATH,
    CHROMA_COLLECTION,
    EMBED_MODEL,
    TOP_K_CANDIDATES,
    TOP_K_FINAL,
    TOKEN_BUDGET,
)

logger = logging.getLogger(__name__)

# Initialize tokenizer for token counting
_tokenizer = tiktoken.get_encoding("cl100k_base")


def _estimate_tokens(text: str) -> int:
    """Estimate token count using tiktoken cl100k_base."""
    return len(_tokenizer.encode(text))


def _embed_query(query: str) -> List[float]:
    """
    Embed query using nomic-embed-text via Ollama with search_query prefix.
    
    Args:
        query: Plain-text query string
        
    Returns:
        768-dimensional embedding vector
    """
    prefixed_query = "search_query: " + query
    
    try:
        resp = ollama.embeddings(
            model=EMBED_MODEL,
            prompt=prefixed_query,
            options={"timeout": 20}
        )
        return resp["embedding"]
    except Exception as e:
        logger.error("Query embedding failed: %s", e)
        raise


def retrieve(query: str, top_k: int = TOP_K_CANDIDATES) -> List[Dict]:
    """
    Retrieve top-k most similar chunks for a query with BM25 reranking and token budget.
    
    Pipeline:
    1. Embed query and retrieve top_k candidates from ChromaDB (cosine similarity)
    2. Rerank candidates using BM25Okapi
    3. Select top TOP_K_FINAL from reranked list
    4. Apply TOKEN_BUDGET constraint (stop before exceeding 5000 tokens)
    
    Args:
        query: Plain-text query string
        top_k: Number of initial candidates to retrieve (default: TOP_K_CANDIDATES from config)
        
    Returns:
        List of dicts with keys: 'text', 'metadata', 'distance', 'bm25_score'
    """
    logger.info("Retrieving top %d results for query: %.50s...", top_k, query)
    
    # Initialize ChromaDB client
    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=CHROMA_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    
    # Get collection
    try:
        collection = client.get_collection(name=CHROMA_COLLECTION)
    except Exception as e:
        logger.error("Failed to get collection '%s': %s", CHROMA_COLLECTION, e)
        raise
    
    # Embed query
    query_embedding = _embed_query(query)
    
    # Query ChromaDB with cosine similarity
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )
    
    # Format initial candidates
    candidates = []
    for i in range(len(results['ids'][0])):
        candidates.append({
            'text': results['documents'][0][i],
            'metadata': results['metadatas'][0][i],
            'distance': results['distances'][0][i],
        })
    
    logger.info("Retrieved %d candidates from ChromaDB", len(candidates))
    
    # ── BM25 RERANKING ────────────────────────────────────────────────────
    # Tokenize by whitespace (no stemming)
    corpus_tokens = [doc['text'].split() for doc in candidates]
    query_tokens = query.split()
    
    # Initialize BM25 and score candidates
    bm25 = BM25Okapi(corpus_tokens)
    bm25_scores = bm25.get_scores(query_tokens)
    
    # Attach BM25 scores and sort by score (descending)
    for i, candidate in enumerate(candidates):
        candidate['bm25_score'] = bm25_scores[i]
    
    reranked = sorted(candidates, key=lambda x: x['bm25_score'], reverse=True)
    
    # Select top TOP_K_FINAL
    top_reranked = reranked[:TOP_K_FINAL]
    logger.info("BM25 reranking: selected top %d from %d candidates", len(top_reranked), len(candidates))
    
    # ── TOKEN BUDGET ENFORCEMENT ──────────────────────────────────────────
    final_chunks = []
    running_token_count = 0
    
    for chunk in top_reranked:
        # Get token count from metadata or compute it
        if 'token_count' in chunk['metadata']:
            chunk_tokens = chunk['metadata']['token_count']
        else:
            chunk_tokens = _estimate_tokens(chunk['text'])
        
        # Check if adding this chunk would exceed budget
        if running_token_count + chunk_tokens > TOKEN_BUDGET:
            logger.info("Token budget reached: %d tokens (would exceed %d)", 
                       running_token_count, TOKEN_BUDGET)
            break
        
        final_chunks.append(chunk)
        running_token_count += chunk_tokens
    
    logger.info("Final retrieval: %d chunks, %d tokens (budget: %d)", 
               len(final_chunks), running_token_count, TOKEN_BUDGET)
    
    return final_chunks
