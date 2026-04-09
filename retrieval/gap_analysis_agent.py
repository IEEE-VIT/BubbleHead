"""
gap_analysis_agent.py - BubbleHead RAG Gap Analysis Agent

Evaluates whether retrieved chunks are sufficient to answer the user's query.
Iteratively refines queries until context is sufficient or max iterations reached.
"""

import logging
from typing import List, Dict, Callable
from dataclasses import dataclass
import ollama
import re
import hashlib

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
    GAP_CONFIDENCE_THRESHOLD,
    GAP_MAX_ITERATIONS,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a gap analysis agent sitting between the retrieval step and the language model in a RAG pipeline. Your job is to decide whether the chunks retrieved from the vector database are good enough to answer the user's question before anything is sent to the generator.

Read the user's question and all retrieved passages carefully. Then classify the context as one of: sufficient, partial, or insufficient.

If partial or insufficient, identify exactly what is missing and rewrite the query to target that gap.

Do not pass anything to the generator until the context is sufficient or the maximum number of retries has been reached.

Respond in the following format:
CLASSIFICATION: [sufficient/partial/insufficient]
CONFIDENCE: [0.0-1.0]
REFINED_QUERY: [your refined query if classification is partial or insufficient, otherwise "N/A"]"""


@dataclass
class GapAnalysisResult:
    """Result of gap analysis evaluation."""
    classification: str  # 'sufficient', 'partial', or 'insufficient'
    confidence: float    # 0.0 to 1.0
    refined_query: str   # Refined query if classification != sufficient, else None


@dataclass
class GapAnalysisOutput:
    """Final output of the gap analysis retry loop."""
    status: str          # 'PASS' or 'FORCE_PASS'
    chunks: List[Dict]   # Accumulated chunks
    confidence: float    # Final confidence score
    iterations: int      # Number of iterations performed
    missing_info: str    # What information could not be found (for FORCE_PASS)


def _format_chunks(chunks: List[Dict]) -> str:
    """Format chunks as numbered passages."""
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(f"[Passage {i}]\n{chunk['text']}\n")
    return "\n".join(formatted)


def _text_hash(text: str) -> str:
    """Generate hash for deduplication."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _parse_response(response: str) -> GapAnalysisResult:
    """
    Parse LLM response to extract classification, confidence, and refined query.
    
    Expected format:
    CLASSIFICATION: sufficient
    CONFIDENCE: 0.85
    REFINED_QUERY: N/A
    """
    classification = "insufficient"
    confidence = 0.0
    refined_query = None
    
    # Extract classification
    class_match = re.search(r'CLASSIFICATION:\s*(sufficient|partial|insufficient)', response, re.IGNORECASE)
    if class_match:
        classification = class_match.group(1).lower()
    
    # Extract confidence
    conf_match = re.search(r'CONFIDENCE:\s*(0?\.\d+|1\.0|1)', response)
    if conf_match:
        try:
            confidence = float(conf_match.group(1))
        except ValueError:
            logger.warning("Failed to parse confidence score: %s", conf_match.group(1))
    
    # Extract refined query
    query_match = re.search(r'REFINED_QUERY:\s*(.+?)(?:\n|$)', response, re.DOTALL)
    if query_match:
        refined_query_text = query_match.group(1).strip()
        if refined_query_text.lower() != "n/a" and classification != "sufficient":
            refined_query = refined_query_text
    
    return GapAnalysisResult(
        classification=classification,
        confidence=confidence,
        refined_query=refined_query
    )


def analyze_gap(query: str, chunks: List[Dict]) -> GapAnalysisResult:
    """
    Analyze whether retrieved chunks are sufficient to answer the query.
    
    Args:
        query: Original user query string
        chunks: List of retrieved chunk dicts with 'text' and 'metadata' keys
        
    Returns:
        GapAnalysisResult with classification, confidence, and refined_query
    """
    logger.info("Gap analysis for query: %.50s... (%d chunks)", query, len(chunks))
    
    # Format chunks as numbered passages
    formatted_chunks = _format_chunks(chunks)
    
    # Build user prompt
    user_prompt = f"""USER QUESTION:
{query}

RETRIEVED PASSAGES:
{formatted_chunks}

Analyze whether these passages are sufficient to answer the user's question."""
    
    # Call LLM
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": LLM_TEMPERATURE,
                "timeout": 30
            }
        )
        
        llm_output = response['message']['content']
        logger.debug("LLM response: %s", llm_output)
        
        # Parse response
        result = _parse_response(llm_output)
        
        logger.info(
            "Gap analysis result: %s (confidence: %.2f)%s",
            result.classification,
            result.confidence,
            f", refined query: {result.refined_query[:50]}..." if result.refined_query else ""
        )
        
        return result
        
    except Exception as e:
        logger.error("Gap analysis failed: %s", e)
        # Return conservative fallback
        return GapAnalysisResult(
            classification="insufficient",
            confidence=0.0,
            refined_query=query
        )


def run_gap_analysis(query: str, retriever_fn: Callable[[str], List[Dict]]) -> GapAnalysisOutput:
    """
    Run iterative gap analysis with retry loop.
    
    Pipeline:
    1. Retrieve initial chunks using retriever_fn(query)
    2. Score chunks with analyze_gap()
    3. If confidence >= GAP_CONFIDENCE_THRESHOLD: return PASS
    4. If confidence < threshold and iterations < GAP_MAX_ITERATIONS:
       - Use refined_query to retrieve new chunks
       - Accumulate new chunks (deduplicated by text hash)
       - Score the full accumulated pool
       - Repeat up to GAP_MAX_ITERATIONS
    5. If max iterations exhausted: return FORCE_PASS with missing info note
    
    Args:
        query: Original user query string
        retriever_fn: Function that takes a query string and returns List[Dict] of chunks
        
    Returns:
        GapAnalysisOutput with status, chunks, confidence, iterations, and missing_info
    """
    logger.info("Starting gap analysis retry loop for query: %.50s...", query)
    
    # Track accumulated chunks with deduplication
    accumulated_chunks = []
    seen_hashes = set()
    
    current_query = query
    iteration = 0
    final_result = None
    
    for iteration in range(1, GAP_MAX_ITERATIONS + 1):
        logger.info("Gap analysis iteration %d/%d", iteration, GAP_MAX_ITERATIONS)
        
        # Retrieve chunks
        new_chunks = retriever_fn(current_query)
        logger.info("Retrieved %d chunks for query: %.50s...", len(new_chunks), current_query)
        
        # Deduplicate and accumulate
        added_count = 0
        for chunk in new_chunks:
            chunk_hash = _text_hash(chunk['text'])
            if chunk_hash not in seen_hashes:
                seen_hashes.add(chunk_hash)
                accumulated_chunks.append(chunk)
                added_count += 1
        
        logger.info("Added %d new chunks (total: %d)", added_count, len(accumulated_chunks))
        
        # Score the full accumulated pool
        result = analyze_gap(query, accumulated_chunks)
        final_result = result
        
        logger.info(
            "Iteration %d: classification=%s, confidence=%.2f (threshold=%.2f)",
            iteration,
            result.classification,
            result.confidence,
            GAP_CONFIDENCE_THRESHOLD
        )
        
        # Check if confidence threshold met
        if result.confidence >= GAP_CONFIDENCE_THRESHOLD:
            logger.info("Gap analysis PASS: confidence %.2f >= %.2f after %d iterations",
                       result.confidence, GAP_CONFIDENCE_THRESHOLD, iteration)
            return GapAnalysisOutput(
                status='PASS',
                chunks=accumulated_chunks,
                confidence=result.confidence,
                iterations=iteration,
                missing_info=None
            )
        
        # Check if we have more iterations
        if iteration < GAP_MAX_ITERATIONS:
            # Use refined query for next iteration
            if result.refined_query:
                current_query = result.refined_query
                logger.info("Refining query for next iteration: %.50s...", current_query)
            else:
                logger.warning("No refined query provided, using original query")
                current_query = query
        else:
            # Max iterations exhausted
            break
    
    # Max iterations exhausted without reaching threshold
    missing_info = "Unable to find sufficient information after %d iterations. " % GAP_MAX_ITERATIONS
    if final_result and final_result.refined_query:
        missing_info += "Missing: %s" % final_result.refined_query
    
    logger.warning(
        "Gap analysis FORCE_PASS: confidence %.2f < %.2f after %d iterations",
        final_result.confidence if final_result else 0.0,
        GAP_CONFIDENCE_THRESHOLD,
        iteration
    )
    
    return GapAnalysisOutput(
        status='FORCE_PASS',
        chunks=accumulated_chunks,
        confidence=final_result.confidence if final_result else 0.0,
        iterations=iteration,
        missing_info=missing_info
    )
