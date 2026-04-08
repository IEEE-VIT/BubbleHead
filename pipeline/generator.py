"""
generator.py - BubbleHead RAG Answer Generator

Generates answers to user queries based on retrieved document chunks.
"""

import logging
from typing import List, Dict, Optional
import ollama

from config import (
    LLM_MODEL,
    LLM_TEMPERATURE,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a research analyst who answers based on documents provided. Generate a clear, complete and correct answer to the user's question. Only use information present in the documents. No prior knowledge or assumptions beyond what is provided. Give the answer in a structured manner. Maintain factual accuracy. Cite which passage supports each claim. If multiple pieces of information are correct, combine them logically."""


def _format_chunks(chunks: List[Dict]) -> str:
    """Format chunks as numbered passages: [1] text... [2] text... etc."""
    formatted = []
    for i, chunk in enumerate(chunks, 1):
        formatted.append(f"[{i}] {chunk['text']}")
    return "\n\n".join(formatted)


def generate_answer(
    user_query: str,
    chunks: List[Dict],
    gap_status: Optional[str] = None,
    missing_info: Optional[str] = None
) -> str:
    """
    Generate an answer to the user's query based on retrieved chunks.
    
    Args:
        user_query: The user's question
        chunks: List of chunk dicts with 'text' key
        gap_status: Optional status from gap analysis ('PASS' or 'FORCE_PASS')
        missing_info: Optional note about missing information (when status is FORCE_PASS)
        
    Returns:
        Generated answer string
    """
    logger.info("Generating answer for query: %.50s... (%d chunks)", user_query, len(chunks))
    
    # Format chunks as numbered passages
    formatted_chunks = _format_chunks(chunks)
    
    # Build user prompt
    user_prompt = f"""QUESTION:
{user_query}

DOCUMENTS:
{formatted_chunks}"""
    
    # Append missing info note if FORCE_PASS
    if gap_status == 'FORCE_PASS' and missing_info:
        user_prompt += f"\n\nNote: the following information could not be retrieved: {missing_info}"
        logger.info("FORCE_PASS status: appending missing info note")
    
    # Call LLM to generate answer
    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
            options={
                "temperature": LLM_TEMPERATURE,
                "timeout": 60
            }
        )
        
        answer = response['message']['content']
        logger.info("Answer generated successfully (%d chars)", len(answer))
        
        return answer
        
    except Exception as e:
        logger.error("Answer generation failed: %s", e)
        raise
