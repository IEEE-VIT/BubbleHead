"""
pipeline.py - BubbleHead RAG Pipeline using LangGraph

Orchestrates the RAG pipeline: retrieval → gap analysis → generation
"""

import logging
from typing import TypedDict, List, Dict, Optional, Literal
from langgraph.graph import StateGraph, END

from retrieval.Retriever import retrieve
from retrieval.gap_analysis_agent import run_gap_analysis
from pipeline.generator import generate_answer

logger = logging.getLogger(__name__)


class PipelineState(TypedDict):
    """State for the RAG pipeline graph."""

    query: str  # User's original query
    chunks: List[Dict]  # Accumulated retrieved chunks
    gap_status: str  # 'PASS' or 'FORCE_PASS'
    missing_info: str  # Missing information note (for FORCE_PASS)
    answer: str  # Final generated answer
    iteration: int  # Current iteration count


def retrieve_node(state: PipelineState) -> PipelineState:
    """
    Retrieval node: retrieves chunks for the query.

    On the first iteration uses the original query. On retries uses the
    refined query stored in missing_info by gap_analysis_node.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with retrieved chunks
    """
    logger.info("Retrieve node: processing query")

    # Use the gap-analysis-refined query on retries, otherwise the original.
    refined = state.get("missing_info", "")
    query = refined if (state.get("iteration", 0) > 0 and refined) else state["query"]

    logger.info(
        "Retrieving with query (iteration=%d): %.60s...",
        state.get("iteration", 0),
        query,
    )

    chunks = retrieve(query)

    logger.info("Retrieved %d chunks", len(chunks))

    return {
        **state,
        "chunks": chunks,
    }


def gap_analysis_node(state: PipelineState) -> PipelineState:
    """
    Gap analysis node: evaluates answer quality and determines if retry is needed.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with gap analysis results
    """
    logger.info("Gap analysis node: evaluating answer quality")

    query = state["query"]
    chunks = state.get("chunks", [])
    answer = state.get("answer", "")

    # Import analyze_gap for evaluation
    from retrieval.gap_analysis_agent import analyze_gap

    # Evaluate the generated answer against the chunks
    result = analyze_gap(query, chunks)

    logger.info(
        "Gap analysis complete: classification=%s, confidence=%.2f",
        result.classification,
        result.confidence,
    )

    # Determine status based on confidence
    from config import GAP_CONFIDENCE_THRESHOLD

    if result.confidence >= GAP_CONFIDENCE_THRESHOLD:
        status = "PASS"
    else:
        status = "RETRY"

    return {
        **state,
        "gap_status": status,
        "missing_info": result.refined_query if status == "RETRY" else "",
        "iteration": state.get("iteration", 0) + 1,
    }


def generate_node(state: PipelineState) -> PipelineState:
    """
    Generation node: generates answer based on chunks.

    Args:
        state: Current pipeline state

    Returns:
        Updated state with generated answer
    """
    logger.info("Generate node: creating answer")

    query = state["query"]
    chunks = state["chunks"]

    # Generate answer (without gap status since gap analysis comes after)
    answer = generate_answer(
        user_query=query, chunks=chunks, gap_status=None, missing_info=None
    )

    logger.info("Answer generated (%d chars)", len(answer))

    return {
        **state,
        "answer": answer,
    }


# ── CONDITIONAL ROUTING ───────────────────────────────────────────────────


def route_after_gap_analysis(state: PipelineState) -> Literal["END", "retrieve_node"]:
    """
    Route based on gap analysis status and iteration count.

    - 'PASS'                              → END
    - 'RETRY' + within max iterations     → retrieve_node
    - 'RETRY' + max iterations exhausted  → END  (FORCE_PASS)
    """
    from config import GAP_MAX_ITERATIONS

    gap_status = state.get("gap_status", "")
    iteration = state.get("iteration", 0)

    if gap_status == "PASS":
        logger.info("Routing to END — gap analysis PASS")
        return "END"

    if iteration >= GAP_MAX_ITERATIONS:
        logger.warning(
            "Routing to END — max iterations (%d) reached without PASS (FORCE_PASS)",
            GAP_MAX_ITERATIONS,
        )
        return "END"

    logger.info("Routing to retrieve_node for retry (iteration=%d)", iteration)
    return "retrieve_node"


# ── GRAPH CONSTRUCTION ────────────────────────────────────────────────────

# Build the StateGraph
workflow = StateGraph(PipelineState)

# Add nodes
workflow.add_node("retrieve_node", retrieve_node)
workflow.add_node("generate_node", generate_node)
workflow.add_node("gap_analysis_node", gap_analysis_node)

# Add edges
workflow.set_entry_point("retrieve_node")
workflow.add_edge("retrieve_node", "generate_node")
workflow.add_edge("generate_node", "gap_analysis_node")

# Conditional edge from gap_analysis_node
workflow.add_conditional_edges(
    "gap_analysis_node",
    route_after_gap_analysis,
    {
        "END": END,
        "retrieve_node": "retrieve_node",
    },
)

# Compile the graph
graph = workflow.compile()

logger.info("RAG pipeline graph compiled successfully")


# ── PUBLIC API ────────────────────────────────────────────────────────────


def run(query: str) -> str:
    """
    Run the RAG pipeline for a given query.

    Args:
        query: User's question

    Returns:
        Generated answer string
    """
    logger.info("Running RAG pipeline for query: %.50s...", query)

    # Initialize state
    initial_state: PipelineState = {
        "query": query,
        "chunks": [],
        "gap_status": "",
        "missing_info": "",
        "answer": "",
        "iteration": 0,
    }

    # Invoke the graph
    final_state = graph.invoke(initial_state)

    answer = final_state["answer"]
    logger.info("Pipeline complete: answer generated (%d chars)", len(answer))

    return answer
