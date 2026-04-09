"""
ui.py - BubbleHead RAG Testing UI

Simple Gradio interface for testing the RAG pipeline.
"""

import logging
import gradio as gr
from pathlib import Path
from typing import List, Tuple

from ingestion.parsers.Parser import parse as parse_file
from ingestion.Chunker import chunk_document
from ingestion.Embedder import embed_and_store, collection_stats
from pipeline.pipeline import run


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def ingest_file(file) -> str:
    """
    Ingest a single uploaded file.
    
    Args:
        file: Gradio file upload object
        
    Returns:
        Status message string
    """
    if file is None:
        return "❌ No file uploaded"
    
    try:
        file_path = file.name
        file_name = Path(file_path).name
        
        logger.info("Processing uploaded file: %s", file_name)
        
        # Parse file - returns list of dicts with 'text' key
        parsed_sections = parse_file(file_path)
        
        if not parsed_sections:
            return f"❌ No content extracted from {file_name}"
        
        # Combine all text sections
        combined_text = "\n\n".join([section['text'] for section in parsed_sections if section.get('text')])
        
        if not combined_text.strip():
            return f"❌ No text extracted from {file_name}"
        
        # Chunk document
        chunks = chunk_document(combined_text)
        
        if not chunks:
            return f"❌ No chunks created from {file_name}"
        
        # Add metadata to chunks
        for chunk in chunks:
            if 'metadata' not in chunk:
                chunk['metadata'] = {}
            chunk['metadata']['source_file'] = file_name
            chunk['metadata']['chunk_index'] = chunks.index(chunk)
            chunk['metadata']['page_number'] = 0
            chunk['metadata']['section_heading'] = ''
            chunk['metadata']['document_type'] = Path(file_name).suffix[1:]
        
        # Convert to proper chunk objects for embedder
        from dataclasses import dataclass
        
        @dataclass
        class ChunkObj:
            text: str
            metadata: dict
        
        chunk_objects = [ChunkObj(text=c['text'], metadata=c.get('metadata', {})) for c in chunks]
        
        # Embed and store
        stats = embed_and_store(chunk_objects)
        
        stored = stats.get('stored', 0)
        rejected = stats.get('rejected', 0)
        
        return f"""✅ Successfully ingested {file_name}
        
📊 Statistics:
- Chunks stored: {stored}
- Chunks rejected: {rejected}
- Average tokens: {stats.get('avg_tokens', 0):.0f}
- Max tokens: {stats.get('max_tokens', 0)}
- Collection total: {stats.get('collection_total', 0)}
"""
        
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        return f"❌ Error: {str(e)}"


def query_rag(question: str) -> Tuple[str, str]:
    """
    Query the RAG pipeline.
    
    Args:
        question: User's question
        
    Returns:
        Tuple of (answer, status_info)
    """
    if not question or not question.strip():
        return "❌ Please enter a question", ""
    
    try:
        logger.info("Processing query: %s", question)
        
        # Run RAG pipeline
        answer = run(question)
        
        # Get collection stats
        stats = collection_stats()
        status_info = f"""📊 Collection Stats:
- Total chunks: {stats.get('count', 0)}
- Status: {'✅ Ready' if stats.get('count', 0) > 0 else '⚠️ Empty'}
"""
        
        return answer, status_info
        
    except Exception as e:
        logger.error("Query failed: %s", e)
        return f"❌ Error: {str(e)}", ""


def get_collection_info() -> str:
    """Get current collection statistics."""
    try:
        stats = collection_stats()
        
        if 'error' in stats:
            return f"⚠️ {stats['error']}"
        
        return f"""📊 Collection Statistics:
        
- Name: {stats.get('name', 'N/A')}
- Total chunks: {stats.get('count', 0)}
- Status: {'✅ Ready for queries' if stats.get('count', 0) > 0 else '⚠️ Empty - ingest documents first'}
"""
    except Exception as e:
        return f"❌ Error: {str(e)}"


# Build Gradio interface
demo = gr.Blocks(title="BubbleHead RAG Testing UI")

with demo:
    gr.Markdown("""
    # 🫧 BubbleHead RAG Testing UI
    
    Test the BubbleHead RAG pipeline with document ingestion and querying.
    """)
    
    with gr.Tabs():
        # Tab 1: Ingestion
        with gr.Tab("📥 Ingest Documents"):
            gr.Markdown("""
            ### Upload and ingest documents
            Supported formats: PDF, DOCX, PPTX, TXT, HTML, CSV
            """)
            
            with gr.Row():
                with gr.Column():
                    file_input = gr.File(
                        label="Upload Document",
                        file_types=[".pdf", ".docx", ".pptx", ".txt", ".html", ".csv"]
                    )
                    ingest_btn = gr.Button("🚀 Ingest Document", variant="primary")
                
                with gr.Column():
                    ingest_output = gr.Textbox(
                        label="Ingestion Status",
                        lines=10,
                        interactive=False
                    )
            
            ingest_btn.click(
                fn=ingest_file,
                inputs=[file_input],
                outputs=[ingest_output]
            )
        
        # Tab 2: Query
        with gr.Tab("🔍 Query RAG"):
            gr.Markdown("""
            ### Ask questions about your documents
            The RAG pipeline will retrieve relevant chunks and generate an answer.
            """)
            
            with gr.Row():
                with gr.Column():
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="What is BubbleHead?",
                        lines=3
                    )
                    query_btn = gr.Button("🔍 Ask Question", variant="primary")
                    
                    gr.Markdown("### Examples")
                    gr.Examples(
                        examples=[
                            ["What is the main topic of the document?"],
                            ["Summarize the key points"],
                            ["What are the technical requirements?"],
                        ],
                        inputs=[question_input]
                    )
            
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="Answer",
                        lines=15,
                        interactive=False
                    )
                
                with gr.Column():
                    status_output = gr.Textbox(
                        label="Pipeline Status",
                        lines=5,
                        interactive=False
                    )
            
            query_btn.click(
                fn=query_rag,
                inputs=[question_input],
                outputs=[answer_output, status_output]
            )
        
        # Tab 3: Collection Info
        with gr.Tab("📊 Collection Info"):
            gr.Markdown("""
            ### View collection statistics
            Check the current state of your document collection.
            """)
            
            refresh_btn = gr.Button("🔄 Refresh Stats", variant="secondary")
            stats_output = gr.Textbox(
                label="Collection Statistics",
                lines=10,
                interactive=False
            )
            
            refresh_btn.click(
                fn=get_collection_info,
                inputs=[],
                outputs=[stats_output]
            )
    
    gr.Markdown("""
    ---
    ### 💡 Tips
    - Ingest documents before querying
    - The pipeline uses gap analysis to iteratively improve retrieval
    - BM25 reranking and token budget enforcement ensure quality results
    """)


if __name__ == "__main__":
    logger.info("Starting BubbleHead RAG Testing UI...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=gr.themes.Soft()
    )
