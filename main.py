"""
main.py - BubbleHead RAG CLI

Command-line interface for document ingestion and querying.
"""

import argparse
import logging
from pathlib import Path
from typing import List

from ingestion.parsers.Parser import parse_file
from ingestion.Chunker import chunk_document
from ingestion.Embedder import embed_and_store
from pipeline.pipeline import run


# Supported file extensions
SUPPORTED_EXTENSIONS = {'.pdf', '.docx', '.pptx', '.txt', '.html', '.csv'}


def find_supported_files(data_dir: str) -> List[Path]:
    """
    Recursively find all supported files in the data directory.
    
    Args:
        data_dir: Path to directory containing documents
        
    Returns:
        List of Path objects for supported files
    """
    data_path = Path(data_dir)
    
    if not data_path.exists():
        logging.error("Data directory does not exist: %s", data_dir)
        return []
    
    supported_files = []
    for ext in SUPPORTED_EXTENSIONS:
        supported_files.extend(data_path.rglob(f'*{ext}'))
    
    return sorted(supported_files)


def ingest_command(data_dir: str) -> None:
    """
    Ingest documents from the data directory.
    
    Pipeline:
    1. Find all supported files recursively
    2. Parse each file
    3. Chunk the parsed text
    4. Embed and store chunks in ChromaDB
    
    Args:
        data_dir: Path to directory containing documents
    """
    logging.info("Starting ingestion from: %s", data_dir)
    
    # Find all supported files
    files = find_supported_files(data_dir)
    
    if not files:
        logging.warning("No supported files found in %s", data_dir)
        print(f"No supported files found in {data_dir}")
        print(f"Supported extensions: {', '.join(SUPPORTED_EXTENSIONS)}")
        return
    
    logging.info("Found %d files to process", len(files))
    print(f"Found {len(files)} files to process")
    
    # Process each file
    total_chunks = 0
    processed_files = 0
    failed_files = 0
    
    for filepath in files:
        try:
            logging.info("Processing: %s", filepath)
            print(f"Processing: {filepath.name}")
            
            # Parse file
            parsed_text = parse_file(str(filepath))
            
            if not parsed_text or not parsed_text.strip():
                logging.warning("No text extracted from %s", filepath)
                continue
            
            # Chunk document
            chunks = chunk_document(parsed_text, source_file=str(filepath.name))
            
            if not chunks:
                logging.warning("No chunks created from %s", filepath)
                continue
            
            # Embed and store
            stats = embed_and_store(chunks)
            
            total_chunks += stats.get('stored', 0)
            processed_files += 1
            
            print(f"  ✓ Stored {stats.get('stored', 0)} chunks")
            
        except Exception as e:
            logging.error("Failed to process %s: %s", filepath, e)
            print(f"  ✗ Failed: {e}")
            failed_files += 1
    
    # Print summary
    print("\n" + "="*60)
    print("INGESTION SUMMARY")
    print("="*60)
    print(f"Files processed: {processed_files}/{len(files)}")
    print(f"Files failed: {failed_files}")
    print(f"Total chunks stored: {total_chunks}")
    print("="*60)
    
    logging.info(
        "Ingestion complete: %d files processed, %d chunks stored, %d failed",
        processed_files, total_chunks, failed_files
    )


def query_command(question: str) -> None:
    """
    Query the RAG pipeline with a question.
    
    Args:
        question: User's question
    """
    logging.info("Processing query: %s", question)
    
    try:
        # Run the RAG pipeline
        answer = run(question)
        
        # Print answer
        print("\n" + "="*60)
        print("QUESTION:")
        print(question)
        print("\n" + "-"*60)
        print("ANSWER:")
        print(answer)
        print("="*60 + "\n")
        
    except Exception as e:
        logging.error("Query failed: %s", e)
        print(f"Error: {e}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BubbleHead RAG - Document ingestion and querying system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Ingest command
    ingest_parser = subparsers.add_parser(
        'ingest',
        help='Ingest documents from a directory'
    )
    ingest_parser.add_argument(
        'data_dir',
        type=str,
        default='./data',
        nargs='?',
        help='Path to directory containing documents (default: ./data)'
    )
    
    # Query command
    query_parser = subparsers.add_parser(
        'query',
        help='Query the RAG system'
    )
    query_parser.add_argument(
        'question',
        type=str,
        help='Question to ask the RAG system'
    )
    
    args = parser.parse_args()
    
    # Show help if no command provided
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'ingest':
        ingest_command(args.data_dir)
    elif args.command == 'query':
        query_command(args.question)


if __name__ == '__main__':
    # Set up logging (entry point - correct place for basicConfig)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
