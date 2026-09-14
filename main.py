"""
main.py - BubbleHead RAG CLI

Command-line interface for document ingestion and querying.
"""

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List

from ingestion.Chunker import chunk_document
from ingestion.Embedder import embed_and_store
from ingestion.parsers.Parser import parse
from pipeline.pipeline import run

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".pptx", ".txt", ".html", ".csv"}


def find_supported_files(data_dir: str) -> List[Path]:
    """
    Recursively find all supported files in the data directory.

    Args:
        data_dir: Path to directory containing documents

    Returns:
        List of supported document paths
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        logging.error("Data directory does not exist: %s", data_dir)
        return []

    supported_files = []
    for ext in SUPPORTED_EXTENSIONS:
        supported_files.extend(data_path.rglob(f"*{ext}"))

    return sorted(supported_files)


def ingest_command(data_dir: str) -> None:
    """
    Ingest documents from a directory into ChromaDB.

    Args:
        data_dir: Path to directory containing documents
    """
    logging.info("Starting ingestion from: %s", data_dir)
    files = find_supported_files(data_dir)

    if not files:
        logging.warning("No supported files found in %s", data_dir)
        print(f"No supported files found in {data_dir}")
        print(f"Supported extensions: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
        return

    logging.info("Found %d files to process", len(files))
    print(f"Found {len(files)} files to process")

    total_chunks = 0
    processed_files = 0
    failed_files = 0

    @dataclass
    class ChunkPayload:
        text: str
        metadata: dict

    for filepath in files:
        try:
            logging.info("Processing: %s", filepath)
            print(f"Processing: {filepath.name}")

            parsed_sections = parse(str(filepath))
            if not parsed_sections:
                logging.warning("No text extracted from %s", filepath)
                continue

            chunks = []
            for section in parsed_sections:
                section_text = (section.get("text") or "").strip()
                if not section_text:
                    continue

                section_chunks = chunk_document(section_text)
                for section_chunk in section_chunks:
                    section_chunk.setdefault("metadata", {})
                    section_chunk["metadata"].update(
                        {
                            "source_file": filepath.name,
                            "chunk_index": len(chunks),
                            "page_number": section.get("page", 0) or 0,
                            "section_heading": section.get("section_heading", "") or "",
                            "document_type": section.get("doc_type")
                            or filepath.suffix.lstrip("."),
                        }
                    )
                    chunks.append(section_chunk)

            if not chunks:
                logging.warning("No chunks created from %s", filepath)
                continue

            chunk_payloads = [
                ChunkPayload(text=chunk["text"], metadata=chunk.get("metadata", {}))
                for chunk in chunks
            ]
            stats = embed_and_store(chunk_payloads)

            total_chunks += stats.get("stored", 0)
            processed_files += 1
            print(f"  [OK] Stored {stats.get('stored', 0)} chunks")

        except Exception as exc:
            logging.error("Failed to process %s: %s", filepath, exc)
            print(f"  [FAIL] {exc}")
            failed_files += 1

    print("\n" + "=" * 60)
    print("INGESTION SUMMARY")
    print("=" * 60)
    print(f"Files processed: {processed_files}/{len(files)}")
    print(f"Files failed: {failed_files}")
    print(f"Total chunks stored: {total_chunks}")
    print("=" * 60)

    logging.info(
        "Ingestion complete: %d files processed, %d chunks stored, %d failed",
        processed_files,
        total_chunks,
        failed_files,
    )


def query_command(question: str) -> None:
    """
    Query the RAG pipeline with a question.

    Args:
        question: User question
    """
    logging.info("Processing query: %s", question)

    try:
        answer = run(question)

        print("\n" + "=" * 60)
        print("QUESTION:")
        print(question)
        print("\n" + "-" * 60)
        print("ANSWER:")
        print(answer)
        print("=" * 60 + "\n")

    except Exception as exc:
        logging.error("Query failed: %s", exc)
        print(f"Error: {exc}")


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="BubbleHead RAG - Document ingestion and querying system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents from a directory",
    )
    ingest_parser.add_argument(
        "data_dir",
        type=str,
        default="./data",
        nargs="?",
        help="Path to directory containing documents (default: ./data)",
    )

    query_parser = subparsers.add_parser(
        "query",
        help="Query the RAG system",
    )
    query_parser.add_argument(
        "question",
        type=str,
        help="Question to ask the RAG system",
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        return

    if args.command == "ingest":
        ingest_command(args.data_dir)
    elif args.command == "query":
        query_command(args.question)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
