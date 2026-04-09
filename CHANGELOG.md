# Changelog

All notable changes to BubbleHead RAG will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-04-08

### Added
- Initial release of BubbleHead RAG pipeline
- ChromaDB vector store integration
- BM25 reranking for improved retrieval
- Token budget enforcement with tiktoken
- Gap analysis agent with iterative refinement
- LangGraph-based pipeline orchestration
- Multi-format document parsing (PDF, DOCX, PPTX, TXT, HTML, CSV)
- Gradio UI for document ingestion and querying
- CLI interface for batch processing
- GPU acceleration support via Ollama
- Comprehensive documentation and examples

### Features
- **Retrieval**: Vector similarity + BM25 hybrid search
- **Generation**: Structured answer generation with citation
- **Gap Analysis**: Confidence-based quality evaluation
- **Pipeline**: retrieve → generate → gap_analysis with retry loop
- **UI**: Three-tab interface (Ingest, Query, Collection Info)

### Configuration
- Configurable models (embedding and LLM)
- Adjustable retrieval parameters
- Tunable gap analysis thresholds
- Token budget management

### Documentation
- README with installation and usage instructions
- CONTRIBUTING guidelines
- LICENSE (MIT)
- Architecture documentation
- API documentation

## [Unreleased]

### Planned
- Additional reranking algorithms (Cohere, Cross-encoder)
- Streaming response support
- Multi-query retrieval
- Document metadata filtering
- Query history and caching
- Batch query processing
- REST API endpoint
- Docker containerization
- Kubernetes deployment configs
