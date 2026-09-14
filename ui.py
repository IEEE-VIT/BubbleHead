"""
ui.py - BubbleHead Research Analysis Interface

FastAPI backend that serves the vanilla HTML/CSS/JS frontend
and exposes REST API endpoints for the RAG pipeline.
"""

import logging
import os
import tempfile
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import MAX_UPLOAD_MB

# ── Lazy pipeline loader ───────────────────────────────────────────────────────
# Heavy imports (chromadb, pymupdf, langchain, ollama) are deferred to a
# background thread so the server starts and serves the frontend immediately.

_ready = False  # True once all pipeline modules are loaded
_load_err: str | None = None

_parse_file = None
_chunk_document = None
_embed_and_store = None
_collection_stats = None
_run = None


def _load_pipeline():
    global _ready, _load_err
    global _parse_file, _chunk_document, _embed_and_store, _collection_stats, _run
    try:
        logger.info("Loading pipeline modules in background…")
        from ingestion.parsers.Parser import parse as _pf
        from ingestion.Chunker import chunk_document as _cd
        from ingestion.Embedder import embed_and_store as _es, collection_stats as _cs
        from pipeline.pipeline import run as _r

        _parse_file = _pf
        _chunk_document = _cd
        _embed_and_store = _es
        _collection_stats = _cs
        _run = _r
        _ready = True
        logger.info("Pipeline ready.")
    except Exception as exc:
        _load_err = str(exc)
        logger.error("Pipeline failed to load: %s", exc)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_load_pipeline, daemon=True).start()
    yield


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ── App ────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="BubbleHead Research Analysis API",
    description="RAG pipeline API: ingest documents, run queries, inspect collection.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

_WARMING_UP = JSONResponse(
    {
        "success": False,
        "message": "Pipeline is warming up, please try again in a few seconds.",
    },
    status_code=503,
)


# ── Routes ─────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("frontend/index.html")


@app.get("/api/status")
async def status():
    """Returns whether the pipeline has finished loading."""
    return JSONResponse({"ready": _ready, "error": _load_err})


@app.get("/api/config.js", include_in_schema=False)
async def frontend_config():
    """Share the upload limit with the browser before it handles file selections."""
    return Response(
        f"const MAX_UPLOAD_MB = {MAX_UPLOAD_MB};",
        media_type="application/javascript",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/api/ingest")
async def ingest(file: UploadFile = File(...)):
    if not _ready:
        return _WARMING_UP

    if not file or not file.filename:
        return JSONResponse({"success": False, "message": "No file provided."})

    suffix = Path(file.filename).suffix.lower()
    allowed = {".pdf", ".docx", ".pptx", ".txt", ".md", ".html", ".csv"}

    if suffix not in allowed:
        return JSONResponse(
            {
                "success": False,
                "message": f"Unsupported file type '{suffix}'. Allowed: {', '.join(sorted(allowed))}",
            }
        )

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(tmp_fd, "wb") as fh:
            # One extra byte detects oversized uploads without reading them in full.
            max_bytes = MAX_UPLOAD_MB * 1024 * 1024
            content = await file.read(max_bytes + 1)
            if len(content) > max_bytes:
                return JSONResponse(
                    {
                        "success": False,
                        "message": f"File too large (max {MAX_UPLOAD_MB}MB).",
                    },
                    status_code=413,
                )
            fh.write(content)

        file_name = file.filename
        logger.info("Ingesting: %s", file_name)

        parsed_sections = _parse_file(tmp_path)
        if not parsed_sections:
            return JSONResponse(
                {
                    "success": False,
                    "message": f"No content could be extracted from '{file_name}'.",
                }
            )

        combined_text = "\n\n".join(s["text"] for s in parsed_sections if s.get("text"))
        if not combined_text.strip():
            return JSONResponse(
                {"success": False, "message": f"No readable text in '{file_name}'."}
            )

        chunks = _chunk_document(combined_text)
        if not chunks:
            return JSONResponse(
                {
                    "success": False,
                    "message": f"Chunking produced no output for '{file_name}'.",
                }
            )

        for i, chunk in enumerate(chunks):
            chunk.setdefault("metadata", {})
            chunk["metadata"].update(
                {
                    "source_file": file_name,
                    "chunk_index": i,
                    "page_number": 0,
                    "section_heading": "",
                    "document_type": suffix.lstrip("."),
                }
            )

        @dataclass
        class ChunkObj:
            text: str
            metadata: dict

        chunk_objects = [
            ChunkObj(text=c["text"], metadata=c.get("metadata", {})) for c in chunks
        ]
        stats = _embed_and_store(chunk_objects)

        logger.info(
            "Ingested '%s': stored=%d, rejected=%d",
            file_name,
            stats.get("stored", 0),
            stats.get("rejected", 0),
        )

        return JSONResponse(
            {
                "success": True,
                "file_name": file_name,
                "stored": stats.get("stored", 0),
                "rejected": stats.get("rejected", 0),
                "avg_tokens": round(stats.get("avg_tokens", 0)),
                "max_tokens": stats.get("max_tokens", 0),
                "collection_total": stats.get("collection_total", 0),
            }
        )

    except Exception as exc:
        logger.exception("Ingestion failed for '%s'", file.filename)
        return JSONResponse({"success": False, "message": str(exc)}, status_code=500)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


class QueryRequest(BaseModel):
    question: str


@app.post("/api/query")
async def query(req: QueryRequest):
    if not _ready:
        return _WARMING_UP

    question = (req.question or "").strip()
    if not question:
        return JSONResponse({"success": False, "message": "Question cannot be empty."})

    try:
        logger.info("Running query: %s", question)
        answer = _run(question)
        stats = _collection_stats()
        count = stats.get("count", 0)

        return JSONResponse(
            {
                "success": True,
                "answer": answer,
                "collection_count": count,
                "retrieval_strategy": "Hybrid (Vector + BM25)",
                "gap_analysis": "Active",
                "collection_status": (
                    "Ready" if count > 0 else "Empty — ingest documents first"
                ),
            }
        )

    except Exception as exc:
        logger.exception("Query failed")
        return JSONResponse({"success": False, "message": str(exc)}, status_code=500)


@app.get("/api/collection")
async def get_collection():
    if not _ready:
        return JSONResponse(
            {"success": True, "name": "—", "count": 0, "status": "Pipeline warming up…"}
        )
    try:
        stats = _collection_stats()
        if "error" in stats:
            return JSONResponse({"success": False, "message": stats["error"]})
        count = stats.get("count", 0)
        return JSONResponse(
            {
                "success": True,
                "name": stats.get("name", "N/A"),
                "count": count,
                "status": (
                    "Ready for queries"
                    if count > 0
                    else "Empty — ingest documents first"
                ),
            }
        )
    except Exception as exc:
        logger.exception("Collection info failed")
        return JSONResponse({"success": False, "message": str(exc)}, status_code=500)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info(
        "Starting BubbleHead Research Analysis Interface on http://localhost:7860"
    )
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=7860,
        log_level="info",
    )
