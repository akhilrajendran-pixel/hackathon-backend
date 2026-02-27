"""
FastAPI application — all route definitions, CORS, logging, concurrency control.
"""
import os
import sys
import time
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Ensure backend directory is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import drive_connector
import extractor
import chunker
import indexer
import retriever
import agent
import guardrails
import session_manager

# ── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("sales_copilot")

# ── App ─────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sales Co-Pilot API",
    description="AI-powered sales assistant with RAG over proposals, case studies & whitepapers",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:8501",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8501",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Concurrency semaphore for LLM calls
_llm_semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_LLM_CALLS)

# Ingestion state
_ingestion_state = {
    "last_ingestion": None,
    "documents_processed": 0,
    "total_chunks": 0,
    "details": [],
}


# ── Request / Response Models ───────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str = Field(..., description="Session identifier")
    query: str = Field(..., description="Natural language query")


class QueryResponse(BaseModel):
    session_id: str
    answer: Optional[str]
    citations: list
    confidence: Optional[str]
    confidence_score: Optional[float]
    intent: Optional[str]
    guardrail_triggered: Optional[dict]


class SessionResponse(BaseModel):
    session_id: str


class HistoryResponse(BaseModel):
    session_id: str
    turns: list


class IngestResponse(BaseModel):
    status: str
    documents_processed: int
    total_chunks: int
    details: list


class PipelineResponse(BaseModel):
    total_documents: int
    total_chunks: int
    documents: list
    last_ingestion: Optional[str]


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int


# ── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    """Simple health check."""
    return {
        "status": "ok",
        "indexed_chunks": indexer.get_chunk_count(),
    }


@app.post("/session/create", response_model=SessionResponse)
async def create_session():
    """Creates a new isolated conversation session."""
    session_id = session_manager.create_session()
    return {"session_id": session_id}


@app.get("/session/{session_id}/history", response_model=HistoryResponse)
async def get_session_history(session_id: str):
    """Returns full conversation history for a session."""
    session = session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return {
        "session_id": session_id,
        "turns": session["turns"],
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    """
    Triggers document ingestion from Google Drive (or local fallback).
    Downloads files, extracts text, chunks, embeds, and indexes.
    """
    start = time.time()
    logger.info("Starting document ingestion...")

    try:
        # 1. List files
        files = drive_connector.list_files()
        if not files:
            return {
                "status": "warning",
                "documents_processed": 0,
                "total_chunks": 0,
                "details": [],
            }

        # 2. Process each file
        all_chunks = []
        details = []
        drive_links = {}

        for file_meta in files:
            filename = file_meta["name"]
            logger.info("Processing: %s", filename)

            try:
                # Download
                file_bytes = drive_connector.download_file(file_meta)

                # Extract text
                doc = extractor.extract_text(file_bytes, filename)
                if not doc["pages"]:
                    logger.warning("No text extracted from %s", filename)
                    details.append({
                        "filename": filename,
                        "doc_type": "unknown",
                        "year": None,
                        "chunks": 0,
                        "status": "no_text",
                    })
                    continue

                # Chunk
                doc_chunks = chunker.chunk_document(doc)
                all_chunks.extend(doc_chunks)

                # Store drive link
                if file_meta.get("webViewLink"):
                    drive_links[filename] = file_meta["webViewLink"]

                # Record detail
                doc_type = doc_chunks[0]["doc_type"] if doc_chunks else "unknown"
                year = doc_chunks[0].get("year") if doc_chunks else None
                details.append({
                    "filename": filename,
                    "doc_type": doc_type,
                    "year": year,
                    "chunks": len(doc_chunks),
                    "status": "indexed",
                })

            except Exception as e:
                logger.error("Failed to process %s: %s", filename, e)
                details.append({
                    "filename": filename,
                    "doc_type": "unknown",
                    "year": None,
                    "chunks": 0,
                    "status": f"error: {str(e)}",
                })

        # 3. Build index
        if all_chunks:
            indexer.set_drive_links(drive_links)
            indexer.build_index(all_chunks)

        elapsed = time.time() - start
        logger.info(
            "Ingestion complete: %d docs, %d chunks in %.1fs",
            len(details), len(all_chunks), elapsed,
        )

        # Update state
        _ingestion_state.update({
            "last_ingestion": datetime.now(timezone.utc).isoformat(),
            "documents_processed": len([d for d in details if d["status"] == "indexed"]),
            "total_chunks": len(all_chunks),
            "details": details,
        })

        return {
            "status": "success",
            "documents_processed": _ingestion_state["documents_processed"],
            "total_chunks": len(all_chunks),
            "details": details,
        }

    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    """Main conversational query endpoint."""
    start = time.time()
    logger.info("Query [%s]: %s", req.session_id, req.query[:100])

    # Ensure session exists (auto-create if missing)
    session = session_manager.get_session(req.session_id)
    if session is None:
        session_manager.create_session.__wrapped__ if hasattr(session_manager.create_session, '__wrapped__') else None
        # Auto-create the session
        session_manager._sessions[req.session_id] = {
            "created_at": time.time(),
            "last_active": time.time(),
            "turns": [],
        }

    # Check if index is built
    if indexer.get_chunk_count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet. Please call POST /ingest first.",
        )

    # Use semaphore to limit concurrent LLM calls
    async with _llm_semaphore:
        result = await agent.process_query(req.session_id, req.query)

    elapsed = time.time() - start
    logger.info(
        "Query completed [%s] intent=%s confidence=%s time=%.2fs",
        req.session_id,
        result.get("intent"),
        result.get("confidence"),
        elapsed,
    )

    return result


@app.get("/admin/pipeline", response_model=PipelineResponse)
async def admin_pipeline():
    """Returns ingestion pipeline stats for admin dashboard."""
    all_chunks = indexer.get_all_chunks()

    # Build per-document summary
    doc_map = {}
    for chunk in all_chunks:
        fn = chunk["filename"]
        if fn not in doc_map:
            doc_map[fn] = {
                "filename": fn,
                "doc_type": chunk["doc_type"],
                "year": chunk.get("year"),
                "chunks": 0,
                "regions": list(chunk.get("regions", [])),
                "status": "indexed",
            }
        doc_map[fn]["chunks"] += 1

    return {
        "total_documents": len(doc_map),
        "total_chunks": len(all_chunks),
        "documents": list(doc_map.values()),
        "last_ingestion": _ingestion_state.get("last_ingestion"),
    }


# ── Startup ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
