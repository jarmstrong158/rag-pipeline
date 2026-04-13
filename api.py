"""
api.py — FastAPI server for the RAG pipeline.

Endpoints:
  POST /query          — ask a question, get an answer
  POST /query/chunks   — retrieve chunks only (no generation)
  GET  /health         — status + chunk count
  POST /ingest         — ingest a file or directory path

Usage:
  pip install fastapi uvicorn
  python api.py
  # or
  uvicorn api:app --reload
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "FastAPI and uvicorn are required to run the API server.\n"
        "Install with: pip install fastapi uvicorn"
    )

from core import chunker, embedder, generator, retriever, store


app = FastAPI(
    title="RAG Pipeline API",
    description="Local document question-answering via Ollama.",
    version="0.1.0",
)


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str
    top_k: int = 5
    min_score: float = 0.0
    model: Optional[str] = None
    temperature: float = 0.1


class ChunkResult(BaseModel):
    text: str
    source: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[ChunkResult]
    elapsed_ms: float


class ChunksResponse(BaseModel):
    chunks: list[ChunkResult]
    elapsed_ms: float


class IngestRequest(BaseModel):
    path: str
    chunk_size: int = 500
    overlap: int = 100
    recursive: bool = True


class IngestResponse(BaseModel):
    files_processed: int
    chunks_added: int
    total_chunks: int
    elapsed_ms: float


class HealthResponse(BaseModel):
    status: str
    chunk_count: int
    embedder_available: bool
    generator_available: bool


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        chunk_count=store.count(),
        embedder_available=embedder.is_available(),
        generator_available=generator.is_available(),
    )


@app.get("/sources")
def sources():
    """Return the list of unique source files in the vector store."""
    chunks, _ = store.load()
    from pathlib import Path
    files = sorted({Path(c.source).name for c in chunks})
    return {"sources": files, "total_chunks": len(chunks)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.time()

    # Embedder preflight
    if not embedder.is_available():
        raise HTTPException(503, "Embedder (Ollama nomic-embed-text) is not available.")

    # Load store
    chunks, vectors = store.load()
    if not chunks:
        raise HTTPException(404, "Vector store is empty. Ingest some documents first.")

    # Embed query + retrieve
    query_vector = embedder.embed(req.question)
    results = retriever.retrieve(
        query_vector, chunks, vectors, top_k=req.top_k, min_score=req.min_score
    )

    if not results:
        raise HTTPException(404, "No relevant chunks found for this query.")

    # Generate
    gen_model = req.model or generator.DEFAULT_MODEL
    if not generator.is_available(gen_model):
        raise HTTPException(503, f"Generator model '{gen_model}' is not available in Ollama.")

    answer = generator.generate(req.question, results, model=gen_model, temperature=req.temperature)

    return QueryResponse(
        answer=answer,
        sources=[
            ChunkResult(
                text=chunk.text[:500],
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=round(score, 4),
            )
            for chunk, score in results
        ],
        elapsed_ms=round((time.time() - t0) * 1000),
    )


@app.post("/query/chunks", response_model=ChunksResponse)
def query_chunks(req: QueryRequest):
    """Retrieve relevant chunks without generating an answer."""
    t0 = time.time()

    if not embedder.is_available():
        raise HTTPException(503, "Embedder (Ollama nomic-embed-text) is not available.")

    chunks, vectors = store.load()
    if not chunks:
        raise HTTPException(404, "Vector store is empty.")

    query_vector = embedder.embed(req.question)
    results = retriever.retrieve(
        query_vector, chunks, vectors, top_k=req.top_k, min_score=req.min_score
    )

    return ChunksResponse(
        chunks=[
            ChunkResult(
                text=chunk.text,
                source=chunk.source,
                chunk_index=chunk.chunk_index,
                score=round(score, 4),
            )
            for chunk, score in results
        ],
        elapsed_ms=round((time.time() - t0) * 1000),
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(req: IngestRequest):
    """Ingest a file or directory into the vector store."""
    t0 = time.time()

    if not embedder.is_available():
        raise HTTPException(503, "Embedder (Ollama nomic-embed-text) is not available.")

    path = Path(req.path.strip().strip('"\''))
    if not path.exists():
        raise HTTPException(404, f"Path does not exist: {req.path}")

    # Chunk
    if path.is_dir():
        chunks = chunker.chunk_directory(path, chunk_size=req.chunk_size, overlap=req.overlap, recursive=req.recursive)
    else:
        chunks = chunker.chunk_file(path, chunk_size=req.chunk_size, overlap=req.overlap)

    if not chunks:
        raise HTTPException(400, "No supported files found or all files were empty.")

    # Embed
    texts = [c.text for c in chunks]
    vectors = embedder.embed_batch(texts)

    # Merge + save
    import numpy as np
    existing_chunks, existing_vectors = store.load()
    all_chunks = existing_chunks + chunks
    if existing_vectors.size > 0:
        all_vectors = existing_vectors.tolist() + vectors
    else:
        all_vectors = vectors

    store.save(all_chunks, all_vectors)
    total = store.count()
    files = len({c.source for c in chunks})

    return IngestResponse(
        files_processed=files,
        chunks_added=len(chunks),
        total_chunks=total,
        elapsed_ms=round((time.time() - t0) * 1000),
    )


@app.post("/clear")
def clear():
    """Wipe the entire vector store."""
    store.clear()
    return {"success": True, "message": "Vector store cleared."}


# ── UI ────────────────────────────────────────────────────────────────────────

@app.get("/")
def serve_ui():
    from fastapi.responses import FileResponse
    ui = Path(__file__).parent / "ui.html"
    return FileResponse(ui)


# ── Dev server ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn, webbrowser, threading

    if not os.environ.get("RAG_NO_BROWSER"):
        def open_browser():
            import time as _t; _t.sleep(1)
            webbrowser.open("http://localhost:8000")
        threading.Thread(target=open_browser, daemon=True).start()

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
