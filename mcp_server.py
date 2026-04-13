#!/usr/bin/env python3
"""
RAG Pipeline MCP Server

Tools:
  rag_query    — ask a question, get an answer from your ingested documents
  rag_ingest   — ingest a file or directory into the vector store
  rag_status   — how many chunks are stored, is the model loaded
  rag_clear    — wipe the vector store

Add to Claude config:
  {
    "mcpServers": {
      "rag-pipeline": {
        "command": "python",
        "args": ["C:\\Users\\jarms\\repos\\rag-pipeline\\mcp_server.py"]
      }
    }
  }
"""

import sys
import os

# Ensure core modules are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rag-pipeline")


# ── Tools ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def rag_query(question: str, top_k: int = 5) -> str:
    """
    Ask a question and get an answer from your ingested documents.

    Args:
        question: The question to answer.
        top_k: Number of document chunks to retrieve (default 5).

    Returns:
        The answer followed by a list of sources.
    """
    from core import embedder, store, retriever, generator

    chunks, vectors = store.load()
    if not chunks:
        return "The vector store is empty. Run rag_ingest first."

    if not embedder.is_available():
        return "Embedding model not available. Check that the GGUF file exists at the configured path."

    query_vector = embedder.embed(question)
    results = retriever.retrieve(query_vector, chunks, vectors, top_k=top_k)

    if not results:
        return "No relevant documents found for that question."

    if not generator.is_available():
        # Retrieval-only fallback
        lines = [f"No generator available. Top {len(results)} relevant chunks:\n"]
        for i, (chunk, score) in enumerate(results, 1):
            fname = os.path.basename(chunk.source)
            lines.append(f"[{i}] {fname} (score: {score:.3f})")
            lines.append(chunk.text[:300])
            lines.append("")
        return "\n".join(lines)

    answer = generator.generate(question, results)

    sources = []
    for chunk, score in results:
        fname = os.path.basename(chunk.source)
        sources.append(f"  [{score:.3f}] {fname} (chunk {chunk.chunk_index})")

    return f"{answer}\n\nSources:\n" + "\n".join(sources)


@mcp.tool()
def rag_ingest(path: str, chunk_size: int = 500, overlap: int = 100) -> str:
    """
    Ingest a file or directory of documents into the vector store.

    Args:
        path: Absolute path to a file or directory.
        chunk_size: Characters per chunk (default 500).
        overlap: Overlap between chunks (default 100).

    Returns:
        Summary of what was ingested.
    """
    import numpy as np
    from pathlib import Path
    from core import chunker, embedder, store

    p = Path(path)
    if not p.exists():
        return f"Path does not exist: {path}"

    if not embedder.is_available():
        return "Embedding model not available. Check that the GGUF file exists."

    if p.is_dir():
        chunks = chunker.chunk_directory(p, chunk_size=chunk_size, overlap=overlap)
    else:
        chunks = chunker.chunk_file(p, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        return "No supported files found or all files were empty."

    texts = [c.text for c in chunks]
    vectors = embedder.embed_batch(texts)

    existing_chunks, existing_vectors = store.load()
    all_chunks = existing_chunks + chunks
    if existing_vectors.size > 0:
        all_vectors = existing_vectors.tolist() + vectors
    else:
        all_vectors = vectors

    store.save(all_chunks, all_vectors)
    total = store.count()
    files = len({c.source for c in chunks})

    return (
        f"Ingested {len(chunks)} chunks from {files} file(s).\n"
        f"Total chunks in store: {total}"
    )


@mcp.tool()
def rag_status() -> str:
    """
    Show the current status of the RAG pipeline.

    Returns:
        Chunk count, model availability, and store location.
    """
    from core import embedder, generator, store

    count = store.count()
    embed_ok = embedder.is_available()
    gen_ok = generator.is_available()

    lines = [
        f"Chunks in store: {count}",
        f"Embedder ready:  {'yes' if embed_ok else 'no — GGUF not found or llama-cpp-python not installed'}",
        f"Generator ready: {'yes' if gen_ok else 'no — GGUF not found or llama-cpp-python not installed'}",
    ]
    return "\n".join(lines)


@mcp.tool()
def rag_clear() -> str:
    """
    Wipe the vector store. All ingested documents will be removed.

    Returns:
        Confirmation message.
    """
    from core import store
    store.clear()
    return "Vector store cleared."


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run(transport="stdio")
