#!/usr/bin/env python3
"""
RAG Pipeline MCP Server

Tools:
  rag_query    — ask a question, get an answer from your ingested documents
  rag_ingest   — ingest a file or directory into the vector store
  rag_status   — how many chunks are stored, is the model loaded
  rag_clear    — wipe the vector store
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TOOLS = [
    {
        "name": "rag_query",
        "description": (
            "Ask a question and get an answer from your ingested documents. "
            "The pipeline embeds the question, finds the most relevant chunks, "
            "and generates a grounded answer using the local LLM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The question to answer"},
                "top_k": {"type": "integer", "description": "Number of chunks to retrieve (default 5)", "default": 5},
            },
            "required": ["question"],
        },
    },
    {
        "name": "rag_ingest",
        "description": (
            "Ingest a file or directory of documents into the vector store. "
            "Supported: .txt .md .py .js .json .yaml .toml .rst .pdf"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path to a file or directory"},
                "chunk_size": {"type": "integer", "description": "Characters per chunk (default 500)", "default": 500},
                "overlap": {"type": "integer", "description": "Overlap between chunks (default 100)", "default": 100},
            },
            "required": ["path"],
        },
    },
    {
        "name": "rag_status",
        "description": "Show how many chunks are stored and whether the models are ready.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "rag_clear",
        "description": "Wipe the entire vector store. All ingested documents will be removed.",
        "inputSchema": {"type": "object", "properties": {}},
    },
]


# ── Handlers ──────────────────────────────────────────────────────────────────

def _ensure_ui():
    """Start the API server if not running, then open the browser."""
    import subprocess, webbrowser, time, urllib.request
    try:
        urllib.request.urlopen("http://localhost:8000/health", timeout=2)
    except Exception:
        subprocess.Popen(
            [sys.executable, os.path.join(os.path.dirname(__file__), "api.py")],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(3)
    webbrowser.open("http://localhost:8000")


def handle_rag_query(params):
    _ensure_ui()
    from core import embedder, store, retriever, generator

    question = params.get("question", "").strip()
    if not question:
        return {"error": "question is required"}

    top_k = int(params.get("top_k", 5))

    chunks, vectors = store.load()
    if not chunks:
        return {"error": "Vector store is empty. Run rag_ingest first."}

    if not embedder.is_available():
        return {"error": "Embedding model not available. Check the GGUF file path."}

    query_vector = embedder.embed(question)
    results = retriever.retrieve(query_vector, chunks, vectors, top_k=top_k)

    if not results:
        return {"error": "No relevant documents found for that question."}

    if not generator.is_available():
        lines = [f"Generator not available. Top {len(results)} relevant chunks:\n"]
        for i, (chunk, score) in enumerate(results, 1):
            fname = os.path.basename(chunk.source)
            lines.append(f"[{i}] {fname} (score: {score:.3f})\n{chunk.text[:300]}\n")
        return {"answer": "\n".join(lines), "sources": []}

    answer = generator.generate(question, results)
    sources = [
        {
            "file": os.path.basename(chunk.source),
            "score": round(score, 4),
            "chunk_index": chunk.chunk_index,
        }
        for chunk, score in results
    ]
    return {"answer": answer, "sources": sources}


def handle_rag_ingest(params):
    import numpy as np
    from pathlib import Path
    from core import chunker, embedder, store

    path = params.get("path", "").strip()
    if not path:
        return {"error": "path is required"}

    p = Path(path)
    if not p.exists():
        return {"error": f"Path does not exist: {path}"}

    if not embedder.is_available():
        return {"error": "Embedding model not available. Check the GGUF file path."}

    chunk_size = int(params.get("chunk_size", 500))
    overlap = int(params.get("overlap", 100))

    if p.is_dir():
        chunks = chunker.chunk_directory(p, chunk_size=chunk_size, overlap=overlap)
    else:
        chunks = chunker.chunk_file(p, chunk_size=chunk_size, overlap=overlap)

    if not chunks:
        return {"error": "No supported files found or all files were empty."}

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

    return {
        "success": True,
        "files_ingested": files,
        "chunks_added": len(chunks),
        "total_chunks": total,
    }


def handle_rag_status(params):
    from core import embedder, generator, store

    return {
        "chunks_in_store": store.count(),
        "embedder_ready": embedder.is_available(),
        "generator_ready": generator.is_available(),
    }


def handle_rag_clear(params):
    from core import store
    store.clear()
    return {"success": True, "message": "Vector store cleared."}


HANDLERS = {
    "rag_query": handle_rag_query,
    "rag_ingest": handle_rag_ingest,
    "rag_status": handle_rag_status,
    "rag_clear": handle_rag_clear,
}


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue

        msg_id = msg.get("id")
        method = msg.get("method", "")
        params = msg.get("params", {})

        if method == "initialize":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "rag-pipeline", "version": "1.0.0"},
                },
            }
        elif method == "notifications/initialized":
            continue
        elif method == "tools/list":
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"tools": TOOLS},
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            tool_args = params.get("arguments", {})
            handler = HANDLERS.get(tool_name)

            if handler is None:
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps({"error": f"Unknown tool: {tool_name}"})}],
                        "isError": True,
                    },
                }
            else:
                try:
                    result = handler(tool_args)
                except Exception as e:
                    result = {"error": f"Tool '{tool_name}' failed: {e}"}
                response = {
                    "jsonrpc": "2.0",
                    "id": msg_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                    },
                }
        elif method.startswith("notifications/"):
            continue
        else:
            response = {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }

        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
