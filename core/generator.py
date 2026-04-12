"""
Generator — send retrieved chunks + query to Ollama (llama3.2) and return an answer.

Requires Ollama running at localhost:11434 with llama3.2 pulled.
  ollama pull llama3.2
"""

from __future__ import annotations

import json
import urllib.request

from core.chunker import Chunk


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on the provided context documents.
Only use the information in the context to answer. If the context doesn't contain enough
information to answer the question, say so clearly.
Do not make up facts. Be concise and direct.\
"""


def build_prompt(query: str, chunks: list[tuple[Chunk, float]]) -> str:
    """Build the prompt with retrieved context injected."""
    context_parts = []
    for i, (chunk, score) in enumerate(chunks, 1):
        context_parts.append(
            f"[Source {i}: {chunk.source} (relevance: {score:.2f})]\n{chunk.text}"
        )

    context = "\n\n---\n\n".join(context_parts)
    return f"Context:\n{context}\n\nQuestion: {query}"


def generate(
    query: str,
    chunks: list[tuple[Chunk, float]],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    stream: bool = False,
) -> str:
    """
    Generate an answer for the query using the retrieved chunks as context.

    Args:
        query: The user's question.
        chunks: List of (Chunk, score) from the retriever.
        model: Ollama model to use.
        temperature: Lower = more factual. Default 0.1 for RAG.
        stream: If True, print tokens as they stream. Returns full response.

    Returns:
        The model's answer as a string.
    """
    prompt = build_prompt(query, chunks)

    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "options": {"temperature": temperature},
        "stream": stream,
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    if stream:
        return _stream_response(req)
    else:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
        return data["message"]["content"].strip()


def _stream_response(req: urllib.request.Request) -> str:
    """Stream tokens to stdout and return the full assembled response."""
    full_response = []
    with urllib.request.urlopen(req, timeout=120) as resp:
        for line in resp:
            if not line.strip():
                continue
            try:
                chunk_data = json.loads(line)
                token = chunk_data.get("message", {}).get("content", "")
                if token:
                    print(token, end="", flush=True)
                    full_response.append(token)
                if chunk_data.get("done"):
                    break
            except json.JSONDecodeError:
                continue
    print()  # newline after streaming
    return "".join(full_response).strip()


def is_available(model: str = DEFAULT_MODEL) -> bool:
    """Return True if Ollama is running and the model is available."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
        models = [m["name"].split(":")[0] for m in data.get("models", [])]
        return model.split(":")[0] in models
    except Exception:
        return False


def require_available(model: str = DEFAULT_MODEL) -> None:
    """Raise a clear error if Ollama or the model isn't ready."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            data = json.loads(resp.read())
    except Exception:
        raise RuntimeError(
            "Ollama is not running. Start it with: ollama serve\n"
            f"Then pull the model: ollama pull {model}"
        )
    models = [m["name"].split(":")[0] for m in data.get("models", [])]
    if model.split(":")[0] not in models:
        raise RuntimeError(
            f"Model '{model}' not found in Ollama.\n"
            f"Pull it with: ollama pull {model}"
        )
