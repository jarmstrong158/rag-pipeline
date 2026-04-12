"""
Embedder — convert text to vectors via Ollama (nomic-embed-text).

Requires Ollama running at localhost:11434 with nomic-embed-text pulled.
  ollama pull nomic-embed-text
"""

from __future__ import annotations

import json
import urllib.request


OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "nomic-embed-text"


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


def embed(text: str, model: str = DEFAULT_MODEL) -> list[float]:
    """Embed a single string. Returns a float vector."""
    payload = json.dumps({"model": model, "input": text}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["embeddings"][0]


def embed_batch(texts: list[str], model: str = DEFAULT_MODEL) -> list[list[float]]:
    """Embed a list of strings. Returns a list of float vectors."""
    payload = json.dumps({"model": model, "input": texts}).encode()
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    return data["embeddings"]
