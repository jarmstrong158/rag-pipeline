"""
Embedder — convert text to vectors via llama-cpp-python (nomic-embed-text GGUF).

No Ollama required. Uses the GGUF file directly.

  pip install llama-cpp-python
"""

from __future__ import annotations

from pathlib import Path

MODEL_PATH = r"C:\Users\jarms\repos\ollama\nomic-embed-text-v1.5.Q4_K_M.gguf"

_model = None


def _get_model():
    global _model
    if _model is None:
        import os, sys
        from llama_cpp import Llama
        # Suppress llama.cpp stderr noise
        devnull = open(os.devnull, 'w')
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            _model = Llama(
                model_path=MODEL_PATH,
                embedding=True,
                n_ctx=2048,
                n_batch=512,
                verbose=False,
            )
        finally:
            sys.stderr = old_stderr
    return _model


def is_available(model: str = MODEL_PATH) -> bool:
    """Return True if the GGUF file exists and llama-cpp-python is installed."""
    if not Path(MODEL_PATH).exists():
        return False
    try:
        import llama_cpp  # noqa
        return True
    except ImportError:
        return False


def require_available(model: str = MODEL_PATH) -> None:
    """Raise a clear error if the model or library isn't ready."""
    if not Path(MODEL_PATH).exists():
        raise RuntimeError(
            f"Embedding model not found at: {MODEL_PATH}\n"
            "Download it from: https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF"
        )
    try:
        import llama_cpp  # noqa
    except ImportError:
        raise RuntimeError(
            "llama-cpp-python is not installed.\n"
            "Install it with: pip install llama-cpp-python"
        )


def embed(text: str, model: str = MODEL_PATH) -> list[float]:
    """Embed a single string. Returns a float vector."""
    m = _get_model()
    result = m.embed(text)
    # llama-cpp-python returns list[float] or list[list[float]]
    if isinstance(result[0], list):
        return result[0]
    return result


def embed_batch(texts: list[str], model: str = MODEL_PATH) -> list[list[float]]:
    """Embed a list of strings. Returns a list of float vectors."""
    results = []
    total = len(texts)
    for i, text in enumerate(texts, 1):
        print(f"\r  Embedding {i}/{total}...", end="", flush=True)
        vec = embed(text)
        results.append(vec)
    print()
    return results
