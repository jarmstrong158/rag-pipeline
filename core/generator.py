"""
Generator — send retrieved chunks + query to llama3.2 GGUF and return an answer.

No Ollama required. Uses the GGUF file directly via llama-cpp-python.

  pip install llama-cpp-python
"""

from __future__ import annotations

from pathlib import Path

from core.chunker import Chunk


MODEL_PATH = r"C:\Users\jarms\repos\ollama\Llama-3.2-3B-Instruct-Q4_K_M.gguf"
DEFAULT_MODEL = MODEL_PATH

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions based on the provided context documents.
Only use the information in the context to answer. If the context doesn't contain enough
information to answer the question, say so clearly.
Do not make up facts. Be concise and direct.\
"""

_model = None


def _get_model(model_path: str = MODEL_PATH):
    global _model
    if _model is None:
        from llama_cpp import Llama
        _model = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_batch=512,
            verbose=False,
        )
    return _model


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
        model: Path to GGUF model file (default: llama3.2 Q4_K_M).
        temperature: Lower = more factual. Default 0.1 for RAG.
        stream: If True, print tokens as they stream. Returns full response.

    Returns:
        The model's answer as a string.
    """
    m = _get_model(model)
    prompt = build_prompt(query, chunks)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]

    if stream:
        response = m.create_chat_completion(
            messages=messages,
            temperature=temperature,
            stream=True,
        )
        full_response = []
        for chunk in response:
            token = chunk["choices"][0]["delta"].get("content", "")
            if token:
                print(token, end="", flush=True)
                full_response.append(token)
        print()
        return "".join(full_response).strip()
    else:
        response = m.create_chat_completion(
            messages=messages,
            temperature=temperature,
            stream=False,
        )
        return response["choices"][0]["message"]["content"].strip()


def is_available(model: str = MODEL_PATH) -> bool:
    """Return True if the GGUF file exists and llama-cpp-python is installed."""
    if not Path(model).exists():
        return False
    from importlib.util import find_spec
    return find_spec("llama_cpp") is not None


def require_available(model: str = MODEL_PATH) -> None:
    """Raise a clear error if the model or library isn't ready."""
    if not Path(model).exists():
        raise RuntimeError(
            f"Generator model not found at: {model}\n"
            "Download it from: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF"
        )
    try:
        import llama_cpp  # noqa
    except ImportError:
        raise RuntimeError(
            "llama-cpp-python is not installed.\n"
            "Install it with: pip install llama-cpp-python"
        )
